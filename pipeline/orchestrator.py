"""Drives the end-to-end pipeline: read input words -> generate -> validate ->
dedup -> judge -> accumulate accepted sentences -> backfill retries -> export.

Designed to be safely interruptible and resumable at 15k+ word scale: state is
checkpointed to disk after every phase (pipeline/state.py), and only words
still short of their passing-sentence quota are re-processed on a re-run.

Both generation and judging are provider-switchable (agents/generator.py and
agents/judge.py each dispatch on config.yaml's llm.<agent>.provider) and each
run as a pool of up to <agent>_max_concurrent_jobs batch jobs at a time,
refilled from a queue as jobs finish -- sized so the pool's *combined*
reserved tokens stay under whichever provider's org-wide quota applies
(OpenAI enforces one; Gemini hasn't hit one yet but the same safety margin is
kept regardless of provider, since either agent may run on either provider).
Both providers' credit balances have run dry at different points during this
project, which is exactly the scenario this design exists to survive.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
from dotenv import load_dotenv

import agents.generator as generator
import agents.judge as judge
import embedding.model as embedding_model_mod
import embedding.similarity as similarity
import pipeline.batching as batching
import pipeline.validation as validation
import storage.excel as excel_storage
import storage.jsonl as jsonl_storage
from pipeline.state import PipelineState

logger = logging.getLogger("pipeline.orchestrator")


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(config: dict) -> None:
    log_cfg = config.get("logging", {})
    logs_dir = config["paths"]["logs_dir"]
    os.makedirs(logs_dir, exist_ok=True)
    level = getattr(logging, str(log_cfg.get("level", "INFO")).upper(), logging.INFO)
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(logs_dir, log_cfg.get("file", "pipeline.log")), encoding="utf-8"),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def _renumber_sentence_ids(row_id: int, sentences: list[dict], state: PipelineState) -> None:
    base = state.next_sentence_suffix(row_id)
    for i, s in enumerate(sentences):
        s["sentence_id"] = f"{row_id}_{base + i}"


def _process_generation_results(
    batch: list[dict],
    gen_results: list[dict],
    state: PipelineState,
    paths: dict,
    embedding_model: str,
    similarity_threshold: float,
) -> list[dict]:
    """Structural validation + near-dup filtering for one generator-batch's
    already-fetched results. Returns a "chunk": the judge-ready record list
    for whichever of this batch's words still have surviving candidates
    (possibly empty)."""
    jsonl_storage.append_jsonl(paths["raw"], gen_results)

    success = [g for g in gen_results if isinstance(g, dict) and g.get("sentences")]
    failed = [g for g in gen_results if not (isinstance(g, dict) and g.get("sentences"))]
    if failed:
        jsonl_storage.append_jsonl(paths["rejected"], [{**f, "stage": "generation"} for f in failed])
    if len(gen_results) < len(batch):
        missing_ids = {r["row_id"] for r in batch} - {g.get("row_id") for g in gen_results}
        jsonl_storage.append_jsonl(paths["rejected"], [
            {"row_id": rid, "status": "FAILED", "stage": "generation", "reason": "no batch response for this word"}
            for rid in missing_ids
        ])

    per_word_candidates: dict[int, dict] = {}
    for g in success:
        row_id = g["row_id"]
        word = g.get("word", "")
        sentences = g.get("sentences", [])
        _renumber_sentence_ids(row_id, sentences, state)

        kept_sentences = []
        for s in sentences:
            ok, reason = validation.validate_generated_sentence(s, word)
            if ok:
                kept_sentences.append(s)
            else:
                jsonl_storage.append_jsonl(paths["rejected"], [{
                    "row_id": row_id,
                    "sentence_id": s.get("sentence_id"),
                    "sentence": s.get("sentence"),
                    "stage": "structural_validation",
                    "reason": reason,
                }])
        if kept_sentences:
            per_word_candidates[row_id] = {
                "domain": g.get("domain"),
                "category": g.get("category"),
                "word": word,
                "sentences": kept_sentences,
            }

    # Batch-embed each word's already-accepted history in one call (empty on round 1).
    history_texts: list[str] = []
    history_spans: dict[int, tuple[int, int]] = {}
    for row_id in per_word_candidates:
        existing = [a["sentence"] for a in state.accepted_sentences(row_id)]
        if existing:
            start = len(history_texts)
            history_texts.extend(existing)
            history_spans[row_id] = (start, start + len(existing))
    history_embeddings = (
        embedding_model_mod.embed_texts(history_texts, model=embedding_model) if history_texts else []
    )

    chunk = []
    for row_id, payload in per_word_candidates.items():
        span = history_spans.get(row_id)
        existing_embeddings = history_embeddings[span[0]: span[1]] if span else []
        kept, rejected = similarity.filter_near_duplicates(
            payload["sentences"], existing_embeddings, similarity_threshold, embedding_model
        )
        if rejected:
            jsonl_storage.append_jsonl(paths["rejected"], [{
                "row_id": row_id,
                "sentence_id": r.get("sentence_id"),
                "sentence": r.get("sentence"),
                "stage": "near_duplicate",
                "reason": r["reason"],
            } for r in rejected])
        if kept:
            chunk.append({
                "row_id": row_id,
                "domain": payload["domain"],
                "category": payload["category"],
                "word": payload["word"],
                "sentences": [{k: v for k, v in s.items() if k != "_embedding"} for s in kept],
            })

    return chunk


def _apply_judge_results(
    chunks: list[list[dict]],
    results_by_custom_id: dict[str, list[dict]],
    state: PipelineState,
    paths: dict,
    thresholds: dict,
    min_passing: int,
    max_rounds: int,
) -> None:
    """Applies judge results and immediately finalizes the status of every
    word touched, chunk by chunk, rather than waiting for an entire round/
    queue to finish -- so a word with enough accepted sentences shows as
    "complete" as soon as ITS chunk is judged, not only once every other
    chunk in the batch has also finished (which could be many hours later)."""
    for i, chunk in enumerate(chunks):
        judge_results = results_by_custom_id.get(f"chunk-{i}")
        row_ids_in_chunk = [item["row_id"] for item in chunk]
        if judge_results is None:
            logger.error("no judge result for chunk %d (%d word(s)); leaving pending for another round", i, len(chunk))
            continue
        jsonl_storage.append_jsonl(paths["processed"], judge_results)

        payload_by_row_id = {item["row_id"]: item for item in chunk}
        for jr in judge_results:
            row_id = jr.get("row_id")
            payload = payload_by_row_id.get(row_id)
            if payload is None:
                continue
            sentences_by_id = {s["sentence_id"]: s for s in payload["sentences"]}
            accepted_this_round = []
            for res in jr.get("results", []):
                sid = res.get("sentence_id")
                cand = sentences_by_id.get(sid)
                if cand is None:
                    continue
                decision, overall_score, reason = validation.compute_decision(res, thresholds)
                if decision == "PASS":
                    accepted_this_round.append({
                        "row_id": row_id,
                        "sentence_id": sid,
                        "domain": payload["domain"],
                        "category": payload["category"],
                        "word": payload["word"],
                        "sentence": cand["sentence"],
                        "context": cand.get("context", ""),
                        "overall_score": round(overall_score, 4),
                    })
                else:
                    jsonl_storage.append_jsonl(paths["rejected"], [{
                        "row_id": row_id,
                        "sentence_id": sid,
                        "sentence": cand["sentence"],
                        "stage": "judge",
                        "reason": reason,
                    }])
            if accepted_this_round:
                state.add_accepted(row_id, accepted_this_round)
        for row_id in row_ids_in_chunk:
            state.finalize_status(row_id, min_passing, max_rounds)
    state.save()


def _drain_judge_queue(
    config: dict,
    prompts_dir: str,
    state: PipelineState,
    paths: dict,
    thresholds: dict,
    min_passing: int,
    max_rounds: int,
    max_concurrent: int = 1,
    poll_interval_seconds: int = 30,
) -> None:
    """Runs a pool of up to `max_concurrent` judge batch jobs at once,
    refilling from state's judge_queue as jobs finish, until the queue is
    empty and nothing is left in flight.

    Concurrency (rather than one job at a time) matters because OpenAI's
    batch queue processes jobs opportunistically against spare capacity --
    submitting job N+1 only after job N fully finishes leaves that capacity
    idle whenever the backend could have run both at once. judge_batch_size
    and judge_max_concurrent_jobs in config.yaml are sized together so the
    pool's combined reserved tokens (prompt + max_output_tokens per request,
    summed across every in-flight job) stay under whichever provider's
    org-wide quota applies -- OpenAI enforces one (2,000,000 for this
    account) and rejects a job outright if exceeded; Gemini hasn't hit one
    yet but the same margin is kept regardless of provider.
    """
    # Migrate the legacy single-job format (from before concurrent pools
    # existed) into the pool, so a job already in flight under the old key
    # isn't lost or resubmitted.
    legacy = state.get_pending_judge_batch()
    if legacy:
        provider = legacy.get("provider", config["llm"]["judge"]["provider"])
        state.add_pending_judge_batch(legacy["batch_id"], legacy["chunks"], provider)
        state.clear_pending_judge_batch()

    # Total reserved-token budget across ALL concurrently in-flight jobs, not
    # just a job count -- matters right after a config change (e.g. shrinking
    # judge_batch_job_max_chunks to enable more concurrency): a queue built
    # under the OLD, larger chunk size can still have oversized items in it,
    # and a legacy-migrated job may also be larger than the current setting.
    # Sizing the budget off the current (possibly just-lowered) max-chunks
    # value keeps total exposure the same regardless of how it's split.
    max_chunks_per_job = config["pipeline"]["judge_batch_job_max_chunks"]
    max_total_chunks_in_flight = max_chunks_per_job * max_concurrent

    while True:
        pending = state.get_pending_judge_batches()
        chunks_in_flight = sum(len(p["chunks"]) for p in pending)

        while len(pending) < max_concurrent:
            sub_batch = state.pop_judge_queue_item()
            if sub_batch is None:
                break
            if pending and chunks_in_flight + len(sub_batch) > max_total_chunks_in_flight:
                # Adding this now risks exceeding the safe token budget while
                # other (possibly differently-sized) jobs are still in
                # flight. Defer it rather than risk an outright rejection --
                # it'll be picked up once something currently pending frees
                # up room.
                state.push_judge_queue_item_front(sub_batch)
                break
            try:
                provider = config["llm"]["judge"]["provider"]
                batch_id = judge.submit_judge_batch_job(sub_batch, config, prompts_dir)
            except Exception:
                logger.exception(
                    "failed to submit judge sub-batch (%d chunks); returning it to the queue for retry", len(sub_batch),
                )
                state.push_judge_queue_item_front(sub_batch)
                break
            state.add_pending_judge_batch(batch_id, sub_batch, provider)
            pending = state.get_pending_judge_batches()
            chunks_in_flight += len(sub_batch)

        if not pending:
            return

        any_finished = False
        for p in pending:
            job = judge.check_batch_status(p["batch_id"], p["provider"])
            if not judge.is_terminal(job, p["provider"]):
                continue
            any_finished = True
            if not judge.is_batch_successful(job, p["provider"]):
                logger.error("judge batch job %s did not succeed; affected words stay pending", p["batch_id"])
            results_by_custom_id = judge.fetch_judge_batch_results(job, p["chunks"], p["provider"])
            _apply_judge_results(p["chunks"], results_by_custom_id, state, paths, thresholds, min_passing, max_rounds)
            state.remove_pending_judge_batch(p["batch_id"])

        if not any_finished:
            time.sleep(poll_interval_seconds)


def _apply_generation_results(
    word_batches: list[list[dict]],
    results_by_chunk_id: dict[str, list[dict]],
    state: PipelineState,
    paths: dict,
    embedding_model: str,
    similarity_threshold: float,
) -> list[list[dict]]:
    """Runs local validation/dedup on one generation job's results. Returns
    the judge-ready chunks (empty chunks, i.e. words with nothing surviving,
    are simply omitted)."""
    judge_chunks: list[list[dict]] = []
    for i, batch in enumerate(word_batches):
        gen_results = results_by_chunk_id.get(str(i))
        if gen_results is None:
            logger.error("no generation result for chunk %d (%d word(s)); leaving pending for another round", i, len(batch))
            jsonl_storage.append_jsonl(paths["rejected"], [
                {"row_id": r["row_id"], "status": "FAILED", "stage": "generation", "reason": "no batch response for this chunk"}
                for r in batch
            ])
            continue
        judge_chunk = _process_generation_results(
            batch, gen_results, state, paths, embedding_model, similarity_threshold,
        )
        if judge_chunk:
            judge_chunks.append(judge_chunk)
        state.save()
    return judge_chunks


def _drain_generation_queue(
    config: dict,
    prompts_dir: str,
    state: PipelineState,
    paths: dict,
    embedding_model: str,
    similarity_threshold: float,
    max_concurrent: int = 1,
    poll_interval_seconds: int = 30,
) -> list[list[dict]]:
    """Generation-side counterpart to _drain_judge_queue: runs a pool of up to
    `max_concurrent` generation batch jobs at once, refilled from
    state's generation_queue as jobs finish, until the queue is empty and
    nothing is left in flight. Returns the combined judge-ready chunks from
    every job that finished. See _drain_judge_queue for the full rationale
    (concurrency against opportunistic batch-queue capacity; combined
    reserved-token budget kept under whichever provider's org-wide quota
    applies)."""
    legacy = state.get_pending_generation_batch()
    if legacy:
        provider = legacy.get("provider", config["llm"]["generator"]["provider"])
        state.add_pending_generation_batch(legacy["job_name"], legacy["chunks"], provider)
        state.clear_pending_generation_batch()

    max_chunks_per_job = config["pipeline"]["generation_batch_job_max_chunks"]
    max_total_chunks_in_flight = max_chunks_per_job * max_concurrent

    all_judge_chunks: list[list[dict]] = []
    while True:
        pending = state.get_pending_generation_batches()
        chunks_in_flight = sum(len(p["chunks"]) for p in pending)

        while len(pending) < max_concurrent:
            sub_batch = state.pop_generation_queue_item()
            if sub_batch is None:
                break
            if pending and chunks_in_flight + len(sub_batch) > max_total_chunks_in_flight:
                state.push_generation_queue_item_front(sub_batch)
                break
            try:
                provider = config["llm"]["generator"]["provider"]
                job_name = generator.submit_generation_batch_job(sub_batch, config, prompts_dir)
            except Exception:
                logger.exception(
                    "failed to submit generation sub-batch (%d chunks); returning it to the queue for retry", len(sub_batch),
                )
                state.push_generation_queue_item_front(sub_batch)
                break
            state.add_pending_generation_batch(job_name, sub_batch, provider)
            pending = state.get_pending_generation_batches()
            chunks_in_flight += len(sub_batch)

        if not pending:
            return all_judge_chunks

        any_finished = False
        for p in pending:
            job = generator.check_batch_status(p["job_name"], p["provider"])
            if not generator.is_terminal(job, p["provider"]):
                continue
            any_finished = True
            if not generator.is_batch_successful(job, p["provider"]):
                logger.error("generation batch job %s did not succeed; affected words stay pending", p["job_name"])
            results_by_chunk_id = generator.fetch_generation_batch_results(job, p["chunks"], p["provider"])
            judge_chunks = _apply_generation_results(
                p["chunks"], results_by_chunk_id, state, paths, embedding_model, similarity_threshold,
            )
            all_judge_chunks.extend(judge_chunks)
            state.remove_pending_generation_batch(p["job_name"])

        if not any_finished:
            time.sleep(poll_interval_seconds)


def export_final(all_records: list[dict], state: PipelineState, paths: dict) -> None:
    final_records = []
    for r in all_records:
        entry = state.get_word(r["row_id"])
        if entry:
            final_records.extend(entry["accepted"])

    os.makedirs(paths["data_final"], exist_ok=True)
    jsonl_storage.write_jsonl(os.path.join(paths["data_final"], "final_sentences.jsonl"), final_records)
    excel_storage.write_final_excel(os.path.join(paths["data_final"], "final_sentences.xlsx"), final_records)
    logger.info("exported %d accepted sentences to %s", len(final_records), paths["data_final"])


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _complete_pending_generation_raw_log(state: PipelineState, paths: dict, config: dict) -> None:
    """If a generation batch job is still tracked as pending (legacy single-job
    form, or the new pool form), poll it (read-only -- no new request
    submitted, no new spend) and flush its raw output to disk without running
    validation/dedup yet. Used by replay_from_raw() so an already-submitted,
    already-paid-for job's results get captured without triggering any *new*
    generation API usage."""
    jobs = []
    legacy = state.get_pending_generation_batch()
    if legacy:
        provider = legacy.get("provider", config["llm"]["generator"]["provider"])
        jobs.append((legacy["job_name"], legacy["chunks"], provider))
    for p in state.get_pending_generation_batches():
        jobs.append((p["job_name"], p["chunks"], p["provider"]))

    for job_name, word_batches, provider in jobs:
        logger.info("polling existing generation batch job %s to complete the raw log (read-only, no new requests)...", job_name)
        final_job = generator.poll_generation_batch_job(job_name, provider, poll_interval_seconds=30)
        if not generator.is_batch_successful(final_job, provider):
            logger.error("generation batch job %s did not succeed; some words may be missing this round's data", job_name)
        results_by_chunk_id = generator.fetch_generation_batch_results(final_job, word_batches, provider)
        for i in range(len(word_batches)):
            gen_results = results_by_chunk_id.get(str(i))
            if gen_results is not None:
                jsonl_storage.append_jsonl(paths["raw"], gen_results)

    state.clear_pending_generation_batch()
    for job_name, _, _ in jobs:
        state.remove_pending_generation_batch(job_name)
    state.clear_pending_generation_batch()


def replay_from_raw(config_path: str = "config/config.yaml") -> None:
    """Reprocesses whatever generator output already exists in data/raw's raw
    log -- merging every word's candidate sentences across ALL prior
    generation attempts into one combined pool -- and judges them, WITHOUT
    calling the generator/Gemini for anything new. Words still short of their
    passing quota after this are left "pending" for a future run() call once
    you're ready to spend more Gemini credits on fresh generation.
    """
    load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
    config = load_config(config_path)
    setup_logging(config)

    paths = {
        "raw": os.path.join(config["paths"]["data_raw"], "generated.jsonl"),
        "processed": os.path.join(config["paths"]["data_processed"], "judged.jsonl"),
        "rejected": os.path.join(config["paths"]["data_rejected"], "rejected.jsonl"),
        "data_final": config["paths"]["data_final"],
        "logs_dir": config["paths"]["logs_dir"],
    }
    prompts_dir = config["paths"]["prompts_dir"]
    embedding_model = config["llm"]["embedding"]["model"]
    similarity_threshold = config["pipeline"]["near_duplicate_similarity_threshold"]
    min_passing = config["pipeline"]["min_passing_sentences_per_word"]
    max_rounds = config["pipeline"]["max_generation_rounds"]
    batch_size = config["pipeline"]["batch_size"]
    judge_batch_job_max_chunks = config["pipeline"]["judge_batch_job_max_chunks"]
    judge_max_concurrent_jobs = config["pipeline"].get("judge_max_concurrent_jobs", 1)
    thresholds = config["judge_thresholds"]

    all_records = excel_storage.read_words(config["paths"]["input_file"])
    state = PipelineState(config["paths"]["state_file"])
    for r in all_records:
        state.ensure_word(r)
    state.save()

    _complete_pending_generation_raw_log(state, paths, config)

    # Reconnect/drain any judge work left over from a prior interrupted run.
    words_to_finalize: dict[int, dict] = {}
    for sub_batch in state.get_judge_queue():
        for chunk in sub_batch:
            for item in chunk:
                words_to_finalize.setdefault(item["row_id"], item)
    pending_judge = state.get_pending_judge_batch()
    if pending_judge:
        for chunk in pending_judge["chunks"]:
            for item in chunk:
                words_to_finalize.setdefault(item["row_id"], item)
    _drain_judge_queue(config, prompts_dir, state, paths, thresholds, min_passing, max_rounds, judge_max_concurrent_jobs)
    for row_id in words_to_finalize:
        state.finalize_status(row_id, min_passing, max_rounds)
    if words_to_finalize:
        state.save()

    # Merge every word's candidate sentences across all prior raw generation
    # attempts (e.g. an interrupted round 1 + round 2) into one combined pool.
    merged_by_row_id: dict[int, dict] = {}
    for rec in jsonl_storage.iter_jsonl(paths["raw"]):
        if not isinstance(rec, dict) or "sentences" not in rec:
            continue
        row_id = rec["row_id"]
        entry = merged_by_row_id.setdefault(row_id, {
            "row_id": row_id,
            "domain": rec.get("domain"),
            "category": rec.get("category"),
            "word": rec.get("word"),
            "sentences": [],
        })
        entry["sentences"].extend(rec.get("sentences", []))
    logger.info("merged raw generation output for %d word(s) across all prior attempts", len(merged_by_row_id))

    pending_words = state.words_needing_round(all_records, min_passing)
    pending_row_ids = {r["row_id"] for r in pending_words}
    merged_records = [rec for row_id, rec in merged_by_row_id.items() if row_id in pending_row_ids]
    logger.info("%d of those still need (more) sentences; reprocessing them now", len(merged_records))

    if merged_records:
        judge_chunks: list[list[dict]] = []
        for batch in batching.make_batches(merged_records, batch_size):
            chunk = _process_generation_results(batch, batch, state, paths, embedding_model, similarity_threshold)
            if chunk:
                judge_chunks.append(chunk)
            state.save()

        if judge_chunks:
            sub_batches = list(batching.make_batches(judge_chunks, judge_batch_job_max_chunks))
            logger.info("replay: %d judge chunk(s) split into %d sequential sub-job(s)", len(judge_chunks), len(sub_batches))
            state.set_judge_queue(sub_batches)
            _drain_judge_queue(config, prompts_dir, state, paths, thresholds, min_passing, max_rounds, judge_max_concurrent_jobs)

        for r in pending_words:
            state.finalize_status(r["row_id"], min_passing, max_rounds)
        state.save()
    else:
        logger.info("nothing to replay -- all words already satisfied")

    export_final(all_records, state, paths)
    still_short = state.words_needing_round(all_records, min_passing)
    logger.info(
        "replay complete: %s -- %d word(s) still need fresh Gemini generation in a future run() call",
        state.summary(), len(still_short),
    )


def run(config_path: str = "config/config.yaml") -> None:
    # Explicit path rather than load_dotenv()'s stack-frame auto-discovery,
    # which is fragile (breaks under exec-from-stdin, differs by caller).
    load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
    config = load_config(config_path)
    setup_logging(config)

    paths = {
        "raw": os.path.join(config["paths"]["data_raw"], "generated.jsonl"),
        "processed": os.path.join(config["paths"]["data_processed"], "judged.jsonl"),
        "rejected": os.path.join(config["paths"]["data_rejected"], "rejected.jsonl"),
        "data_final": config["paths"]["data_final"],
        "logs_dir": config["paths"]["logs_dir"],
    }
    prompts_dir = config["paths"]["prompts_dir"]
    embedding_model = config["llm"]["embedding"]["model"]
    similarity_threshold = config["pipeline"]["near_duplicate_similarity_threshold"]
    min_passing = config["pipeline"]["min_passing_sentences_per_word"]
    max_rounds = config["pipeline"]["max_generation_rounds"]
    batch_size = config["pipeline"]["batch_size"]
    # Default is intentionally large (effectively "one job") for backward
    # compatibility with Gemini-only configs that predate this cap -- Gemini
    # has handled a full 620-chunk round in one job with no issue.
    generation_batch_job_max_chunks = config["pipeline"].get("generation_batch_job_max_chunks", 1000)
    generation_max_concurrent_jobs = config["pipeline"].get("generation_max_concurrent_jobs", 1)
    judge_batch_job_max_chunks = config["pipeline"]["judge_batch_job_max_chunks"]
    judge_max_concurrent_jobs = config["pipeline"].get("judge_max_concurrent_jobs", 1)
    thresholds = config["judge_thresholds"]

    logger.info("loading input words from %s", config["paths"]["input_file"])
    all_records = excel_storage.read_words(config["paths"]["input_file"])
    logger.info("loaded %d words", len(all_records))

    state = PipelineState(config["paths"]["state_file"])
    for r in all_records:
        state.ensure_word(r)
    state.save()

    # Reconnect to any batch job left in-flight by a previous, interrupted run
    # instead of silently losing/resubmitting (and double-paying for) it --
    # generation queue/pool first (draining it produces judge chunks), then
    # judge queue/pool. Words touched here need finalize_status too (the
    # round loop below only finalizes words it processed itself), or they'd
    # stay "pending" forever even once they have enough accepted sentences.
    words_to_finalize: dict[int, dict] = {}

    for sub_batch in state.get_generation_queue():
        for chunk in sub_batch:
            for item in chunk:
                words_to_finalize.setdefault(item["row_id"], item)
    for p in state.get_pending_generation_batches():
        for chunk in p["chunks"]:
            for item in chunk:
                words_to_finalize.setdefault(item["row_id"], item)
    legacy_gen = state.get_pending_generation_batch()
    if legacy_gen:
        for batch in legacy_gen["chunks"]:
            for r in batch:
                words_to_finalize.setdefault(r["row_id"], r)

    leftover_judge_chunks = _drain_generation_queue(
        config, prompts_dir, state, paths, embedding_model, similarity_threshold, generation_max_concurrent_jobs,
    )
    if leftover_judge_chunks:
        sub_batches = list(batching.make_batches(leftover_judge_chunks, judge_batch_job_max_chunks))
        state.set_judge_queue(state.get_judge_queue() + sub_batches)

    for sub_batch in state.get_judge_queue():
        for chunk in sub_batch:
            for item in chunk:
                words_to_finalize.setdefault(item["row_id"], item)
    for p in state.get_pending_judge_batches():
        for chunk in p["chunks"]:
            for item in chunk:
                words_to_finalize.setdefault(item["row_id"], item)
    legacy_judge = state.get_pending_judge_batch()
    if legacy_judge:
        for chunk in legacy_judge["chunks"]:
            for item in chunk:
                words_to_finalize.setdefault(item["row_id"], item)

    _drain_judge_queue(config, prompts_dir, state, paths, thresholds, min_passing, max_rounds, judge_max_concurrent_jobs)

    for row_id in words_to_finalize:
        state.finalize_status(row_id, min_passing, max_rounds)
    if words_to_finalize:
        state.save()

    for round_num in range(1, max_rounds + 1):
        pending = state.words_needing_round(all_records, min_passing)
        if not pending:
            logger.info("all words satisfied after round %d", round_num - 1)
            break

        logger.info("round %d/%d: %d word(s) need (more) sentences", round_num, max_rounds, len(pending))
        for r in pending:
            state.record_round_attempt(r["row_id"])
        word_batches = list(batching.make_batches(pending, batch_size))

        gen_sub_batches = list(batching.make_batches(word_batches, generation_batch_job_max_chunks))
        logger.info(
            "round %d: %d word-batch(es) split into %d generation sub-job(s) of <=%d chunks each",
            round_num, len(word_batches), len(gen_sub_batches), generation_batch_job_max_chunks,
        )
        state.set_generation_queue(gen_sub_batches)
        judge_chunks = _drain_generation_queue(
            config, prompts_dir, state, paths, embedding_model, similarity_threshold, generation_max_concurrent_jobs,
        )

        if not judge_chunks:
            logger.info("round %d: nothing survived to judge", round_num)
        else:
            sub_batches = list(batching.make_batches(judge_chunks, judge_batch_job_max_chunks))
            logger.info(
                "round %d: %d judge chunk(s) split into %d sub-job(s) of <=%d chunks each",
                round_num, len(judge_chunks), len(sub_batches), judge_batch_job_max_chunks,
            )
            state.set_judge_queue(sub_batches)
            _drain_judge_queue(config, prompts_dir, state, paths, thresholds, min_passing, max_rounds, judge_max_concurrent_jobs)

        for r in pending:
            state.finalize_status(r["row_id"], min_passing, max_rounds)
        state.save()

    export_final(all_records, state, paths)
    logger.info("pipeline complete: %s", state.summary())


def _process_word_batch_synchronously(
    batch: list[dict],
    config: dict,
    prompts_dir: str,
    state: PipelineState,
    paths: dict,
    embedding_model: str,
    similarity_threshold: float,
    thresholds: dict,
    min_passing: int,
    max_rounds: int,
    state_lock: threading.Lock,
) -> None:
    """Runs one word-batch fully through generate -> validate/dedup -> judge
    -> apply, entirely via synchronous (non-batch-API) calls. Used when a
    provider's async Batch API is unavailable -- no per-provider token-cap
    concerns apply since there's no batch job to size.

    Safe to call from multiple threads concurrently: all reads/writes of
    `state` are serialized via `state_lock`, while the slow network calls
    (generate_batch/judge_batch) run without holding it, so threads overlap
    during the actual waiting."""
    row_ids = [r["row_id"] for r in batch]
    try:
        gen_results = generator.generate_batch(batch, config, prompts_dir)
    except Exception:
        logger.exception("synchronous generation failed for row_ids=%s; leaving pending for another round", row_ids)
        return

    with state_lock:
        judge_chunk = _process_generation_results(
            batch, gen_results, state, paths, embedding_model, similarity_threshold,
        )

    if not judge_chunk:
        logger.info("nothing survived to judge for row_ids=%s", row_ids)
        return

    try:
        judge_results = judge.judge_batch(judge_chunk, config, prompts_dir)
    except Exception:
        logger.exception("synchronous judging failed for row_ids=%s; leaving pending for another round", row_ids)
        return

    with state_lock:
        _apply_judge_results(
            [judge_chunk], {"chunk-0": judge_results}, state, paths, thresholds, min_passing, max_rounds,
        )


def run_synchronous(config_path: str = "config/config.yaml", max_workers: int | None = None) -> None:
    """Fallback execution mode for when a provider's async Batch API is
    unavailable (e.g. the OpenAI Batch API file-access outage encountered on
    2026-08-31, confirmed account-wide and unrelated to this codebase):
    processes every pending word-batch via plain synchronous calls
    (generator.generate_batch / judge.judge_batch), a handful at a time via a
    thread pool since these are independent, I/O-bound HTTP calls. No batch
    jobs, no queues, no token-cap concerns -- just N workers making regular
    API calls. Costs full price (no ~50% batch discount) in exchange for
    working regardless of Batch API availability.
    """
    load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
    config = load_config(config_path)
    setup_logging(config)

    paths = {
        "raw": os.path.join(config["paths"]["data_raw"], "generated.jsonl"),
        "processed": os.path.join(config["paths"]["data_processed"], "judged.jsonl"),
        "rejected": os.path.join(config["paths"]["data_rejected"], "rejected.jsonl"),
        "data_final": config["paths"]["data_final"],
        "logs_dir": config["paths"]["logs_dir"],
    }
    prompts_dir = config["paths"]["prompts_dir"]
    embedding_model = config["llm"]["embedding"]["model"]
    similarity_threshold = config["pipeline"]["near_duplicate_similarity_threshold"]
    min_passing = config["pipeline"]["min_passing_sentences_per_word"]
    max_rounds = config["pipeline"]["max_generation_rounds"]
    batch_size = config["pipeline"]["batch_size"]
    thresholds = config["judge_thresholds"]
    if max_workers is None:
        max_workers = config["pipeline"].get("sync_max_workers", 5)

    logger.info("loading input words from %s", config["paths"]["input_file"])
    all_records = excel_storage.read_words(config["paths"]["input_file"])
    logger.info("loaded %d words", len(all_records))

    state = PipelineState(config["paths"]["state_file"])
    for r in all_records:
        state.ensure_word(r)
    state.save()

    state_lock = threading.Lock()

    for round_num in range(1, max_rounds + 1):
        with state_lock:
            pending = state.words_needing_round(all_records, min_passing)
        if not pending:
            logger.info("all words satisfied after round %d", round_num - 1)
            break

        logger.info("round %d/%d (synchronous): %d word(s) need (more) sentences", round_num, max_rounds, len(pending))
        with state_lock:
            for r in pending:
                state.record_round_attempt(r["row_id"])
            state.save()
        word_batches = list(batching.make_batches(pending, batch_size))
        logger.info("round %d: %d word-batch(es), %d worker(s)", round_num, len(word_batches), max_workers)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _process_word_batch_synchronously,
                    batch, config, prompts_dir, state, paths, embedding_model, similarity_threshold,
                    thresholds, min_passing, max_rounds, state_lock,
                ): batch
                for batch in word_batches
            }
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                batch = futures[future]
                try:
                    future.result()
                except Exception:
                    logger.exception("worker crashed for row_ids=%s", [r["row_id"] for r in batch])
                if done_count % 10 == 0 or done_count == len(word_batches):
                    logger.info("round %d: %d/%d word-batch(es) done", round_num, done_count, len(word_batches))

        with state_lock:
            for r in pending:
                state.finalize_status(r["row_id"], min_passing, max_rounds)
            state.save()

    export_final(all_records, state, paths)
    logger.info("pipeline complete: %s", state.summary())
