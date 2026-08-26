"""Generator agent: turns (row_id, domain, category, word) records into candidate
sentences by calling the Gemini model, driven by prompts/generator.txt.
"""

from __future__ import annotations

import json
import logging
import os
import time

from google import genai
from google.genai import types

from pipeline.retry import call_with_backoff

logger = logging.getLogger("agents.generator")

_client: genai.Client | None = None

TERMINAL_JOB_STATES = (
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
)


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in the environment")
        _client = genai.Client(api_key=api_key)
    return _client


def load_system_prompt(prompts_dir: str) -> str:
    path = os.path.join(prompts_dir, "generator.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _build_user_message(records: list[dict], sentences_per_word: int) -> str:
    payload = {
        "sentences_per_word": sentences_per_word,
        "records": records,
    }
    return (
        "Process every record in `records` below according to your instructions. "
        f"Generate {sentences_per_word} sentences per record where possible (4-5 is "
        "acceptable per your instructions if fewer/more are not natural). "
        "Return a JSON array with exactly one output object per input record, in the "
        "same order as the input records, using the object shapes described in your "
        "instructions (either the success shape with `sentences`, or the "
        '`{"row_id": ..., "status": "FAILED", "reason": ...}` shape). '
        "Do not wrap the array in any other object, and return nothing but the JSON array.\n\n"
        f"records = {json.dumps(payload['records'], ensure_ascii=False)}"
    )


def _parse_generation_response(raw_text: str, records: list[dict]) -> list[dict]:
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.error("generator returned invalid JSON: %s", raw_text[:500])
        return [
            {"row_id": r["row_id"], "status": "FAILED", "reason": "invalid JSON from generator"}
            for r in records
        ]

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        logger.error("generator returned unexpected JSON shape: %s", type(parsed))
        return [
            {"row_id": r["row_id"], "status": "FAILED", "reason": "unexpected JSON shape from generator"}
            for r in records
        ]

    by_row_id = {item.get("row_id"): item for item in parsed if isinstance(item, dict)}
    results = []
    for record in records:
        item = by_row_id.get(record["row_id"])
        if item is None:
            results.append(
                {"row_id": record["row_id"], "status": "FAILED", "reason": "record omitted by generator"}
            )
        else:
            results.append(item)
    return results


def generate_batch(
    records: list[dict],
    config: dict,
    prompts_dir: str,
) -> list[dict]:
    """Generate sentences for a batch of word records via a single synchronous call.

    Returns a list with one entry per input record: either the success shape
    ({row_id, domain, category, word, sentences: [...]}) or a failure shape
    ({row_id, status: "FAILED", reason}).
    """
    if not records:
        return []

    gen_cfg = config["llm"]["generator"]
    sentences_per_word = config["pipeline"]["sentences_requested_per_word"]
    retry_cfg = config.get("retry", {})

    system_prompt = load_system_prompt(prompts_dir)
    user_message = _build_user_message(records, sentences_per_word)

    def _call() -> str:
        client = _get_client()
        response = client.models.generate_content(
            model=gen_cfg["model"],
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=gen_cfg.get("temperature", 0.9),
                max_output_tokens=gen_cfg.get("max_output_tokens", 4096),
                response_mime_type="application/json",
            ),
        )
        if not response.text:
            raise RuntimeError("empty response from generator model")
        return response.text

    raw_text = call_with_backoff(
        _call,
        max_attempts=retry_cfg.get("max_api_attempts", 4),
        base_backoff_seconds=retry_cfg.get("base_backoff_seconds", 2.0),
        max_backoff_seconds=retry_cfg.get("max_backoff_seconds", 30.0),
    )
    return _parse_generation_response(raw_text, records)


# --- Gemini Batch Mode (async, ~50% cheaper than the synchronous calls above) ---
# One batch JOB covers every chunk for a whole round, same reasoning as the
# judge's batch path: submitting per-chunk would pay the job-startup overhead
# hundreds of times over.

def submit_generation_batch_job(chunks: list[list[dict]], config: dict, prompts_dir: str) -> str:
    """chunks: a list of word-record lists, each shaped like a single
    generate_batch() call's input. Submits them all as one Batch Mode job and
    returns the job name immediately (does not wait for completion)."""
    client = _get_client()
    gen_cfg = config["llm"]["generator"]
    sentences_per_word = config["pipeline"]["sentences_requested_per_word"]
    system_prompt = load_system_prompt(prompts_dir)

    requests = []
    for i, records in enumerate(chunks):
        requests.append(types.InlinedRequest(
            model=gen_cfg["model"],
            contents=_build_user_message(records, sentences_per_word),
            metadata={"chunk_id": str(i)},
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=gen_cfg.get("temperature", 0.9),
                max_output_tokens=gen_cfg.get("max_output_tokens", 4096),
                response_mime_type="application/json",
            ),
        ))

    job = client.batches.create(
        model=gen_cfg["model"],
        src=requests,
        config=types.CreateBatchJobConfig(display_name="sentence-pipeline-generation"),
    )
    logger.info("submitted generation batch job %s with %d chunk(s)", job.name, len(requests))
    return job.name


def poll_generation_batch_job(job_name: str, poll_interval_seconds: int = 30, timeout_seconds: int = 24 * 3600):
    """Blocks until the batch job reaches a terminal state, logging status on
    every poll. Safe to call again after a process restart with the same
    job_name -- it just reconnects to the existing job."""
    client = _get_client()
    t0 = time.time()
    while True:
        job = client.batches.get(name=job_name)
        logger.info("generation batch %s state=%s stats=%s", job_name, job.state, job.completion_stats)
        if job.state.name in TERMINAL_JOB_STATES:
            return job
        if time.time() - t0 > timeout_seconds:
            raise TimeoutError(f"generation batch job {job_name} did not reach a terminal state within {timeout_seconds}s")
        time.sleep(poll_interval_seconds)


def fetch_generation_batch_results(job, chunks: list[list[dict]]) -> dict[str, list[dict]]:
    """Returns {chunk_id: parsed generate_batch()-shaped records} for whichever
    chunks got a response. Missing/failed chunks are simply absent; the caller
    decides how to treat them (e.g. leave those words pending for another round)."""
    results: dict[str, list[dict]] = {}
    if not (job.dest and job.dest.inlined_responses):
        logger.error("generation batch job %s produced no inlined_responses (dest=%s)", job.name, job.dest)
        return results

    for r in job.dest.inlined_responses:
        chunk_id = (r.metadata or {}).get("chunk_id")
        if chunk_id is None:
            continue
        idx = int(chunk_id)
        records = chunks[idx]
        if r.error:
            logger.error("generation batch chunk %s errored: %s", chunk_id, r.error)
            continue
        text = r.response.text if r.response else None
        if not text:
            logger.error("generation batch chunk %s had empty response", chunk_id)
            continue
        results[chunk_id] = _parse_generation_response(text, records)

    return results
