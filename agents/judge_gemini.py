"""Gemini implementation of the judge agent: scores candidate sentences for
quality, driven by prompts/judge.txt. Selected via agents/judge.py's
dispatcher when config.yaml's llm.judge.provider is "gemini".

Note: the judge's own `overall_score`/`decision` fields are returned for
audit/logging but are NOT the source of truth for pass/fail -- see
pipeline/validation.py, which recomputes the decision deterministically from
the per-dimension scores against config.yaml's judge_thresholds.
"""

from __future__ import annotations

import json
import logging
import os
import time

from google import genai
from google.genai import types

from pipeline.retry import call_with_backoff

logger = logging.getLogger("agents.judge")

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
        # Separate key allowed (e.g. a fresh project with its own credits),
        # falling back to the generator's key if only one is configured.
        api_key = os.environ.get("GEMINI_JUDGE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_JUDGE_API_KEY or GEMINI_API_KEY must be set in the environment")
        _client = genai.Client(api_key=api_key)
    return _client


def load_system_prompt(prompts_dir: str) -> str:
    path = os.path.join(prompts_dir, "judge.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _build_user_message(batches: list[dict]) -> str:
    return (
        "Evaluate every record in `records` below according to your instructions. "
        "Return a JSON array with exactly one evaluation object per input record, "
        "in the same order as the input records, each shaped as described in your "
        "instructions (`{row_id, results: [...]}`). "
        "Do not wrap the array in any other object, and return nothing but the JSON array.\n\n"
        f"records = {json.dumps(batches, ensure_ascii=False)}"
    )


def _response_instruction() -> str:
    return (
        "\n\nRespond with a single JSON object of shape "
        '{"evaluations": [<one evaluation object per record>]}.'
    )


def _generation_config(config: dict, system_prompt: str) -> types.GenerateContentConfig:
    judge_cfg = config["llm"]["judge"]
    return types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=judge_cfg.get("temperature", 0.0),
        max_output_tokens=judge_cfg.get("max_output_tokens", 16384),
        response_mime_type="application/json",
    )


def _parse_evaluations(raw_text: str, generated_records: list[dict]) -> list[dict]:
    try:
        parsed = json.loads(raw_text)
        evaluations = parsed["evaluations"] if isinstance(parsed, dict) else parsed
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.error("judge returned invalid/unexpected JSON: %s", raw_text[:500])
        return [
            {"row_id": r["row_id"], "results": [], "_judge_error": "invalid JSON from judge"}
            for r in generated_records
        ]

    by_row_id = {item.get("row_id"): item for item in evaluations if isinstance(item, dict)}
    results = []
    for record in generated_records:
        item = by_row_id.get(record["row_id"])
        if item is None:
            results.append(
                {"row_id": record["row_id"], "results": [], "_judge_error": "record omitted by judge"}
            )
        else:
            results.append(item)
    return results


def judge_batch(
    generated_records: list[dict],
    config: dict,
    prompts_dir: str,
) -> list[dict]:
    """Judge a batch of generator output records via a single synchronous call.

    `generated_records` is a list of {row_id, domain, category, word, sentences}
    (the generator's success-shape output). Returns a list with one entry per
    input record: {row_id, results: [{sentence_id, ...scores..., decision, reason}]}.
    """
    if not generated_records:
        return []

    judge_cfg = config["llm"]["judge"]
    retry_cfg = config.get("retry", {})
    system_prompt = load_system_prompt(prompts_dir)
    user_message = _build_user_message(generated_records) + _response_instruction()

    def _call() -> str:
        client = _get_client()
        response = client.models.generate_content(
            model=judge_cfg["model"],
            contents=user_message,
            config=_generation_config(config, system_prompt),
        )
        if not response.text:
            raise RuntimeError("empty response from judge model")
        return response.text

    raw_text = call_with_backoff(
        _call,
        max_attempts=retry_cfg.get("max_api_attempts", 4),
        base_backoff_seconds=retry_cfg.get("base_backoff_seconds", 2.0),
        max_backoff_seconds=retry_cfg.get("max_backoff_seconds", 30.0),
    )
    return _parse_evaluations(raw_text, generated_records)


# --- Gemini Batch Mode (async, ~50% cheaper than the synchronous calls above) ---
# One batch JOB covers every chunk for a whole round, rather than one job per
# chunk -- submitting per-chunk would pay the job-startup overhead hundreds of
# times over.

def submit_judge_batch_job(chunks: list[list[dict]], config: dict, prompts_dir: str) -> str:
    """chunks: a list of "generated_records" lists, each shaped like a single
    judge_batch() call's input. Submits them all as one Batch Mode job and
    returns the job name immediately (does not wait for completion)."""
    client = _get_client()
    judge_cfg = config["llm"]["judge"]
    system_prompt = load_system_prompt(prompts_dir)

    requests = []
    for i, records in enumerate(chunks):
        user_message = _build_user_message(records) + _response_instruction()
        requests.append(types.InlinedRequest(
            model=judge_cfg["model"],
            contents=user_message,
            metadata={"chunk_id": f"chunk-{i}"},
            config=_generation_config(config, system_prompt),
        ))

    job = client.batches.create(
        model=judge_cfg["model"],
        src=requests,
        config=types.CreateBatchJobConfig(display_name="sentence-pipeline-judging"),
    )
    logger.info("submitted judge batch job %s with %d chunk(s)", job.name, len(requests))
    return job.name


def poll_judge_batch_job(job_name: str, poll_interval_seconds: int = 30, timeout_seconds: int = 24 * 3600):
    """Blocks until the batch job reaches a terminal state, logging status on
    every poll. Safe to call again after a process restart with the same
    job_name -- it just reconnects to the existing job."""
    client = _get_client()
    t0 = time.time()
    while True:
        job = client.batches.get(name=job_name)
        logger.info("judge batch %s state=%s stats=%s", job_name, job.state, job.completion_stats)
        if job.state.name in TERMINAL_JOB_STATES:
            return job
        if time.time() - t0 > timeout_seconds:
            raise TimeoutError(f"judge batch job {job_name} did not reach a terminal state within {timeout_seconds}s")
        time.sleep(poll_interval_seconds)


def check_batch_status(job_name: str):
    """Single, non-blocking status check (one API call), for callers managing
    several concurrent jobs themselves rather than blocking on just one."""
    client = _get_client()
    job = client.batches.get(name=job_name)
    logger.info("judge batch %s state=%s stats=%s", job_name, job.state, job.completion_stats)
    return job


def is_terminal(job) -> bool:
    return job.state.name in TERMINAL_JOB_STATES


def is_batch_successful(job) -> bool:
    return job.state.name == "JOB_STATE_SUCCEEDED"


def fetch_judge_batch_results(job, chunks: list[list[dict]]) -> dict[str, list[dict]]:
    """Returns {custom_id: parsed-evaluations-in-judge_batch()-shape}. Missing/
    failed chunks are simply absent from the returned dict; the caller decides
    how to treat them (e.g. leave those words pending for another round)."""
    results: dict[str, list[dict]] = {}
    if not (job.dest and job.dest.inlined_responses):
        logger.error("judge batch job %s produced no inlined_responses (dest=%s)", job.name, job.dest)
        return results

    for r in job.dest.inlined_responses:
        chunk_id = (r.metadata or {}).get("chunk_id")
        if chunk_id is None:
            continue
        idx = int(chunk_id.split("-", 1)[1])
        records = chunks[idx]
        if r.error:
            logger.error("judge batch chunk %s errored: %s", chunk_id, r.error)
            continue
        text = r.response.text if r.response else None
        if not text:
            logger.error("judge batch chunk %s had empty response", chunk_id)
            continue
        results[chunk_id] = _parse_evaluations(text, records)

    return results
