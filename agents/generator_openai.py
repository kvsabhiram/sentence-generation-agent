"""OpenAI implementation of the generator agent: turns (row_id, domain,
category, word) records into candidate sentences, driven by
prompts/generator.txt. Selected via agents/generator.py's dispatcher when
config.yaml's llm.generator.provider is "openai".

OpenAI's JSON mode (response_format={"type": "json_object"}) requires the
top-level response to be a JSON *object*, not a bare array -- unlike Gemini,
which accepts either. Requests therefore ask for {"records": [...]} and this
module unwraps that key before handing records back in the same shape
agents/generator_gemini.py returns them in.

OpenAI's batch API enforces an org-wide *enqueued* token cap per model
(2,000,000 on this account) that rejects a job outright if exceeded -- see
pipeline/orchestrator.py's generation_batch_job_max_chunks / concurrent-pool
draining, which exists specifically to stay under this cap (the same
mechanism already used for judging).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time

from openai import OpenAI

from pipeline.retry import call_with_backoff

logger = logging.getLogger("agents.generator_openai")

_client: OpenAI | None = None

TERMINAL_BATCH_STATUSES = ("completed", "failed", "expired", "cancelled")


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPEN_AI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPEN_AI_API_KEY is not set in the environment")
        _client = OpenAI(api_key=api_key)
    return _client


def load_system_prompt(prompts_dir: str) -> str:
    path = os.path.join(prompts_dir, "generator.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _build_user_message(records: list[dict], sentences_per_word: int) -> str:
    return (
        "Process every record in `records` below according to your instructions. "
        f"Generate {sentences_per_word} sentences per record where possible (4-5 is "
        "acceptable per your instructions if fewer/more are not natural). "
        'Return a single JSON object of shape {"records": [<one output object per '
        "input record, in the same order as the input records>]}, using the object "
        "shapes described in your instructions (either the success shape with "
        '`sentences`, or the `{"row_id": ..., "status": "FAILED", "reason": ...}` shape). '
        "Do not include anything outside that one JSON object.\n\n"
        f"records = {json.dumps(records, ensure_ascii=False)}"
    )


def _build_request_body(records: list[dict], config: dict, prompts_dir: str) -> dict:
    gen_cfg = config["llm"]["generator"]
    sentences_per_word = config["pipeline"]["sentences_requested_per_word"]
    system_prompt = load_system_prompt(prompts_dir)
    user_message = _build_user_message(records, sentences_per_word)
    body = dict(
        model=gen_cfg["model"],
        max_completion_tokens=gen_cfg.get("max_output_tokens", 4096),
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    # Some reasoning-tier models reject any explicit temperature override and
    # only support their default (1) -- so only send it when the config asks
    # for something other than the default.
    temperature = gen_cfg.get("temperature")
    if temperature is not None and temperature != 1:
        body["temperature"] = temperature
    return body


def _parse_generation_response(raw_text: str, records: list[dict]) -> list[dict]:
    try:
        parsed = json.loads(raw_text)
        items = parsed["records"] if isinstance(parsed, dict) and "records" in parsed else parsed
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.error("generator returned invalid/unexpected JSON: %s", raw_text[:500])
        return [
            {"row_id": r["row_id"], "status": "FAILED", "reason": "invalid JSON from generator"}
            for r in records
        ]

    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        logger.error("generator returned unexpected JSON shape: %s", type(items))
        return [
            {"row_id": r["row_id"], "status": "FAILED", "reason": "unexpected JSON shape from generator"}
            for r in records
        ]

    by_row_id = {item.get("row_id"): item for item in items if isinstance(item, dict)}
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
    """Generate sentences for a batch of word records via a single synchronous call."""
    if not records:
        return []

    retry_cfg = config.get("retry", {})
    body = _build_request_body(records, config, prompts_dir)

    def _call() -> str:
        client = _get_client()
        response = client.chat.completions.create(**body)
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("empty response from generator model")
        return content

    raw_text = call_with_backoff(
        _call,
        max_attempts=retry_cfg.get("max_api_attempts", 4),
        base_backoff_seconds=retry_cfg.get("base_backoff_seconds", 2.0),
        max_backoff_seconds=retry_cfg.get("max_backoff_seconds", 30.0),
    )
    return _parse_generation_response(raw_text, records)


# --- OpenAI Batch API (async, ~50% cheaper than the synchronous calls above) ---
# One batch JOB covers every chunk for a whole round, rather than one job per
# chunk. The caller (pipeline/orchestrator.py) additionally splits work into
# sequential/concurrent sub-jobs to stay under OpenAI's org-wide
# enqueued-token cap, since one oversized job gets rejected outright.

def submit_generation_batch_job(chunks: list[list[dict]], config: dict, prompts_dir: str) -> str:
    """chunks: a list of word-record lists, each shaped like a single
    generate_batch() call's input. Submits them all as one Batch API job and
    returns the job id immediately (does not wait for completion)."""
    client = _get_client()
    lines = []
    for i, records in enumerate(chunks):
        body = _build_request_body(records, config, prompts_dir)
        lines.append({
            "custom_id": str(i),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        })

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        tmp_path = f.name

    try:
        with open(tmp_path, "rb") as fh:
            uploaded = client.files.create(file=fh, purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
    finally:
        os.remove(tmp_path)

    logger.info("submitted generation batch job %s with %d chunk(s)", batch.id, len(lines))
    return batch.id


def poll_generation_batch_job(batch_id: str, poll_interval_seconds: int = 30, timeout_seconds: int = 24 * 3600):
    """Blocks until the batch job reaches a terminal state, logging status on
    every poll. Safe to call again after a process restart with the same
    batch_id -- it just reconnects to the existing job."""
    client = _get_client()
    t0 = time.time()
    while True:
        batch = client.batches.retrieve(batch_id)
        logger.info("generation batch %s status=%s counts=%s", batch_id, batch.status, batch.request_counts)
        if batch.status in TERMINAL_BATCH_STATUSES:
            return batch
        if time.time() - t0 > timeout_seconds:
            raise TimeoutError(f"generation batch job {batch_id} did not reach a terminal state within {timeout_seconds}s")
        time.sleep(poll_interval_seconds)


def check_batch_status(batch_id: str):
    """Single, non-blocking status check (one API call), for callers managing
    several concurrent jobs themselves rather than blocking on just one."""
    client = _get_client()
    batch = client.batches.retrieve(batch_id)
    logger.info("generation batch %s status=%s counts=%s", batch_id, batch.status, batch.request_counts)
    return batch


def is_terminal(batch) -> bool:
    return batch.status in TERMINAL_BATCH_STATUSES


def is_batch_successful(batch) -> bool:
    return batch.status == "completed"


def fetch_generation_batch_results(batch, chunks: list[list[dict]]) -> dict[str, list[dict]]:
    """Returns {chunk_id: parsed generate_batch()-shaped records} for whichever
    chunks got a response. Missing/failed chunks are simply absent; the caller
    decides how to treat them (e.g. leave those words pending for another round)."""
    client = _get_client()
    results: dict[str, list[dict]] = {}

    if batch.output_file_id:
        content = client.files.content(batch.output_file_id).text
        for line in content.strip().split("\n"):
            if not line:
                continue
            obj = json.loads(line)
            custom_id = obj["custom_id"]
            idx = int(custom_id)
            body = (obj.get("response") or {}).get("body")
            if body is None:
                logger.error("generation batch chunk %s had no response body: %s", custom_id, obj.get("error"))
                continue
            content_str = body.get("choices", [{}])[0].get("message", {}).get("content")
            if not content_str:
                logger.error("generation batch chunk %s had empty content", custom_id)
                continue
            results[custom_id] = _parse_generation_response(content_str, chunks[idx])

    if batch.error_file_id:
        err_text = client.files.content(batch.error_file_id).text
        logger.error("generation batch job had per-request errors: %s", err_text[:2000])

    return results
