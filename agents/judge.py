"""Judge agent dispatcher: routes to agents/judge_gemini.py or
agents/judge_openai.py based on config.yaml's llm.judge.provider.

Both providers' credit balances have run dry at different points during this
project, so the judge needs to be switchable without code changes. The
concrete implementations expose an identical function surface
(judge_batch, submit_judge_batch_job, poll_judge_batch_job,
fetch_judge_batch_results, is_batch_successful) so callers never need to know
which provider is active -- except pipeline/orchestrator.py, which persists
the provider used at submission time alongside the batch id/job name in
state.json, so a resumed run reconnects with the SAME provider even if
config.yaml's setting has since changed (it has, more than once).
"""

from __future__ import annotations

import agents.judge_gemini as judge_gemini
import agents.judge_openai as judge_openai

_PROVIDERS = {
    "gemini": judge_gemini,
    "openai": judge_openai,
}


def _impl(provider: str):
    try:
        return _PROVIDERS[provider]
    except KeyError:
        raise ValueError(f"unknown judge provider {provider!r}; expected one of {list(_PROVIDERS)}") from None


def judge_batch(generated_records: list[dict], config: dict, prompts_dir: str) -> list[dict]:
    provider = config["llm"]["judge"]["provider"]
    return _impl(provider).judge_batch(generated_records, config, prompts_dir)


def submit_judge_batch_job(chunks: list[list[dict]], config: dict, prompts_dir: str) -> str:
    provider = config["llm"]["judge"]["provider"]
    return _impl(provider).submit_judge_batch_job(chunks, config, prompts_dir)


def poll_judge_batch_job(job_id: str, provider: str, poll_interval_seconds: int = 30, timeout_seconds: int = 24 * 3600):
    return _impl(provider).poll_judge_batch_job(job_id, poll_interval_seconds, timeout_seconds)


def check_batch_status(job_id: str, provider: str):
    return _impl(provider).check_batch_status(job_id)


def is_terminal(job, provider: str) -> bool:
    return _impl(provider).is_terminal(job)


def is_batch_successful(job, provider: str) -> bool:
    return _impl(provider).is_batch_successful(job)


def fetch_judge_batch_results(job, chunks: list[list[dict]], provider: str) -> dict[str, list[dict]]:
    return _impl(provider).fetch_judge_batch_results(job, chunks)
