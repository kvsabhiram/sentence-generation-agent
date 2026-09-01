"""Generator agent dispatcher: routes to agents/generator_gemini.py or
agents/generator_openai.py based on config.yaml's llm.generator.provider.

Mirrors agents/judge.py's dispatcher for the same reason: both providers'
credit balances have run dry at different points during this project, so
either agent needs to be switchable without code changes. See
pipeline/orchestrator.py, which persists the provider used at submission
time alongside the batch id/job name in state.json, so a resumed run
reconnects with the SAME provider even if config.yaml's setting has since
changed.
"""

from __future__ import annotations

import agents.generator_gemini as generator_gemini
import agents.generator_openai as generator_openai

_PROVIDERS = {
    "gemini": generator_gemini,
    "openai": generator_openai,
}


def _impl(provider: str):
    try:
        return _PROVIDERS[provider]
    except KeyError:
        raise ValueError(f"unknown generator provider {provider!r}; expected one of {list(_PROVIDERS)}") from None


def generate_batch(records: list[dict], config: dict, prompts_dir: str) -> list[dict]:
    provider = config["llm"]["generator"]["provider"]
    return _impl(provider).generate_batch(records, config, prompts_dir)


def submit_generation_batch_job(chunks: list[list[dict]], config: dict, prompts_dir: str) -> str:
    provider = config["llm"]["generator"]["provider"]
    return _impl(provider).submit_generation_batch_job(chunks, config, prompts_dir)


def poll_generation_batch_job(job_id: str, provider: str, poll_interval_seconds: int = 30, timeout_seconds: int = 24 * 3600):
    return _impl(provider).poll_generation_batch_job(job_id, poll_interval_seconds, timeout_seconds)


def check_batch_status(job_id: str, provider: str):
    return _impl(provider).check_batch_status(job_id)


def is_terminal(job, provider: str) -> bool:
    return _impl(provider).is_terminal(job)


def is_batch_successful(job, provider: str) -> bool:
    return _impl(provider).is_batch_successful(job)


def fetch_generation_batch_results(job, chunks: list[list[dict]], provider: str) -> dict[str, list[dict]]:
    return _impl(provider).fetch_generation_batch_results(job, chunks)
