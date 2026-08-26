"""Generic retry-with-backoff wrapper for transient API failures.

This handles *transport* failures (rate limits, timeouts, 5xx errors) for any
LLM/embedding call. It is unrelated to the pipeline-level "backfill" retries
in orchestrator.py, which regenerate sentences for a word that didn't reach
its passing-sentence quota -- that is a data-quality retry, not a transport one.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Callable, TypeVar

T = TypeVar("T")

logger = logging.getLogger("pipeline.retry")


def call_with_backoff(
    fn: Callable[[], T],
    max_attempts: int = 4,
    base_backoff_seconds: float = 2.0,
    max_backoff_seconds: float = 30.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    last_error: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except retry_on as exc:  # noqa: BLE001 - intentionally broad, caller narrows via retry_on
            last_error = exc
            if attempt == max_attempts:
                break
            sleep_for = min(base_backoff_seconds * (2 ** (attempt - 1)), max_backoff_seconds)
            sleep_for *= 1 + random.uniform(-0.2, 0.2)  # jitter
            logger.warning(
                "attempt %d/%d failed (%s); retrying in %.1fs",
                attempt,
                max_attempts,
                exc,
                sleep_for,
            )
            time.sleep(sleep_for)
    assert last_error is not None
    raise last_error
