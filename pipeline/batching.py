"""Chunks word records into fixed-size batches for the generator/judge calls."""

from __future__ import annotations

from typing import Iterator


def make_batches(records: list[dict], batch_size: int) -> Iterator[list[dict]]:
    for i in range(0, len(records), batch_size):
        yield records[i : i + batch_size]
