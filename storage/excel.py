"""Tabular I/O: reads the input word list (CSV or Excel) and writes the final
accepted dataset to Excel."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd

REQUIRED_INPUT_COLUMNS = ["domain", "category", "word"]


def read_words(path: str) -> list[dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"unsupported input file type {ext!r} for {path}")
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"input excel {path} is missing required column(s): {missing}. "
            f"found columns: {list(df.columns)}"
        )

    if "row_id" not in df.columns:
        df.insert(0, "row_id", range(1, len(df) + 1))

    records = []
    for row in df.to_dict(orient="records"):
        word = str(row["word"]).strip()
        domain = str(row["domain"]).strip()
        category = str(row["category"]).strip()
        if not word or word.lower() == "nan":
            continue
        records.append(
            {
                "row_id": int(row["row_id"]),
                "domain": domain,
                "category": category,
                "word": word,
            }
        )
    return records


def write_final_excel(path: str, sentence_records: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(
        sentence_records,
        columns=[
            "row_id",
            "sentence_id",
            "domain",
            "category",
            "word",
            "sentence",
            "context",
            "overall_score",
        ],
    )
    df.to_excel(path, index=False)
