"""Deterministic validation rules that sit between the LLM agents and the
pipeline state/storage layers.

Two things are intentionally NOT trusted from the LLMs, per the pipeline
design decisions:
  1. Word-usage correctness is re-checked here with a literal, word-boundary,
     case-insensitive match -- the generator's own claim of compliance isn't
     enough.
  2. The judge's self-reported `decision`/`overall_score` are logged for
     audit only; the actual PASS/FAIL gate is recomputed here from the
     judge's per-dimension scores against config.yaml's judge_thresholds.

Near-duplicate rejection is NOT done here -- that lives entirely in
embedding/similarity.py, which pipeline/orchestrator.py calls directly.
"""

from __future__ import annotations

import re

DIMENSION_KEYS = [
    "target_usage",
    "grammar",
    "naturalness",
    "domain_relevance",
    "category_relevance",
    "context_quality",
    "training_usefulness",
]


def _is_word_char(ch: str) -> bool:
    return bool(re.match(r"\w", ch, re.UNICODE))


def _word_pattern(word: str) -> re.Pattern:
    """\\b requires an actual word/non-word transition on BOTH sides, which
    silently fails to match when the target word itself starts/ends in
    punctuation next to other punctuation in the sentence (e.g. target word
    "Really?" inside `...said, "Really?"` -- "?" and the closing quote are
    both non-word chars, so no \\b exists between them even though the word
    is clearly present). Only assert a non-word-char boundary on a given
    side when that side of the target word is itself a word character --
    that's the only side where a false "substring of a longer word" match
    (e.g. "Store" inside "Stores") is even possible."""
    w = word.strip()
    escaped = re.escape(w)
    prefix = r"(?<!\w)" if _is_word_char(w[0]) else ""
    suffix = r"(?!\w)" if _is_word_char(w[-1]) else ""
    return re.compile(prefix + escaped + suffix, re.IGNORECASE)


def contains_target_word(sentence: str, word: str) -> bool:
    """Literal, case-insensitive, word-boundary match. No stemming/lemmatizing:
    "Store" matches "store"/"Store" but not "Stores"/"storing". Works for
    multi-word phrase targets (e.g. "Drop Off") since the boundary anchors
    only apply at the very start/end of the phrase, not between its words."""
    return _word_pattern(word).search(sentence) is not None


def word_count(sentence: str) -> int:
    return len(sentence.split())


def validate_generated_sentence(sentence_record: dict, word: str) -> tuple[bool, str]:
    """Structural/word-usage validation applied before a candidate is even
    sent to the judge, to save an API call on obviously-broken output."""
    sentence = sentence_record.get("sentence", "")
    if not sentence or not sentence.strip():
        return False, "empty sentence"
    if not contains_target_word(sentence, word):
        return False, f"target word {word!r} not found (exact, word-boundary match)"
    return True, ""


def compute_decision(scores: dict, thresholds: dict) -> tuple[str, float, str]:
    """Recompute PASS/FAIL deterministically from per-dimension scores.

    Returns (decision, overall_score, reason). overall_score is the plain
    mean of the seven dimensions, computed here rather than trusted from the
    judge response.
    """
    missing = [k for k in DIMENSION_KEYS if k not in scores]
    if missing:
        return "FAIL", 0.0, f"judge response missing dimension(s): {missing}"

    values = [float(scores[k]) for k in DIMENSION_KEYS]
    overall_score = sum(values) / len(values)

    failed_dims = [
        f"{dim} {scores[dim]:.2f} < {thresholds[dim]:.2f}"
        for dim in DIMENSION_KEYS
        if dim in thresholds and float(scores[dim]) < thresholds[dim]
    ]
    if failed_dims:
        return "FAIL", overall_score, "; ".join(failed_dims)
    return "PASS", overall_score, ""
