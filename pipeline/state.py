"""Persistent run state so a 15k+ word pipeline can be safely interrupted and
resumed without re-processing already-completed words or losing accepted
sentences.

Embeddings are deliberately NOT persisted here (they would make this file
huge at this scale). On resume, orchestrator.py re-embeds only the already
-accepted sentences of words that still need more rounds, which is cheap
compared to a full LLM regeneration pass.
"""

from __future__ import annotations

import json
import os
from typing import Any


class PipelineState:
    def __init__(self, path: str):
        self.path = path
        self._data: dict[str, Any] = {"words": {}}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self._data = json.load(f)

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.path)

    def ensure_word(self, record: dict) -> dict:
        key = str(record["row_id"])
        if key not in self._data["words"]:
            self._data["words"][key] = {
                "row_id": record["row_id"],
                "domain": record["domain"],
                "category": record["category"],
                "word": record["word"],
                "status": "pending",
                "rounds_attempted": 0,
                "accepted": [],
            }
        return self._data["words"][key]

    def get_word(self, row_id: int) -> dict | None:
        return self._data["words"].get(str(row_id))

    def words_needing_round(self, all_records: list[dict], min_passing: int) -> list[dict]:
        """Records that still need more generation rounds: not yet at their
        passing-sentence quota and haven't given up on this word yet."""
        pending = []
        for record in all_records:
            entry = self.ensure_word(record)
            if entry["status"] == "pending" and len(entry["accepted"]) < min_passing:
                pending.append(record)
        return pending

    def record_round_attempt(self, row_id: int) -> None:
        self._data["words"][str(row_id)]["rounds_attempted"] += 1

    def add_accepted(self, row_id: int, sentence_records: list[dict]) -> None:
        self._data["words"][str(row_id)]["accepted"].extend(sentence_records)

    def accepted_sentences(self, row_id: int) -> list[dict]:
        entry = self.get_word(row_id)
        return entry["accepted"] if entry else []

    def next_sentence_suffix(self, row_id: int) -> int:
        """1-based suffix for the next sentence_id to mint for this word,
        continuing the count across generation rounds."""
        entry = self.get_word(row_id)
        return len(entry["accepted"]) + 1 if entry else 1

    def finalize_status(self, row_id: int, min_passing: int, max_rounds: int) -> None:
        entry = self._data["words"][str(row_id)]
        if len(entry["accepted"]) >= min_passing:
            entry["status"] = "complete"
        elif entry["rounds_attempted"] >= max_rounds:
            entry["status"] = "exhausted"
        # else stays "pending" for another round

    def all_entries(self) -> list[dict]:
        return list(self._data["words"].values())

    def set_pending_judge_batch(self, batch_id: str, chunks: list[list[dict]], provider: str) -> None:
        """Legacy single-job form, kept only so old state.json files (and any
        live job already tracked this way) still parse; see
        add_pending_judge_batch for the current multi-job form."""
        self._data["pending_judge_batch"] = {"batch_id": batch_id, "chunks": chunks, "provider": provider}
        self.save()

    def get_pending_judge_batch(self) -> dict | None:
        return self._data.get("pending_judge_batch")

    def clear_pending_judge_batch(self) -> None:
        self._data.pop("pending_judge_batch", None)
        self.save()

    def add_pending_judge_batch(self, batch_id: str, chunks: list[list[dict]], provider: str) -> None:
        """Persist one in-flight judge batch job (of possibly several running
        concurrently) so a restart can reconnect and poll/apply it instead of
        resubmitting (and double-paying for) the same work. `provider` (the
        one actually used to submit it) is stored alongside so a resume
        reconnects correctly even if config.yaml's judge provider setting has
        since changed."""
        pending = self._data.setdefault("pending_judge_batches", [])
        pending.append({"batch_id": batch_id, "chunks": chunks, "provider": provider})
        self.save()

    def get_pending_judge_batches(self) -> list[dict]:
        return self._data.get("pending_judge_batches", [])

    def remove_pending_judge_batch(self, batch_id: str) -> None:
        pending = self._data.get("pending_judge_batches", [])
        self._data["pending_judge_batches"] = [p for p in pending if p["batch_id"] != batch_id]
        self.save()

    def set_pending_generation_batch(self, job_name: str, chunks: list[list[dict]]) -> None:
        """Persist an in-flight Gemini generation batch job so a restart can
        reconnect instead of resubmitting (and double-paying for) the same work."""
        self._data["pending_generation_batch"] = {"job_name": job_name, "chunks": chunks}
        self.save()

    def get_pending_generation_batch(self) -> dict | None:
        return self._data.get("pending_generation_batch")

    def clear_pending_generation_batch(self) -> None:
        self._data.pop("pending_generation_batch", None)
        self.save()

    def set_judge_queue(self, sub_batches: list[list[list[dict]]]) -> None:
        """Persist the full list of not-yet-submitted judge sub-batches for the
        current round (each sub-batch is itself a list of chunks, sized to stay
        under OpenAI's org-wide enqueued-token cap). Submitted one at a time --
        see pending_judge_batch for the one currently in flight."""
        self._data["judge_queue"] = sub_batches
        self.save()

    def get_judge_queue(self) -> list[list[list[dict]]]:
        return self._data.get("judge_queue", [])

    def pop_judge_queue_item(self) -> list[list[dict]] | None:
        queue = self._data.get("judge_queue", [])
        if not queue:
            return None
        item = queue.pop(0)
        self._data["judge_queue"] = queue
        self.save()
        return item

    def push_judge_queue_item_front(self, sub_batch: list[list[dict]]) -> None:
        """Puts a sub-batch back at the front of the queue -- used when a
        submission attempt fails or is deliberately deferred (e.g. to avoid
        exceeding a token budget while other jobs are in flight), so the work
        isn't silently dropped after being popped."""
        queue = self._data.get("judge_queue", [])
        queue.insert(0, sub_batch)
        self._data["judge_queue"] = queue
        self.save()

    def clear_judge_queue(self) -> None:
        self._data.pop("judge_queue", None)
        self.save()

    def summary(self) -> dict:
        statuses: dict[str, int] = {}
        total_accepted = 0
        for entry in self._data["words"].values():
            statuses[entry["status"]] = statuses.get(entry["status"], 0) + 1
            total_accepted += len(entry["accepted"])
        return {"status_counts": statuses, "total_accepted_sentences": total_accepted}
