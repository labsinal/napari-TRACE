"""
Persistent per-cell validation state.

Tracks individual track IDs the user confirmed as correct during a validation
sample (as opposed to whole lineages, which LineageReview handles). Persisted to
a small JSON file in the working directory, sibling to lineage_review.json.

All file I/O is best-effort and never raises into the UI: a missing or corrupt
file simply means "no individually validated cells yet".
"""

from __future__ import annotations

import os
import json
import datetime as _dt

VALIDATION_FILE = "cell_validation.json"


class CellValidation:
    """Load/save the set of individually validated track IDs for a working dir."""

    def __init__(self, work_dir):
        self.path = os.path.join(work_dir, VALIDATION_FILE) if work_dir else None
        self._validated: dict[int, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r") as fh:
                data = json.load(fh)
            raw = data.get("validated", {}) if isinstance(data, dict) else {}
            out: dict[int, str] = {}
            for k, v in raw.items():
                try:
                    out[int(k)] = str(v)
                except (TypeError, ValueError):
                    continue
            self._validated = out
        except Exception:
            self._validated = {}

    def _save(self) -> bool:
        if not self.path:
            return False
        try:
            payload = {
                "validated": {str(k): v for k, v in self._validated.items()},
                "saved_at": _dt.datetime.now().isoformat(timespec="seconds"),
            }
            tmp = self.path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(tmp, self.path)
            return True
        except Exception:
            return False

    def is_validated(self, track_id: int) -> bool:
        return int(track_id) in self._validated

    def ids(self) -> set:
        return set(self._validated.keys())

    def count(self) -> int:
        return len(self._validated)

    def mark(self, track_id: int) -> bool:
        self._validated[int(track_id)] = _dt.datetime.now().isoformat(timespec="seconds")
        return self._save()

    def unmark(self, track_id: int) -> bool:
        self._validated.pop(int(track_id), None)
        return self._save()

    def set(self, track_id: int, validated: bool) -> bool:
        return self.mark(track_id) if validated else self.unmark(track_id)

    def prune(self, valid_ids) -> None:
        valid = {int(t) for t in valid_ids}
        before = len(self._validated)
        self._validated = {k: v for k, v in self._validated.items() if k in valid}
        if len(self._validated) != before:
            self._save()
