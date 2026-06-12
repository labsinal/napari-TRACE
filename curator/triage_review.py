"""
Persistent per-cell triage-review state.

Remembers which cells in the triage queue the user has already checked off, so
the marks survive across sessions. Stored as a small JSON file in the working
directory (created on first use, so old curations get it automatically the first
time they are reopened). All file I/O is best-effort and never raises into the
UI.
"""

from __future__ import annotations

import os
import json
import datetime as _dt

TRIAGE_REVIEW_FILE = "triage_review.json"


class TriageReview:
    """Load/save the set of triage-checked track IDs for a working directory."""

    def __init__(self, work_dir):
        self.path = os.path.join(work_dir, TRIAGE_REVIEW_FILE) if work_dir else None
        self._checked: dict[int, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r") as fh:
                data = json.load(fh)
            raw = data.get("checked", {}) if isinstance(data, dict) else {}
            out: dict[int, str] = {}
            for k, v in raw.items():
                try:
                    out[int(k)] = str(v)
                except (TypeError, ValueError):
                    continue
            self._checked = out
        except Exception:
            self._checked = {}

    def _save(self) -> bool:
        if not self.path:
            return False
        try:
            payload = {
                "checked": {str(k): v for k, v in self._checked.items()},
                "saved_at": _dt.datetime.now().isoformat(timespec="seconds"),
            }
            tmp = self.path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(tmp, self.path)
            return True
        except Exception:
            return False

    def is_checked(self, track_id: int) -> bool:
        return int(track_id) in self._checked

    def ids(self) -> set:
        return set(self._checked.keys())

    def count(self) -> int:
        return len(self._checked)

    def set(self, track_id: int, checked: bool) -> bool:
        if checked:
            self._checked[int(track_id)] = _dt.datetime.now().isoformat(timespec="seconds")
        else:
            self._checked.pop(int(track_id), None)
        return self._save()

    def prune(self, valid_ids) -> None:
        valid = {int(t) for t in valid_ids}
        before = len(self._checked)
        self._checked = {k: v for k, v in self._checked.items() if k in valid}
        if len(self._checked) != before:
            self._save()
