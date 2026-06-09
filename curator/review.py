"""
Persistent lineage-review state.

Tracks which cell lineages the user has already analysed, keyed by the lineage's
ancestral root ID, and persists that set to a small JSON file in the working
directory so it survives between curator sessions (a sibling of the audit log).

Design notes
------------
* A lineage is identified by its root track ID (see ``lineage.root_of`` /
  ``lineage.lineage_roots``). Marking any member as reviewed marks the whole
  family, because review happens per genealogy, not per track.
* All file I/O is best-effort and never raises into the UI: a missing or
  corrupt file simply means "nothing reviewed yet".
* The stored payload also keeps a human-readable timestamp per root so an
  external viewer (or a future panel) can show when each lineage was done.
"""

from __future__ import annotations

import os
import json
import datetime as _dt

REVIEW_FILE = "lineage_review.json"


class LineageReview:
    """Load/save the set of reviewed lineage roots for one working directory."""

    def __init__(self, work_dir):
        self.path = os.path.join(work_dir, REVIEW_FILE) if work_dir else None
        # root_id (int) -> ISO timestamp string when it was marked reviewed.
        self._reviewed: dict[int, str] = {}
        self._load()

    # -- persistence -----------------------------------------------------
    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r") as fh:
                data = json.load(fh)
            raw = data.get("reviewed", {}) if isinstance(data, dict) else {}
            out: dict[int, str] = {}
            for k, v in raw.items():
                try:
                    out[int(k)] = str(v)
                except (TypeError, ValueError):
                    continue
            self._reviewed = out
        except Exception:
            # Corrupt / unreadable file: start clean rather than crash.
            self._reviewed = {}

    def _save(self) -> bool:
        if not self.path:
            return False
        try:
            payload = {
                "reviewed": {str(k): v for k, v in self._reviewed.items()},
                "saved_at": _dt.datetime.now().isoformat(timespec="seconds"),
            }
            # Atomic-ish write: temp file then replace.
            tmp = self.path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(tmp, self.path)
            return True
        except Exception:
            return False

    # -- queries ---------------------------------------------------------
    def is_reviewed(self, root_id: int) -> bool:
        return int(root_id) in self._reviewed

    def reviewed_roots(self) -> set:
        return set(self._reviewed.keys())

    def reviewed_at(self, root_id: int):
        return self._reviewed.get(int(root_id))

    def count(self) -> int:
        return len(self._reviewed)

    # -- mutations (persist immediately) ---------------------------------
    def mark(self, root_id: int) -> bool:
        """Mark a lineage (by root ID) reviewed and persist. Returns saved-ok."""
        self._reviewed[int(root_id)] = _dt.datetime.now().isoformat(timespec="seconds")
        return self._save()

    def unmark(self, root_id: int) -> bool:
        """Remove a lineage's reviewed mark and persist. Returns saved-ok."""
        self._reviewed.pop(int(root_id), None)
        return self._save()

    def toggle(self, root_id: int) -> bool:
        """Flip the reviewed state of a lineage. Returns the NEW state (reviewed?)."""
        rid = int(root_id)
        if rid in self._reviewed:
            self.unmark(rid)
            return False
        self.mark(rid)
        return True

    def prune(self, valid_roots) -> None:
        """Drop reviewed entries whose root no longer exists, then persist.

        Called after operations that can renumber or remove lineages (e.g.
        re-sequencing), so the reviewed set never points at stale IDs.
        """
        valid = {int(r) for r in valid_roots}
        before = len(self._reviewed)
        self._reviewed = {k: v for k, v in self._reviewed.items() if k in valid}
        if len(self._reviewed) != before:
            self._save()

    def progress_text(self, total_lineages: int) -> str:
        n = self.count()
        if total_lineages <= 0:
            return f"{n} lineages reviewed"
        pct = 100.0 * n / total_lineages
        return f"{n} / {total_lineages} lineages reviewed ({pct:.0f}%)"


TRIAGE_REVIEW_FILE = "triage_review.json"


class TriageReview:
    """Persistent set of triage-queue cells the user has ticked as reviewed.

    Same design as :class:`LineageReview` but keyed by individual track ID (the
    triage queue works per cell, not per genealogy) and stored in its own JSON
    sidecar so the two never clobber each other. All file I/O is best-effort and
    never raises into the UI; a missing/corrupt file means "nothing reviewed".
    """

    def __init__(self, work_dir):
        self.path = os.path.join(work_dir, TRIAGE_REVIEW_FILE) if work_dir else None
        # track_id (int) -> ISO timestamp string when it was marked reviewed.
        self._reviewed: dict[int, str] = {}
        self._load()

    # -- persistence -----------------------------------------------------
    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r") as fh:
                data = json.load(fh)
            raw = data.get("reviewed", {}) if isinstance(data, dict) else {}
            out: dict[int, str] = {}
            for k, v in raw.items():
                try:
                    out[int(k)] = str(v)
                except (TypeError, ValueError):
                    continue
            self._reviewed = out
        except Exception:
            self._reviewed = {}

    def _save(self) -> bool:
        if not self.path:
            return False
        try:
            payload = {
                "reviewed": {str(k): v for k, v in self._reviewed.items()},
                "saved_at": _dt.datetime.now().isoformat(timespec="seconds"),
            }
            tmp = self.path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(tmp, self.path)
            return True
        except Exception:
            return False

    # -- queries ---------------------------------------------------------
    def is_reviewed(self, track_id: int) -> bool:
        return int(track_id) in self._reviewed

    def reviewed_ids(self) -> set:
        return set(self._reviewed.keys())

    def reviewed_at(self, track_id: int):
        return self._reviewed.get(int(track_id))

    def count(self) -> int:
        return len(self._reviewed)

    # -- mutations (persist immediately) ---------------------------------
    def mark(self, track_id: int) -> bool:
        self._reviewed[int(track_id)] = _dt.datetime.now().isoformat(timespec="seconds")
        return self._save()

    def unmark(self, track_id: int) -> bool:
        self._reviewed.pop(int(track_id), None)
        return self._save()

    def set_reviewed(self, track_id: int, reviewed: bool) -> bool:
        """Explicitly set a cell's reviewed state and persist."""
        return self.mark(track_id) if reviewed else self.unmark(track_id)

    def prune(self, valid_ids) -> None:
        """Drop reviewed marks for tracks that no longer exist, then persist."""
        valid = {int(t) for t in valid_ids}
        before = len(self._reviewed)
        self._reviewed = {k: v for k, v in self._reviewed.items() if k in valid}
        if len(self._reviewed) != before:
            self._save()
