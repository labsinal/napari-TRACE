"""
Audit log.

Appends one line per curation action with timestamp, action type and the IDs
involved, for reproducibility and a session history. Pure file I/O; safe to use
from anywhere.
"""

from __future__ import annotations

import os
import csv
import datetime as _dt

AUDIT_FILE = "curation_audit.log.csv"


class AuditLog:
    def __init__(self, work_dir):
        self.path = os.path.join(work_dir, AUDIT_FILE) if work_dir else None
        if self.path and not os.path.exists(self.path):
            try:
                with open(self.path, "w", newline="") as fh:
                    csv.writer(fh).writerow(["timestamp", "action", "ids", "detail"])
            except Exception:
                self.path = None

    def record(self, action, ids=None, detail=""):
        if not self.path:
            return
        ts = _dt.datetime.now().isoformat(timespec="seconds")
        id_str = ",".join(str(int(i)) for i in ids) if ids else ""
        try:
            with open(self.path, "a", newline="") as fh:
                csv.writer(fh).writerow([ts, action, id_str, detail])
        except Exception:
            pass
