"""Append-only traces, including failures before a request has a result."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import uuid


class Trace:
    def __init__(self, directory, *, label="request", metadata=None):
        self.id = f"{label}-{uuid.uuid4().hex[:12]}"
        self.directory = Path(directory) / self.id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.sequence = 0
        self.event("start", metadata or {})

    def event(self, kind, data):
        self.sequence += 1
        record = {
            "request_id": self.id, "sequence": self.sequence, "kind": kind,
            "utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic() - self.started, "data": data,
        }
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return self.sequence

    def result(self, value):
        self.event("result", value)
        (self.directory / "result.json").write_text(json.dumps(value, indent=2, ensure_ascii=False))
