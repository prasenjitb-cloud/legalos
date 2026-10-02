"""Resolve explicitly named Acts to their pinned source corpora."""

import json
from pathlib import Path
import re


class ActCatalog:
    def __init__(self, path):
        self.path = Path(path)
        self.entries = json.loads(self.path.read_text())
        identifiers = [entry["act_id"] for entry in self.entries]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Duplicate Act identifiers in catalog")

    def resolve(self, question):
        matches = [entry for entry in self.entries if any(
            re.search(r"\b" + re.escape(alias) + r"\b", question, re.I)
            for alias in entry["aliases"])]
        if len(matches) != 1:
            return None, {"status": "CLARIFY", "matches": [e["act_id"] for e in matches],
                          "clarification": "Name one Act for this request; the POC resolves one Act at a time."}
        entry = matches[0]
        path = Path(entry["corpus"])
        if not path.is_absolute():
            path = self.path.parent / path
        return path, {"status": "RESOLVED", "act_id": entry["act_id"], "act_name": entry["act_name"], "corpus": str(path)}
