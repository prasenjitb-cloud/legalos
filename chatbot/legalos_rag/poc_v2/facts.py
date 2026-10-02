"""Explicit user facts for the POC's worker-count condition."""

import re


COUNT_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
               "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
               "eleven": 11, "twelve": 12}
COUNT_PATTERN = re.compile(
    r"(?:^|[.!?]\s+)(?:we have|(?:my|our) (?:posh )?(?:workplace|company|establishment) (?:has|employs)) "
    r"(\d+|" + "|".join(COUNT_WORDS) + r")\s+(?:workers|employees)\b", re.I)


def worker_count(question):
    matches = []
    for match in COUNT_PATTERN.finditer(question):
        tail = re.split(r"[.!?]", question[match.end():], maxsplit=1)[0]
        if question[match.end():].lstrip().startswith("?") or re.search(r"\b(?:if|or|maybe|perhaps)\b", tail, re.I):
            continue
        matches.append(match)
    values = {int(m.group(1)) if m.group(1).isdigit() else COUNT_WORDS[m.group(1).lower()] for m in matches}
    return {"name": "worker_count", "status": "KNOWN" if len(values) == 1 else "UNKNOWN",
            "value": next(iter(values)) if len(values) == 1 else None,
            "source_quotes": [m.group(0).strip() for m in matches],
            "reason": "Conflicting stated counts" if len(values) > 1 else "No supported count statement" if not values else "Explicit count statement"}


def required_worker_count(question, part):
    personal = re.search(r"\b(?:my|our)\s+(?:posh\s+)?(?:workplace|company|establishment)\b|\bwe have\b", question, re.I)
    relevant = re.search(r"\b(?:worker|employee)[ -]count\b|\bsection\s+6\b|\b(?:fewer|less)[ -]than[ -](?:ten|10)", part, re.I)
    applies = re.search(r"\b(?:meet|meets|met|qualif\w*|covered|fall|falls|come|comes)\b", question, re.I)
    return worker_count(question) if personal and relevant and applies else None


def required_user_facts(question, part):
    facts = []
    count = required_worker_count(question, part)
    if count:
        facts.append({**count, "question": "What is the actual number of workers at your workplace?"})
    personal_time = re.search(r"\b(?:am i|can i|i am|i still)\b", question, re.I)
    deadline = re.search(r"\b(?:time|period|late|deadline)\b", part, re.I)
    supplied_date = re.search(r"\b\d{4}-\d{1,2}-\d{1,2}\b|\b\d+\s+(?:days?|weeks?|months?|years?)\s+ago\b|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\b", question, re.I)
    if personal_time and deadline and not supplied_date:
        facts.append({"name": "incident_date", "status": "UNKNOWN", "value": None, "source_quotes": [],
                      "reason": "No incident date supplied for a personal filing-time question",
                      "question": "When did the incident, or the last incident in the series, occur?"})
    return facts


def worker_condition_units(units):
    return [uid for uid, unit in units.items()
            if re.search(r"\bworkers\b", unit["text"], re.I)
            and re.search(r"\bless\s+than\s+ten\b", unit["text"], re.I)]
