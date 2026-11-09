"""Minimal redaction for exported public training artifacts."""
from __future__ import annotations

import re
from typing import Any


_KEY = re.compile(r"(?:api[_-]?key|authorization|access[_-]?token|secret|password)", re.I)
_TOKEN = re.compile(r"\b(?:sk|hf|ghp|github_pat)[_-][A-Za-z0-9_-]{12,}\b")


def scrub(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: "[REDACTED]" if _KEY.search(str(key)) else scrub(item) for key, item in value.items()}
    if isinstance(value, list):
        return [scrub(item) for item in value]
    if isinstance(value, str):
        return _TOKEN.sub("[REDACTED]", value)
    return value
