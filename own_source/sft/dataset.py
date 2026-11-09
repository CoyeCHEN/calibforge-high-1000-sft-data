"""JSONL I/O and task-group-safe splitting for canonical trajectories."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .validation import ensure_valid


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number}: invalid JSON") from error
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number}: expected object")
        rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def split_by_task_group(trajectories: list[dict[str, Any]], holdout_fraction: float = 0.1) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between zero and one")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in trajectories:
        ensure_valid(row)
        group = row.get("task_group_id") or row.get("task_id") or row["trajectory_id"]
        groups.setdefault(str(group), []).append(row)
    holdout_groups = {group for group in groups if int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < holdout_fraction}
    if not holdout_groups and len(groups) > 1:
        holdout_groups = {sorted(groups)[-1]}
    train = [row for group, rows in groups.items() if group not in holdout_groups for row in rows]
    holdout = [row for group, rows in groups.items() if group in holdout_groups for row in rows]
    return train, holdout
