"""Strict checks used before a trajectory can become an SFT sample."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str


class TrajectoryInvalid(ValueError):
    def __init__(self, issues: Iterable[ValidationIssue]):
        self.issues = list(issues)
        detail = "; ".join(f"{item.path}: {item.message}" for item in self.issues)
        super().__init__(detail or "invalid trajectory")


_STATUSES = {"completed", "failed", "interrupted", "aborted", "incomplete", "unknown"}
_REWARD_STATUSES = {"missing", "unscored", "zero", "positive", "negative", "invalid"}


def _issues_for_call(call: Any, prefix: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not isinstance(call, dict):
        return [ValidationIssue(prefix, "call must be an object")]
    if not isinstance(call.get("call_id"), str) or not call["call_id"]:
        issues.append(ValidationIssue(prefix + ".call_id", "missing call ID"))
    messages = call.get("request_messages")
    if not isinstance(messages, list):
        issues.append(ValidationIssue(prefix + ".request_messages", "must be a list"))
    else:
        for index, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant", "tool"}:
                issues.append(ValidationIssue(f"{prefix}.request_messages[{index}]", "invalid message role"))
    if not isinstance(call.get("assistant_response"), dict):
        issues.append(ValidationIssue(prefix + ".assistant_response", "must be an object"))
    tool_calls = call.get("tool_calls", [])
    observations = call.get("tool_observations", [])
    if not isinstance(tool_calls, list) or not isinstance(observations, list):
        issues.append(ValidationIssue(prefix, "tool_calls and tool_observations must be lists"))
        return issues
    call_ids: set[str] = set()
    for index, item in enumerate(tool_calls):
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
            issues.append(ValidationIssue(f"{prefix}.tool_calls[{index}]", "missing tool call ID"))
            continue
        if item["id"] in call_ids:
            issues.append(ValidationIssue(f"{prefix}.tool_calls[{index}]", "duplicate tool call ID"))
        call_ids.add(item["id"])
        if not isinstance(item.get("name"), str) or not item["name"]:
            issues.append(ValidationIssue(f"{prefix}.tool_calls[{index}].name", "missing tool name"))
        if "arguments" not in item:
            issues.append(ValidationIssue(f"{prefix}.tool_calls[{index}].arguments", "missing arguments"))
    seen_obs: set[str] = set()
    for index, item in enumerate(observations):
        if not isinstance(item, dict) or not isinstance(item.get("tool_call_id"), str) or not item["tool_call_id"]:
            issues.append(ValidationIssue(f"{prefix}.tool_observations[{index}]", "missing tool_call_id"))
            continue
        ident = item["tool_call_id"]
        if ident in seen_obs:
            issues.append(ValidationIssue(f"{prefix}.tool_observations[{index}]", "duplicate observation"))
        seen_obs.add(ident)
        if ident not in call_ids:
            issues.append(ValidationIssue(f"{prefix}.tool_observations[{index}]", "orphan observation"))
    missing = call_ids - seen_obs
    if missing:
        issues.append(ValidationIssue(prefix + ".tool_observations", "missing observation for " + ", ".join(sorted(missing))))
    return issues


def validate_trajectory(trajectory: dict[str, Any], *, require_completed: bool = False) -> list[ValidationIssue]:
    """Return all issues; callers can use :func:`ensure_valid` to raise."""
    issues: list[ValidationIssue] = []
    if not isinstance(trajectory, dict):
        return [ValidationIssue("trajectory", "must be an object")]
    for key in ("trajectory_id", "source", "status", "reward_status", "calls"):
        if key not in trajectory:
            issues.append(ValidationIssue(key, "required"))
    if not isinstance(trajectory.get("trajectory_id"), str) or not trajectory.get("trajectory_id"):
        issues.append(ValidationIssue("trajectory_id", "must be non-empty"))
    if trajectory.get("status") not in _STATUSES:
        issues.append(ValidationIssue("status", "unknown status"))
    if trajectory.get("reward_status") not in _REWARD_STATUSES:
        issues.append(ValidationIssue("reward_status", "unknown reward status"))
    reward = trajectory.get("reward")
    if trajectory.get("reward_status") in {"missing", "unscored"} and reward is not None:
        issues.append(ValidationIssue("reward", "must be null when reward is missing/unscored"))
    if trajectory.get("reward_status") in {"zero", "positive", "negative"} and (
            not isinstance(reward, (int, float)) or isinstance(reward, bool) or not math.isfinite(reward)):
        issues.append(ValidationIssue("reward", "finite numeric reward required"))
    for issue in trajectory.get("metadata", {}).get("recorder_integrity", {}).get("issues", []):
        issues.append(ValidationIssue("recorder_integrity", issue))
    calls = trajectory.get("calls")
    if not isinstance(calls, list) or not calls:
        issues.append(ValidationIssue("calls", "must contain at least one call"))
    else:
        ids: set[str] = set()
        for index, call in enumerate(calls):
            issues.extend(_issues_for_call(call, f"calls[{index}]"))
            if isinstance(call, dict) and isinstance(call.get("call_id"), str):
                if call["call_id"] in ids:
                    issues.append(ValidationIssue(f"calls[{index}].call_id", "duplicate call ID"))
                ids.add(call["call_id"])
    if require_completed and (trajectory.get("status") != "completed" or trajectory.get("reward_status") != "positive"):
        issues.append(ValidationIssue("trajectory", "not a verified positive completed rollout"))
    return issues


def ensure_valid(trajectory: dict[str, Any], *, require_completed: bool = False) -> dict[str, Any]:
    issues = validate_trajectory(trajectory, require_completed=require_completed)
    if issues:
        raise TrajectoryInvalid(issues)
    return trajectory


def successful_only(trajectories: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only complete, positive-reward, structurally valid trajectories."""
    result = []
    for trajectory in trajectories:
        ensure_valid(trajectory, require_completed=True)
        result.append(trajectory)
    return result
