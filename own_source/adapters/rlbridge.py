"""Lightweight parser for RLBridge Recorder JSON and adjacent reward logs."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from sft.schema import canonical_trajectory
from sft.security import scrub
from sft.validation import validate_trajectory


def _reward(path: Path) -> tuple[float | int | None, str]:
    log = path.with_suffix(".log")
    if not log.exists():
        return None, "missing"
    match = re.search(r"^REWARD:\s*([-+]?\d+(?:\.\d+)?)", log.read_text(encoding="utf-8"), re.MULTILINE)
    if not match:
        return None, "unscored"
    value = float(match.group(1))
    if value.is_integer():
        value = int(value)
    return value, "positive" if value > 0 else "zero" if value == 0 else "negative"


def _tool_call(item: dict[str, Any]) -> dict[str, Any]:
    function = item.get("function") if isinstance(item.get("function"), dict) else item
    return {"id": item.get("id"), "name": function.get("name"), "arguments": function.get("arguments")}


def load_rollout(path: str | Path, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("turns"), list):
        raise ValueError("RLBridge JSON must contain a turns list")
    base_messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
    all_tool_results = {message.get("tool_call_id"): message for message in base_messages
                        if isinstance(message, dict) and message.get("role") == "tool"}
    calls = []
    for index, turn in enumerate(payload["turns"]):
        if not isinstance(turn, dict):
            raise ValueError(f"turn {index} is not an object")
        response_text = turn.get("actual_response_text", "")
        raw_tool_calls = turn.get("actual_tool_calls") or []
        tool_calls = [_tool_call(item) for item in raw_tool_calls if isinstance(item, dict)]
        observations = []
        for item in tool_calls:
            result = all_tool_results.get(item.get("id"))
            if result:
                observations.append({"tool_call_id": item["id"], "content": result.get("content"),
                                     "name": item.get("name"), "is_error": False})
        structured = turn.get("actual_prompt_messages")
        if isinstance(structured, list):
            request_messages = structured
            fidelity = "exact"
        elif index == 0 and base_messages:
            request_messages = base_messages
            fidelity = "context_only"
        else:
            # actual_prompt_text is retained as metadata, but assigning it a
            # fabricated user role would make an invalid SFT sample look valid.
            request_messages = []
            fidelity = "incomplete"
        response = {"role": "assistant", "content": response_text}
        if tool_calls:
            response["tool_calls"] = [{"id": item["id"], "type": "function",
                                       "function": {"name": item["name"], "arguments": item["arguments"]}}
                                      for item in tool_calls]
        calls.append({"call_id": f"{payload.get('rollout_id', source.stem)}:turn:{index}",
                      "request_messages": request_messages, "assistant_response": response,
                      "tool_calls": tool_calls, "tool_observations": observations,
                      "finish_reason": turn.get("finish_reason"), "provider_payload": None,
                      "context_fidelity": fidelity,
                      "flags": (["missing_observation"] if len(observations) != len(tool_calls) else []) +
                               (["actual_prompt_text_only"] if not structured else [])})
    reward, reward_status = _reward(source)
    md = dict(metadata or {})
    md.setdefault("collector_rollout_id", payload.get("rollout_id"))
    md.setdefault("actual_prompt_text_available", any("actual_prompt_text" in turn for turn in payload["turns"] if isinstance(turn, dict)))
    result = canonical_trajectory(trajectory_id=str(payload.get("rollout_id") or source.stem), source="rlbridge",
        task_id=md.get("task_id"), trial_id=md.get("trial_id"), task_group_id=md.get("task_group_id") or md.get("task_id"),
        teacher_model=payload.get("model"), model_revision=md.get("model_revision"), harness_revision=md.get("harness_revision"),
        status="completed" if calls else "incomplete", reward=reward, reward_status=reward_status, calls=calls,
        raw_path=str(source), raw_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), metadata=scrub(md))
    issues = validate_trajectory(result)
    if issues:
        result["metadata"]["validation_issues"] = [{"path": issue.path, "message": issue.message} for issue in issues]
        result["status"] = "incomplete"
    return result
