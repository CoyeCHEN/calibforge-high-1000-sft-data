"""Independent reader for Open-Harness Recorder v2 exports.

This module mirrors the public interchange format without importing Open-Harness,
so the SFT repository can be installed on a training host by itself.
Incomplete journals retain only the verified prefix and carry integrity issues;
verifier reward never substitutes for run/call/tool closure.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterator

from sft.schema import canonical_trajectory
from sft.security import scrub
from sft.validation import validate_trajectory


SCHEMA = "open-harness-rollout-v2"


def _file_sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as error:
                raise ValueError(f"{path.name}:{number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path.name}:{number}: expected object")
            yield row


def _resolve(value: Any, root: Path) -> Any:
    if isinstance(value, list):
        return [_resolve(item, root) for item in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {"$artifact", "encoding", "size"}:
        ident = value["$artifact"]
        if not isinstance(ident, str) or len(ident) != 64 or any(char not in "0123456789abcdef" for char in ident):
            raise ValueError("invalid artifact reference")
        data = (root / "artifacts" / ident).read_bytes()
        if len(data) != value["size"] or hashlib.sha256(data).hexdigest() != ident:
            raise ValueError(f"corrupt artifact {ident}")
        return data.decode("utf-8") if value.get("encoding") == "utf-8" else data
    return {key: _resolve(item, root) for key, item in value.items()}


def _response_message(response: Any) -> dict[str, Any]:
    if not isinstance(response, dict):
        return {"role": "assistant", "content": response}
    if response.get("role") == "assistant" and not isinstance(response.get("content"), list):
        return dict(response)
    if isinstance(response.get("content"), list):
        text, thinking, calls = [], [], []
        for block in response["content"]:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text":
                text.append(str(block.get("text", "")))
            elif kind == "thinking":
                thinking.append(str(block.get("thinking", "")))
            elif kind == "toolCall":
                arguments = block.get("arguments")
                if isinstance(arguments, str):
                    try:
                        parsed = json.loads(arguments)
                    except json.JSONDecodeError:
                        parsed = None
                    if isinstance(parsed, dict):
                        arguments = parsed
                elif not isinstance(arguments, dict):
                    arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
                calls.append({"id": block.get("id"), "type": "function",
                              "function": {"name": block.get("name"), "arguments": arguments}})
        result = {"role": "assistant", "content": "".join(text)}
        if thinking:
            result["reasoning_content"] = "".join(thinking)
        if calls:
            result["tool_calls"] = calls
        if response.get("stopReason") is not None:
            result["finish_reason"] = response["stopReason"]
        return result
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or choices[0].get("delta")
        if isinstance(message, dict):
            result = dict(message)
            result.setdefault("role", "assistant")
            if choices[0].get("finish_reason") is not None:
                result.setdefault("finish_reason", choices[0]["finish_reason"])
            return result
    return {"role": "assistant", "content": response.get("content", ""),
            **({"tool_calls": response["tool_calls"]} if "tool_calls" in response else {})}


def _tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for item in message.get("tool_calls", []) or []:
        if not isinstance(item, dict):
            result.append({"id": None, "name": None, "arguments": None})
            continue
        function = item.get("function") if isinstance(item.get("function"), dict) else item
        result.append({"id": item.get("id"), "name": function.get("name"),
                       "arguments": function.get("arguments")})
    return result


def _normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    role = message.get("role")
    if role == "toolResult":
        return {"role": "tool", "tool_call_id": message.get("toolCallId"), "content": message.get("content")}
    if role == "assistant" and isinstance(message.get("content"), list):
        return _response_message(message)
    result = dict(message)
    if role == "assistant" and isinstance(result.get("tool_calls"), list):
        normalized_calls = []
        for item in result["tool_calls"]:
            if not isinstance(item, dict):
                raise ValueError("assistant tool call must be an object")
            function = item.get("function")
            if not isinstance(function, dict):
                raise ValueError("assistant tool call is missing function")
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as error:
                    raise ValueError("assistant tool call arguments are invalid JSON") from error
                if not isinstance(arguments, dict):
                    raise ValueError("assistant tool call arguments must be a JSON object")
            elif not isinstance(arguments, dict):
                raise ValueError("assistant tool call arguments must be a mapping")
            normalized = dict(item)
            normalized["function"] = {**function, "arguments": arguments}
            normalized_calls.append(normalized)
        result["tool_calls"] = normalized_calls
    return result


def _messages(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict) and isinstance(value.get("messages"), list):
        value = value["messages"]
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError("call input does not contain a messages list")
    return [_normalize_message(item) for item in value]


def _raw_rows(directory: Path):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError("unsupported Open-Harness rollout schema")
    integrity: dict[str, Any] = {"issues": [], "source_sha256": {}}
    events_path = directory / "events.jsonl"
    calls, tools, runs = {}, {}, {}
    if events_path.exists():
        integrity["verified_prefix_events"] = 0
        integrity["source_sha256"]["events.jsonl"] = _file_sha256(events_path)
        try:
            _resolve(manifest, directory)
            for index, event in enumerate(_jsonl(events_path), 1):
                if event.get("seq") != index or not isinstance(event.get("data"), dict):
                    raise ValueError(f"invalid event sequence at {index}")
                kind, data = event["type"], _resolve(event["data"], directory)
                if kind.startswith(("call_", "tool_")):
                    row = (calls.get(data.get("call_id")) if kind.startswith("call_")
                           else tools.get(data.get("execution_id")))
                    if row is None and kind not in ("call_start", "tool_start"):
                        raise ValueError(f"orphan {kind}")
                    parent = calls.get(data.get("call_id")) if kind == "tool_start" else row
                    run_id = data.get("run_id") or (parent or {}).get("run_id")
                    if run_id not in runs or "ended_at" in runs[run_id]:
                        raise ValueError(f"{kind} outside an active run")
                    if parent and parent.get("run_id") != run_id:
                        raise ValueError(f"{kind} run mismatch")
                    if row and "ended_at" in row:
                        raise ValueError(f"{kind} after terminal event")
                    if kind == "tool_start" and parent is None:
                        raise ValueError("orphan tool_start")
                if kind in ("run_start", "call_start", "tool_start"):
                    key, target = {"run_start": ("run_id", runs), "call_start": ("call_id", calls),
                                   "tool_start": ("execution_id", tools)}[kind]
                    ident = data.get(key)
                    if not ident or ident in target:
                        raise ValueError(f"missing or duplicate {kind} ID")
                    target[ident] = {**data, "status": "interrupted", "started_at": event["timestamp"],
                                     "session_id": manifest.get("session_id")}
                    if kind == "tool_start":
                        target[ident]["run_id"] = run_id
                elif kind in ("run_end", "call_end", "tool_end"):
                    key, target = {"run_end": ("run_id", runs), "call_end": ("call_id", calls),
                                   "tool_end": ("execution_id", tools)}[kind]
                    ident = data.get(key)
                    if ident not in target or "ended_at" in target[ident]:
                        raise ValueError(f"orphan or duplicate {kind}")
                    target[ident].update(data, ended_at=event["timestamp"])
                elif kind == "call_payload":
                    calls[data["call_id"]].update(data)
                elif kind in ("tool_prepared", "tool_raw", "tool_parent"):
                    source, target = {"tool_prepared": ("actual_args", "actual_args"),
                                      "tool_raw": ("result", "raw_result"),
                                      "tool_parent": ("parent_execution_id", "parent_execution_id")}[kind]
                    tools[data["execution_id"]][target] = data[source]
                elif kind not in {"call_chunk", "call_event", "tool_chunk", "tool_artifact", "tool_update", "context_event", "origin", "child_start", "child_end", "child_origin"}:
                    raise ValueError(f"unknown rollout event {kind}")
                integrity["verified_prefix_events"] = index
        except (ValueError, OSError, KeyError) as error:
            # Keep only the checked prefix. Never rewrite or repair the source journal.
            integrity["issues"].append(str(error))
        return manifest, list(calls.values()), list(tools.values()), list(runs.values()), integrity
    calls_path = directory / "calls.jsonl"
    if not calls_path.exists():
        raise ValueError("rollout has neither events.jsonl nor calls.jsonl")
    call_rows, tool_rows = [], []
    try:
        manifest = _resolve(manifest, directory)
        for filename, rows in (("calls.jsonl", call_rows), ("tools.jsonl", tool_rows)):
            path = directory / filename
            if path.exists():
                integrity["source_sha256"][filename] = _file_sha256(path)
                for row in _jsonl(path):
                    rows.append(_resolve(row, directory))
    except (ValueError, OSError) as error:
        integrity["issues"].append(str(error))
    run_rows = [{**run, "session_id": session.get("session_id")} for session in manifest.get("sessions", [manifest])
                for run in session.get("runs", [])]
    return manifest, call_rows, tool_rows, run_rows, integrity


def _closure_issues(runs, calls, tools):
    issues = []
    run_by_id = {row.get("run_id"): row for row in runs}
    call_by_id = {row.get("call_id"): row for row in calls}
    if len(call_by_id) != len(calls) or None in call_by_id:
        issues.append("missing or duplicate call ID")
    if not runs:
        issues.append("missing run_start/run_end evidence")
    if len(run_by_id) != len(runs) or None in run_by_id:
        issues.append("missing or duplicate run ID")
    execution_ids = [row.get("execution_id") for row in tools]
    if None in execution_ids or len(set(execution_ids)) != len(execution_ids):
        issues.append("missing or duplicate tool execution ID")
    for run in runs:
        if "started_at" not in run or "ended_at" not in run:
            issues.append(f"missing run_start/run_end {run.get('run_id')}")
        if run.get("status") != "completed":
            issues.append(f"unsuccessful run {run.get('run_id')}: {run.get('status')}")
    for call in calls:
        if call.get("run_id") not in run_by_id or "started_at" not in call or "ended_at" not in call or "response" not in call:
            issues.append(f"unclosed call or missing run {call.get('call_id')}")
        run = run_by_id.get(call.get("run_id"), {})
        if all("started_at" in row and "ended_at" in row for row in (run, call)) and not (
                run["started_at"] <= call["started_at"] <= call["ended_at"] <= run["ended_at"]):
            issues.append(f"call outside run bounds {call.get('call_id')}")
    for tool in tools:
        parent = call_by_id.get(tool.get("call_id"))
        if parent is None or (tool.get("run_id") is not None and tool["run_id"] != parent.get("run_id")):
            issues.append(f"orphan tool or run mismatch {tool.get('execution_id')}")
        if "raw_result" not in tool and "result" not in tool:
            issues.append(f"missing tool observation {tool.get('execution_id')}")
        if "started_at" not in tool or "ended_at" not in tool or tool.get("status") not in {"completed", "error"}:
            issues.append(f"unclosed tool {tool.get('execution_id')}")
        run = run_by_id.get((parent or {}).get("run_id"), {})
        if all("started_at" in row and "ended_at" in row for row in (run, tool)) and not (
                run["started_at"] <= tool["started_at"] <= tool["ended_at"] <= run["ended_at"]):
            issues.append(f"tool outside run bounds {tool.get('execution_id')}")
    return issues


def _reward(metadata: dict[str, Any] | None, trial_dir: str | Path | None) -> tuple[Any, str, dict[str, Any]]:
    def numeric(value: Any) -> Any:
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            return value
        if isinstance(value, dict):
            if "reward" in value:
                return numeric(value["reward"])
            values = [item for item in value.values() if isinstance(item, (int, float))]
            if len(values) == 1:
                return numeric(values[0])
        return None

    value = None
    extra: dict[str, Any] = {}
    if metadata:
        extra.update(metadata)
        verifier = metadata.get("verifier_result") if isinstance(metadata.get("verifier_result"), dict) else {}
        rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
        candidate = rewards if rewards else metadata.get("reward")
        value = numeric(candidate)
        if candidate is not None and value is None:
            return None, "invalid", extra
    if value is None and trial_dir:
        root = Path(trial_dir)
        for candidate in (root / "verifier" / "reward.json", root / "verifier" / "reward.txt"):
            if candidate.exists():
                text = candidate.read_text(encoding="utf-8").strip()
                try:
                    parsed = json.loads(text) if candidate.suffix == ".json" else float(text)
                except (ValueError, json.JSONDecodeError):
                    parsed = None
                parsed_value = numeric(parsed)
                if parsed_value is not None:
                    value = parsed_value
                    break
    if not isinstance(value, (int, float)):
        return None, "missing", extra
    return value, ("positive" if value > 0 else "zero" if value == 0 else "negative"), extra


def _convert(manifest: dict[str, Any], calls: list[dict[str, Any]], tools: list[dict[str, Any]], directory: Path,
             *, runs=(), integrity=None, metadata: dict[str, Any] | None = None, trial_dir: str | Path | None = None) -> dict[str, Any]:
    sessions = manifest.get("sessions") if isinstance(manifest.get("sessions"), list) else [manifest]
    session_ids = {row.get("session_id") for row in sessions if isinstance(row, dict)}
    groups: dict[Any, list[dict[str, Any]]] = {ident: [] for ident in session_ids if ident}
    if not groups:
        groups = {manifest.get("session_id", "default"): calls}
    else:
        for row in calls:
            groups.setdefault(row.get("session_id", next(iter(groups))), []).append(row)
    # load_rollout is intentionally single-session; exported bundles can be
    # handled by load_rollouts below.
    ident = next(iter(groups))
    call_rows = groups[ident]
    call_rows.sort(key=lambda row: row.get("started_at", ""))
    tool_by_call: dict[str, list[dict[str, Any]]] = {}
    for tool in tools:
        tool_by_call.setdefault(tool.get("call_id"), []).append(tool)
    canonical_calls = []
    for row in call_rows:
        response = _response_message(row.get("response"))
        calls_for_message = _tool_calls(response)
        observations = []
        for tool in tool_by_call.get(row.get("call_id"), []):
            tool_id = tool.get("tool_call_id")
            observations.append({"tool_call_id": tool_id, "name": tool.get("name"),
                                 "content": tool.get("raw_result", tool.get("result")), "is_error": bool(tool.get("is_error", False)),
                                 "execution_id": tool.get("execution_id")})
        provider_payload = scrub(row.get("payload") or row.get("call_payload"))
        # ``call_start.input`` is the harness-level context and may omit the
        # system prompt.  A provider-payload event records the exact messages
        # sent to the model, including system/history/tool turns; use it when
        # available and retain the start input as the compatibility fallback.
        payload_messages = (
            provider_payload.get("messages")
            if isinstance(provider_payload, dict)
            else None
        )
        request_messages = _messages(
            payload_messages if isinstance(payload_messages, list) else row.get("input", [])
        )
        canonical_calls.append({"call_id": row.get("call_id"), "request_messages": request_messages,
                                "assistant_response": response, "tool_calls": calls_for_message,
                                "tool_observations": observations,
                                "finish_reason": response.get("finish_reason", row.get("status")),
                                "provider_payload": provider_payload,
                                "context_fidelity": row.get("fidelity", "context_only"),
                                "flags": (["failed_call"] if row.get("status") not in {"completed", "stop", "toolUse", "end", "success"} else [])})
    session_metadata = {}
    for session in sessions:
        if isinstance(session, dict) and session.get("session_id") == ident:
            session_metadata = dict(session.get("metadata") or {})
            break
    md = dict(manifest.get("metadata") or {})
    md.update(session_metadata)
    md.update(metadata or {})
    reward, reward_status, reward_meta = _reward(md, trial_dir)
    md.update(reward_meta)
    recorder_integrity = dict(integrity or {})
    recorder_integrity["issues"] = list(recorder_integrity.get("issues", [])) + _closure_issues(runs, call_rows, tools)
    md["recorder_integrity"] = recorder_integrity
    status = "completed" if canonical_calls and not recorder_integrity["issues"] and not any("failed_call" in call["flags"] for call in canonical_calls) else "incomplete"
    result = canonical_trajectory(trajectory_id=str(md.get("trajectory_id") or ident), source="openharness_v2",
        task_id=md.get("task_id"), trial_id=md.get("trial_id"), task_group_id=md.get("task_group_id") or md.get("task_id"),
        teacher_model=md.get("teacher_model") or (call_rows[0].get("model") if call_rows else None),
        model_revision=md.get("model_revision"), harness_revision=md.get("harness_revision"), status=status,
        reward=reward, reward_status=reward_status, calls=canonical_calls, raw_path=str(directory),
        raw_sha256=hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest(), metadata=scrub(md))
    issues = validate_trajectory(result)
    if issues:
        result["metadata"]["validation_issues"] = [{"path": issue.path, "message": issue.message} for issue in issues]
        if result["status"] == "completed":
            result["status"] = "incomplete"
    return result


def load_rollouts(path: str | Path, *, metadata: dict[str, Any] | None = None, trial_dir: str | Path | None = None) -> list[dict[str, Any]]:
    directory = Path(path)
    manifest, calls, tools, runs, integrity = _raw_rows(directory)
    sessions = {row.get("session_id") for row in calls + runs if row.get("session_id")}
    if len(sessions) <= 1:
        return [_convert(manifest, calls, tools, directory, runs=runs, integrity=integrity, metadata=metadata, trial_dir=trial_dir)]
    return [_convert({**manifest, "session_id": session, "sessions": [item for item in manifest.get("sessions", []) if item.get("session_id") == session]},
                     [row for row in calls if row.get("session_id") == session],
                     [row for row in tools if row.get("session_id") == session], directory,
                     runs=[row for row in runs if row.get("session_id") == session], integrity=integrity,
                     metadata=metadata, trial_dir=trial_dir)
            for session in sorted(sessions)]


def load_rollout(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    rows = load_rollouts(path, **kwargs)
    if len(rows) != 1:
        raise ValueError(f"expected one rollout, found {len(rows)}")
    return rows[0]
