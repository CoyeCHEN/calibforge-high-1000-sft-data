"""Small, JSON-serializable schema shared by rollout adapters.

The schema deliberately keeps each model request separate.  A request's context is
not reconstructed by concatenating later requests, because the harness may compact
or truncate history between calls.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CanonicalCall:
    call_id: str
    request_messages: list[dict[str, Any]] = field(default_factory=list)
    assistant_response: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_observations: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None
    provider_payload: dict[str, Any] | None = None
    context_fidelity: str = "unknown"
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CanonicalTrajectory:
    trajectory_id: str
    source: str
    task_id: str | None = None
    trial_id: str | None = None
    task_group_id: str | None = None
    teacher_model: str | None = None
    model_revision: str | None = None
    harness_revision: str | None = None
    status: str = "unknown"
    reward: float | int | None = None
    reward_status: str = "missing"
    calls: list[CanonicalCall] = field(default_factory=list)
    raw_path: str | None = None
    raw_sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_call(**kwargs: Any) -> dict[str, Any]:
    """Build a call dict while keeping JSON output stable for downstream tools."""
    return CanonicalCall(**kwargs).to_dict()


def canonical_trajectory(**kwargs: Any) -> dict[str, Any]:
    return CanonicalTrajectory(**kwargs).to_dict()
