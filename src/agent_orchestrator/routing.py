"""Parsing and validation for supervisor and peer-routing decisions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    next_agent: str
    reason: str = ""


def parse_routing_decision(content: str, allowed_agents: set[str]) -> RoutingDecision | None:
    """Extract the last JSON object and validate its next-agent field."""
    candidates = re.findall(r"\{[^{}]*\}", content, flags=re.DOTALL)
    for candidate in reversed(candidates):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        next_agent = payload.get("next")
        reason = payload.get("reason", "")
        if not isinstance(next_agent, str):
            continue
        normalized = next_agent.strip().lower()
        if normalized != "finish" and normalized not in allowed_agents:
            continue
        return RoutingDecision(
            next_agent=normalized,
            reason=reason.strip() if isinstance(reason, str) else "",
        )
    return None
