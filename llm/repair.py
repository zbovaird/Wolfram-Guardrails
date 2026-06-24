"""Extract and repair JSON from LLM output."""

from __future__ import annotations

import json
import re
from typing import Any


JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
JSON_OBJECT_PATTERN = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


def normalize_policy_dict(data: Any) -> dict[str, Any] | None:
    """Coerce common LLM policy JSON shapes into a SymbolicDecision-compatible dict."""
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "decision" in item:
                data = item
                break
        else:
            return None
    if not isinstance(data, dict):
        return None

    normalized = dict(data)
    decision = normalized.get("decision")
    if isinstance(decision, list):
        normalized["decision"] = decision[0] if decision else ""
    elif isinstance(decision, dict):
        normalized["decision"] = (
            decision.get("value")
            or decision.get("decision")
            or next(iter(decision.values()), "")
        )

    rules = normalized.get("triggeredRules")
    if isinstance(rules, dict):
        normalized["triggeredRules"] = [str(key) for key in rules]
    elif isinstance(rules, list):
        normalized["triggeredRules"] = [
            str(rule.get("name") or rule.get("rule") or rule)
            if isinstance(rule, dict)
            else str(rule)
            for rule in rules
        ]

    severity = normalized.get("severity")
    if isinstance(severity, list):
        normalized["severity"] = severity[0] if severity else 0.0

    return normalized


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from model output."""
    stripped = text.strip()
    if not stripped:
        return None

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    return item
    except json.JSONDecodeError:
        pass

    fence_match = JSON_FENCE_PATTERN.search(stripped)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    for match in JSON_OBJECT_PATTERN.finditer(stripped):
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    return None


def build_repair_prompt(invalid_output: str, schema_hint: str) -> str:
    return (
        "Your previous response was invalid. Return strict JSON only matching this schema:\n"
        f"{schema_hint}\n\n"
        "Invalid output:\n"
        f"{invalid_output}\n\n"
        "Return corrected JSON only."
    )


SEMANTIC_SCHEMA_HINT = (
    '{"intent":"string","target":"string","techniques":["string"],'
    '"risk":0.0,"notes":"string"}'
)

POLICY_SCHEMA_HINT = (
    '{"decision":"ALLOW|BLOCK|REVIEW|REWRITE","reason":"string",'
    '"triggeredRules":["string"],"severity":0.0}'
)
