"""LLM-assisted authoring of conditional text generation rules."""

from __future__ import annotations

import json
from typing import Any, Optional

from .llm import LLMClient, LLMResponseError
from .schemas import TEXT_RULE_AUTHOR_SCHEMA
from .utils import stable_hash

TEMPLATE_KINDS = {"literal", "faker", "regex", "pattern"}


def heuristic_text_rules(field_path: str, summary: dict[str, Any]) -> dict[str, Any]:
    """Produce baseline text templates using deterministic heuristics."""

    dominant_type = _dominant_type(summary)
    if dominant_type != "string":
        return {}

    semantic = (summary.get("semantic_type") or "").lower() if summary.get("semantic_type") else ""
    regex_candidates = summary.get("regex_candidates") or []
    enum_candidates = summary.get("enum_candidates") or []
    required_rate = float(summary.get("required_rate", 1.0))

    templates: list[dict[str, Any]] = []

    if semantic in {"email", "name", "phone"}:
        templates.append(
            {
                "kind": "faker",
                "value": semantic,
                "weight": 1.0,
                "description": f"Generate {semantic} values via Faker.",
            }
        )
    elif enum_candidates:
        candidates = [candidate.get("value") for candidate in enum_candidates if isinstance(candidate, dict)]
        unique_literals = []
        for candidate in candidates:
            if isinstance(candidate, str) and candidate not in unique_literals:
                unique_literals.append(candidate)
            if len(unique_literals) >= 5:
                break
        for literal in unique_literals:
            templates.append(
                {
                    "kind": "literal",
                    "value": literal,
                    "weight": 1.0 / max(len(unique_literals), 1),
                    "description": "Replay high-frequency enum candidate.",
                }
            )
    elif regex_candidates:
        pattern = regex_candidates[0].get("pattern")
        if isinstance(pattern, str):
            templates.append(
                {
                    "kind": "regex",
                    "value": pattern,
                    "weight": 1.0,
                    "description": "Sample string adhering to dominant regex pattern.",
                }
            )
    else:
        templates.append(
            {
                "kind": "faker",
                "value": "pystr",
                "weight": 1.0,
                "description": "Fallback to random alphanumeric string generation.",
            }
        )

    conditions: list[dict[str, Any]] = []
    if required_rate < 1.0:
        conditions.append(
            {
                "when": "missing",
                "template": {
                    "kind": "literal",
                    "value": "N/A",
                    "weight": 1.0,
                    "description": "Fill missing source values with explicit placeholder.",
                },
                "description": "Provide placeholder when original field is null.",
            }
        )

    notes = []
    narratives = summary.get("narratives") or []
    for narrative in narratives[:5]:
        if isinstance(narrative, str):
            notes.append(narrative)

    result: dict[str, Any] = {"templates": templates}
    if conditions:
        result["conditions"] = conditions
    if notes:
        result["notes"] = notes
    return result


def refine_text_rules(profile: dict[str, Any], rules: dict[str, Any], llm: LLMClient) -> None:
    """Ask the LLM for richer conditional templates per field."""

    field_summaries = profile.get("field_summaries", {})
    if not isinstance(field_summaries, dict):
        return

    field_rules = rules.get("fields", {})
    if not isinstance(field_rules, dict):
        return

    for path, summary in field_summaries.items():
        if not isinstance(summary, dict):
            continue
        rule = field_rules.get(path)
        if not isinstance(rule, dict):
            continue
        if (rule.get("type") or _dominant_type(summary)) != "string":
            continue

        llm_rules = _llm_author_text_rule(path, summary, llm)
        if not llm_rules:
            continue

        existing = rule.get("text_rules")
        merged = merge_text_rules(existing, llm_rules)
        rule["text_rules"] = merged


def merge_text_rules(base: Optional[dict[str, Any]], incoming: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Combine and normalise two text rule payloads."""

    normalized_base = _normalize_text_rules(base)
    normalized_incoming = _normalize_text_rules(incoming)

    templates = _dedupe_templates(normalized_base["templates"] + normalized_incoming["templates"])
    conditions = _dedupe_conditions(normalized_base["conditions"] + normalized_incoming["conditions"])

    notes = normalized_base["notes"] + [note for note in normalized_incoming["notes"] if note not in normalized_base["notes"]]

    result: dict[str, Any] = {"templates": templates}
    if conditions:
        result["conditions"] = conditions
    if notes:
        result["notes"] = notes
    return result


def _llm_author_text_rule(field_path: str, summary: dict[str, Any], llm: LLMClient) -> dict[str, Any]:
    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You design conditional text generation rules for synthetic data. "
                    "Given a field summary, propose templates and optional conditional overrides. "
                    "Return JSON matching the provided schema, using supported template kinds "
                    "('literal', 'faker', 'regex', or 'pattern')."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "field_path": field_path,
                        "summary": {
                            "semantic_type": summary.get("semantic_type"),
                            "regex_candidates": summary.get("regex_candidates"),
                            "enum_candidates": summary.get("enum_candidates"),
                            "pii_likelihood": summary.get("pii_likelihood"),
                            "confidence": summary.get("confidence"),
                            "narratives": summary.get("narratives"),
                            "length": summary.get("length"),
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 4096,
    }

    seed = int(stable_hash([field_path])[:16], 16) % 2_147_483_647
    payload["seed"] = seed

    try:
        response = llm.generate_text(payload, schema=TEXT_RULE_AUTHOR_SCHEMA, parse_json=True)
    except (LLMResponseError, Exception):
        return {}

    return _normalize_text_rules(response)


def _normalize_text_rules(payload: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"templates": [], "conditions": [], "notes": []}

    templates = _normalize_templates(payload.get("templates"))
    conditions = _normalize_conditions(payload.get("conditions"))

    notes: list[str] = []
    raw_notes = payload.get("notes") or []
    if isinstance(raw_notes, list):
        for note in raw_notes:
            if isinstance(note, str) and note not in notes:
                notes.append(note)

    return {"templates": templates, "conditions": conditions, "notes": notes}


def _normalize_templates(raw_templates: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_templates, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in raw_templates:
        template: dict[str, Any]
        if isinstance(item, str):
            template = {"kind": "literal", "value": item}
        elif isinstance(item, dict):
            kind = item.get("kind") or item.get("type") or item.get("mode")
            value = item.get("value") or item.get("pattern") or item.get("template")
            if not isinstance(kind, str) or not isinstance(value, str):
                continue
            template = {"kind": kind.lower(), "value": value}
            weight = item.get("weight")
            if isinstance(weight, (int, float)):
                template["weight"] = float(weight)
            description = item.get("description")
            if isinstance(description, str):
                template["description"] = description
        else:
            continue

        if template["kind"] not in TEMPLATE_KINDS:
            if template["kind"] == "template":
                template["kind"] = "pattern"
            else:
                continue

        template.setdefault("weight", 1.0)
        normalized.append(template)

    return _dedupe_templates(normalized)


def _normalize_conditions(raw_conditions: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_conditions, list):
        return []

    normalized: list[dict[str, Any]] = []
    for condition in raw_conditions:
        if not isinstance(condition, dict):
            continue
        when = condition.get("when")
        template_data = condition.get("template")
        if not isinstance(when, str) or not template_data:
            continue

        templates = _normalize_templates([template_data])
        if not templates:
            continue
        template = templates[0]

        normalized_condition = {"when": when, "template": template}
        weight = condition.get("weight")
        if isinstance(weight, (int, float)):
            normalized_condition["weight"] = float(weight)
        description = condition.get("description")
        if isinstance(description, str):
            normalized_condition["description"] = description

        normalized.append(normalized_condition)

    return _dedupe_conditions(normalized)


def _dedupe_templates(templates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped: list[dict[str, Any]] = []
    for template in templates:
        key = (template.get("kind"), template.get("value"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(template)
    return deduped


def _dedupe_conditions(conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped: list[dict[str, Any]] = []
    for condition in conditions:
        template = condition.get("template", {})
        key = (condition.get("when"), template.get("kind"), template.get("value"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(condition)
    return deduped


def _dominant_type(summary: dict[str, Any]) -> Optional[str]:
    types = summary.get("types")
    if not isinstance(types, list) or not types:
        return None
    dominant = max(
        (entry for entry in types if isinstance(entry, dict) and isinstance(entry.get("confidence"), (int, float))),
        key=lambda item: item.get("confidence", 0.0),
        default=None,
    )
    if dominant is None:
        dominant = types[0] if isinstance(types[0], dict) else None
    name = dominant.get("name") if isinstance(dominant, dict) else None
    return str(name) if name else None
