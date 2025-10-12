"""Rule synthesis from profiling output."""

from __future__ import annotations

import json
from typing import Any, Optional

from . import text_rules as text_rules_module
from .llm import LLMClient, LLMResponseError
from .schemas import RULE_REFINEMENT_SCHEMA


def synthesize_rules(profile: dict[str, Any]) -> dict[str, Any]:
    """Construct a ruleset from profiling metadata."""

    field_summaries: dict[str, Any] = profile.get("field_summaries", {})
    rules: dict[str, Any] = {"fields": {}}

    for path, summary in field_summaries.items():
        types = summary.get("types", [])
        if not types:
            continue
        dominant = max(types, key=lambda item: item.get("confidence", 0.0))
        rule: dict[str, Any] = {
            "type": dominant.get("name", "string"),
            "required_rate": summary.get("required_rate", 0.0),
        }

        enum_candidates = summary.get("enum_candidates", [])
        if enum_candidates:
            rule["enum"] = [candidate["value"] for candidate in enum_candidates]

        semantic_type = summary.get("semantic_type")
        if semantic_type:
            rule["semantic_type"] = semantic_type

        regex_candidates = summary.get("regex_candidates") or []
        if regex_candidates:
            best = max(regex_candidates, key=lambda item: item.get("support", 0))
            rule["regex"] = best.get("pattern")

        pii_likelihood = summary.get("pii_likelihood")
        if pii_likelihood is not None:
            rule["pii_likelihood"] = pii_likelihood

        numeric = summary.get("numeric", {})
        if numeric:
            rule["numeric"] = {
                "min": numeric.get("min"),
                "max": numeric.get("max"),
                "mean": numeric.get("mean"),
                "std": numeric.get("std"),
            }

        length = summary.get("length", {})
        if length:
            rule["length"] = {
                "min": length.get("min", 0),
                "max": length.get("max", length.get("p95", length.get("p75", 32))),
            }

        datetime_info = summary.get("datetime", {})
        if datetime_info:
            rule["datetime"] = {
                "min": datetime_info.get("min"),
                "max": datetime_info.get("max"),
            }

        array_info = summary.get("array", {})
        if array_info and any(value is not None for value in array_info.values()):
            rule["array"] = {
                key: value for key, value in array_info.items() if value is not None
            }

        text_rule = text_rules_module.heuristic_text_rules(path, summary)
        if text_rule:
            rule["text_rules"] = text_rules_module.merge_text_rules(None, text_rule)

        rules["fields"][path] = rule

    return rules


def rules_from_profile(
    profile: dict[str, Any],
    *,
    llm: Optional[LLMClient] = None,
) -> dict[str, Any]:
    """Construct rules from a profile, optionally refining via LLM."""

    base_rules = synthesize_rules(profile)
    if llm is None:
        return base_rules

    text_rules_module.refine_text_rules(profile, base_rules, llm)

    refined = _refine_rules_with_llm(profile, base_rules, llm)
    return refined or base_rules


def _refine_rules_with_llm(
    profile: dict[str, Any],
    rules: dict[str, Any],
    llm: LLMClient,
) -> Optional[dict[str, Any]]:
    field_summaries = profile.get("field_summaries", {})
    if not isinstance(field_summaries, dict):
        return None

    trimmed = dict(list(field_summaries.items())[:40])
    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You refine synthetic data rules. Return JSON {\"fields\": {<jsonpath>: {optional keys}}} "
                    "with semantic_type, regex, enum, pii_likelihood, and optional text_rules (templates/conditions)."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "field_summaries": trimmed,
                        "current_rules": rules.get("fields", {}),
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
    }

    try:
        refinements = llm.generate_text(
            payload,
            schema=RULE_REFINEMENT_SCHEMA,
            parse_json=True,
        )
    except (LLMResponseError, Exception):
        return None

    if not isinstance(refinements, dict):
        return None

    fields = refinements.get("fields")
    if not isinstance(fields, dict):
        return None

    merged = {"fields": {**rules.get("fields", {})}}
    for path, details in fields.items():
        if not isinstance(details, dict):
            continue
        target = merged["fields"].setdefault(path, {})
        semantic = details.get("semantic_type")
        if semantic:
            target["semantic_type"] = semantic
        regex = details.get("regex")
        if regex:
            target["regex"] = regex
        enum_values = details.get("enum")
        if isinstance(enum_values, list) and enum_values:
            target["enum"] = enum_values
        pii = details.get("pii_likelihood")
        if isinstance(pii, (int, float)):
            target["pii_likelihood"] = float(pii)

        text_rules = details.get("text_rules")
        if isinstance(text_rules, dict):
            merged_rules = text_rules_module.merge_text_rules(target.get("text_rules"), text_rules)
            if merged_rules.get("templates"):
                target["text_rules"] = merged_rules

    return merged
