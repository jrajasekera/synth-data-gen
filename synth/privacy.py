"""Privacy tooling: PII detection, k-anonymity, differential privacy scaffolding."""

from __future__ import annotations

import copy
import re
from typing import Any, Iterable

try:
    from presidio_analyzer import Pattern, PatternRecognizer
except ImportError:  # pragma: no cover - fallback when Presidio is unavailable
    Pattern = None
    PatternRecognizer = None


_EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_REGEX = re.compile(r"\+?\d[\d\s().-]{6,}\d")
_NAME_REGEX = re.compile(r"^[A-Z][a-z]+(?:\s[A-Z][a-z]+)*$")


def detect_pii(profile: dict[str, Any] | None) -> dict[str, list[str]]:
    """Detect fields containing PII-like values using Presidio or regex heuristics."""

    if not profile:
        return {}

    field_summaries: dict[str, Any] = profile.get("field_summaries", {})
    findings: dict[str, set[str]] = {}

    for path, summary in field_summaries.items():
        samples = [candidate.get("value") for candidate in summary.get("enum_candidates", [])]
        samples = [value for value in samples if isinstance(value, str)]
        if not samples:
            continue

        categories = _classify_samples(samples)
        if categories:
            findings[path] = categories

    return {path: sorted(categories) for path, categories in findings.items()}


def enforce_privacy_constraints(
    ruleset: dict[str, Any],
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Adjust ruleset to honor simple privacy protections."""

    adjusted = copy.deepcopy(ruleset)
    fields = adjusted.setdefault("fields", {})
    pii_findings = detect_pii(profile)

    for path, categories in pii_findings.items():
        rule = fields.setdefault(path, {})
        rule.pop("enum", None)
        rule.pop("regex", None)

        if "EMAIL_ADDRESS" in categories:
            rule["semantic_type"] = "email"
        if "PHONE_NUMBER" in categories:
            rule["semantic_type"] = "phone"
        if "PERSON" in categories:
            rule["semantic_type"] = "name"

    return adjusted


def _classify_samples(samples: Iterable[str]) -> set[str]:
    categories: set[str] = set()

    recognizers = _presidio_recognizers()
    if recognizers:
        for value in samples:
            for recognizer in recognizers:
                results = recognizer.analyze(text=value, entities=None, nlp_artifacts=None)
                for result in results:
                    categories.add(result.entity_type)

    # Fallback heuristic detection.
    for value in samples:
        if _EMAIL_REGEX.search(value):
            categories.add("EMAIL_ADDRESS")
        if _PHONE_REGEX.search(value):
            categories.add("PHONE_NUMBER")
        if _NAME_REGEX.match(value):
            categories.add("PERSON")

    return categories


def _presidio_recognizers() -> list[Any]:  # type: ignore[valid-type]
    if PatternRecognizer is None or Pattern is None:  # pragma: no cover - handled in tests via heuristics
        return []

    email_pattern = Pattern("email", _EMAIL_REGEX.pattern, 0.8)
    phone_pattern = Pattern("phone", _PHONE_REGEX.pattern, 0.6)
    name_pattern = Pattern("name", _NAME_REGEX.pattern, 0.4)

    email_recognizer = PatternRecognizer(supported_entity="EMAIL_ADDRESS", patterns=[email_pattern])
    phone_recognizer = PatternRecognizer(supported_entity="PHONE_NUMBER", patterns=[phone_pattern])
    name_recognizer = PatternRecognizer(supported_entity="PERSON", patterns=[name_pattern])

    return [email_recognizer, phone_recognizer, name_recognizer]
