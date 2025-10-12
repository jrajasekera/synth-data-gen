"""Privacy tooling: PII detection, k-anonymity, differential privacy scaffolding."""

from __future__ import annotations

import copy
import re
from collections import Counter
from typing import Any, Iterable, Optional

try:
    from presidio_analyzer import Pattern, PatternRecognizer
except ImportError:  # pragma: no cover - fallback when Presidio is unavailable
    Pattern = None
    PatternRecognizer = None


_EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_REGEX = re.compile(r"\+?\d[\d\s().-]{6,}\d")
_NAME_REGEX = re.compile(r"^[A-Z][a-z]+(?:\s[A-Z][a-z]+)*$")

DEFAULT_K = 5
DEFAULT_DP_EPSILON = 1.0
_MIN_BUCKET_WIDTH = 0.01


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

    privacy_settings = adjusted.setdefault("privacy", {})
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

    quasi_identifiers = _resolve_quasi_identifiers(privacy_settings, profile, pii_findings)
    target_k = _resolve_target_k(privacy_settings, profile)

    generalized_fields: list[str] = []
    bucketized_fields: list[str] = []
    for path in quasi_identifiers:
        rule = fields.get(path)
        if rule is None:
            continue
        changed, bucketized = _generalize_quasi_identifier(rule)
        if changed:
            generalized_fields.append(path)
        if bucketized:
            bucketized_fields.append(path)

    privacy_settings["k_anonymity"] = {
        "k": target_k,
        "quasi_identifiers": quasi_identifiers,
        "generalized_fields": sorted(set(generalized_fields)),
        "bucketized_fields": sorted(set(bucketized_fields)),
    }

    dp_config = _resolve_dp_config(privacy_settings, profile)
    dp_applied: list[str] = []
    if dp_config["enabled"]:
        epsilon = max(float(dp_config["epsilon"]), 1e-3)
        for path, rule in fields.items():
            if _apply_dp_to_numeric_rule(rule, epsilon):
                dp_applied.append(path)
        privacy_settings["differential_privacy"] = {
            "enabled": True,
            "epsilon": float(epsilon),
            "applied_fields": sorted(set(dp_applied)),
        }
    else:
        for rule in fields.values():
            meta = rule.get("privacy")
            if isinstance(meta, dict):
                meta.pop("dp", None)
        privacy_settings["differential_privacy"] = {"enabled": False, "epsilon": None}

    return adjusted


def _resolve_quasi_identifiers(
    privacy_settings: dict[str, Any],
    profile: Optional[dict[str, Any]],
    pii_findings: dict[str, list[str]],
) -> list[str]:
    configured = []
    k_config = privacy_settings.get("k_anonymity")
    if isinstance(k_config, dict):
        configured = [
            path for path in k_config.get("quasi_identifiers", []) if isinstance(path, str)
        ]

    derived = derive_quasi_identifiers(profile)
    pii_candidates = [path for path, categories in pii_findings.items() if categories]

    combined: list[str] = []
    for path in configured + derived + pii_candidates:
        if isinstance(path, str) and path not in combined:
            combined.append(path)
    return combined


def _resolve_target_k(privacy_settings: dict[str, Any], profile: Optional[dict[str, Any]]) -> int:
    profile_privacy = ((profile or {}).get("privacy") or {}) if profile else {}
    profile_k = (
        (profile_privacy.get("k_anonymity") or {}).get("k")
        if isinstance(profile_privacy.get("k_anonymity"), dict)
        else None
    )
    ruleset_k = None
    k_config = privacy_settings.get("k_anonymity")
    if isinstance(k_config, dict):
        ruleset_k = k_config.get("k")

    for candidate in (profile_k, ruleset_k):
        if isinstance(candidate, int) and candidate > 0:
            return candidate
    return DEFAULT_K


def _resolve_dp_config(
    privacy_settings: dict[str, Any],
    profile: Optional[dict[str, Any]],
) -> dict[str, Any]:
    profile_privacy = ((profile or {}).get("privacy") or {}) if profile else {}
    profile_dp = profile_privacy.get("differential_privacy")
    if not isinstance(profile_dp, dict):
        profile_dp = {}

    ruleset_dp = privacy_settings.get("differential_privacy")
    if not isinstance(ruleset_dp, dict):
        ruleset_dp = {}

    combined = {**profile_dp, **ruleset_dp}
    enabled = bool(combined.get("enabled"))
    epsilon = combined.get("epsilon", DEFAULT_DP_EPSILON)
    try:
        epsilon_value = float(epsilon)
    except (TypeError, ValueError):
        epsilon_value = DEFAULT_DP_EPSILON

    return {"enabled": enabled, "epsilon": epsilon_value}


def _generalize_quasi_identifier(rule: dict[str, Any]) -> tuple[bool, bool]:
    changed = False
    bucketized = False

    for key in ("enum", "regex", "text_rules"):
        if key in rule:
            rule.pop(key, None)
            changed = True

    meta = rule.setdefault("privacy", {})
    meta["generalized"] = True

    numeric = rule.get("numeric")
    if isinstance(numeric, dict):
        if _bucketize_numeric_rule(numeric):
            bucketized = True
            changed = True

    datetime_rule = rule.get("datetime")
    if isinstance(datetime_rule, dict):
        granularity = datetime_rule.get("granularity")
        if granularity not in {"day", "hour", "minute"}:
            datetime_rule["granularity"] = "day"
            changed = True

    return changed, bucketized


def _bucketize_numeric_rule(numeric: dict[str, Any]) -> bool:
    lower = numeric.get("min")
    upper = numeric.get("max")
    std = numeric.get("std")

    spread: float
    if lower is not None and upper is not None:
        try:
            spread = float(upper) - float(lower)
        except (TypeError, ValueError):
            spread = 0.0
    else:
        spread = 0.0

    if spread <= 0.0 and isinstance(std, (int, float)):
        spread = abs(float(std)) * 6.0
    if spread <= 0.0:
        spread = 1.0

    bucket = max(spread / 20.0, _MIN_BUCKET_WIDTH)
    if spread < 1.0:
        bucket = max(spread / 10.0, _MIN_BUCKET_WIDTH)

    bucket = float(round(bucket, 6))
    if numeric.get("bucket") == bucket:
        return False
    numeric["bucket"] = bucket
    return True


def _apply_dp_to_numeric_rule(rule: dict[str, Any], epsilon: float) -> bool:
    field_type = rule.get("type")
    if field_type not in {"number", "integer"}:
        return False
    numeric = rule.get("numeric")
    if not isinstance(numeric, dict):
        return False

    lower = numeric.get("min")
    upper = numeric.get("max")
    std = numeric.get("std")

    spread = None
    if lower is not None and upper is not None:
        try:
            spread = float(upper) - float(lower)
        except (TypeError, ValueError):
            spread = None

    if isinstance(std, (int, float)):
        baseline = abs(float(std))
    elif spread is not None and spread > 0:
        baseline = spread / 6.0
    else:
        baseline = 1.0

    scale = max(baseline / max(epsilon, 1e-6), _MIN_BUCKET_WIDTH)

    if isinstance(lower, (int, float)):
        numeric["min"] = float(lower) - 3 * scale
    else:
        numeric["min"] = float(-3 * scale)
    if isinstance(upper, (int, float)):
        numeric["max"] = float(upper) + 3 * scale
    else:
        numeric["max"] = float(3 * scale)

    if numeric["max"] < numeric["min"]:
        numeric["max"] = numeric["min"]

    meta = rule.setdefault("privacy", {})
    dp_meta = meta.setdefault("dp", {})
    dp_meta["epsilon"] = float(epsilon)
    dp_meta["scale"] = float(scale)
    return True


def derive_quasi_identifiers(profile: dict[str, Any] | None) -> list[str]:
    """Infer quasi-identifier fields from the profile metadata."""

    if not profile:
        return []

    field_summaries = profile.get("field_summaries", {}) or {}
    quasi_identifiers: list[str] = []
    for path, summary in field_summaries.items():
        if not path.startswith("$.") or "[]" in path:
            continue
        semantic = summary.get("semantic_type")
        pii = summary.get("pii_likelihood", 0.0)
        if semantic in {"email", "phone", "name"} or (isinstance(pii, (int, float)) and pii >= 0.5):
            quasi_identifiers.append(path)
    return quasi_identifiers


def compute_k_anonymity(
    records: Iterable[dict[str, Any]],
    quasi_identifiers: list[str],
    *,
    minimum_count: Optional[int] = None,
) -> dict[str, Any]:
    """Compute the minimum group size for the provided quasi identifiers."""

    if not quasi_identifiers:
        return {"k": None, "groups": {}, "quasi_identifiers": []}

    counter: Counter[str] = Counter()
    total_records = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        key_values: list[str] = []
        skip_record = False
        for path in quasi_identifiers:
            value = _resolve_simple_path(record, path)
            if value is None:
                skip_record = True
                break
            key_values.append(str(value))
        if skip_record:
            continue
        counter["|".join(key_values)] += 1
        total_records += 1

    if not counter:
        return {"k": None, "groups": {}, "quasi_identifiers": quasi_identifiers}

    min_group = min(counter.values())
    status = {
        "k": min_group,
        "groups": dict(counter.most_common(10)),
        "quasi_identifiers": quasi_identifiers,
        "records_considered": total_records,
    }
    if minimum_count is not None:
        status["meets_threshold"] = min_group >= minimum_count
    return status


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


def _resolve_simple_path(record: dict[str, Any], path: str) -> Any:
    if not path.startswith("$.") or "[]" in path or "*" in path:
        return None
    current: Any = record
    for segment in path[2:].split('.'):
        if not isinstance(current, dict):
            return None
        current = current.get(segment)
        if current is None:
            return None
    return current
