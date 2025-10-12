"""Synthetic data generation logic."""

from __future__ import annotations

import base64
import copy
import json
import logging
import math
import pickle
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, IO, Iterable, Optional

from datetime import datetime

from faker import Faker
import yaml

try:
    import exrex
except ImportError:  # pragma: no cover - exrex should be installed via dependencies
    exrex = None


logger = logging.getLogger(__name__)


@dataclass
class ObjectNode:
    scalars: dict[str, dict[str, Any]] = field(default_factory=dict)
    objects: dict[str, "ObjectNode"] = field(default_factory=dict)
    arrays: dict[str, "ArraySpec"] = field(default_factory=dict)
    object_rules: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class ArraySpec:
    rule: Optional[dict[str, Any]] = None
    element_rule: Optional[dict[str, Any]] = None
    element_node: Optional[ObjectNode] = None


_CHECKPOINT_VERSION = 2


def _encode_random_state(state: Any) -> str:
    return base64.b64encode(pickle.dumps(state)).decode("ascii")


def _decode_random_state(payload: str) -> Any:
    data = base64.b64decode(payload.encode("ascii"))
    return pickle.loads(data)


def _load_checkpoint_payload(path: Path) -> Optional[dict[str, Any]]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _collect_state_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "warning_emitted": bool(state.get("warning_emitted", False)),
        "unique_sets": {path: sorted(values) for path, values in state.get("unique_sets", {}).items()},
        "composite_sets": [
            {"paths": list(paths), "values": sorted(values)}
            for paths, values in state.get("composite_sets", {}).items()
        ],
    }
    rand = state.get("rand")
    if isinstance(rand, random.Random):
        snapshot["rand_state"] = _encode_random_state(rand.getstate())
    faker = state.get("faker")
    faker_random = getattr(faker, "random", None)
    if isinstance(faker_random, random.Random):
        snapshot["faker_state"] = _encode_random_state(faker_random.getstate())
    return snapshot


def _apply_state_snapshot(state: dict[str, Any], snapshot: dict[str, Any]) -> None:
    rand_state = snapshot.get("rand_state")
    if rand_state and isinstance(state.get("rand"), random.Random):
        state["rand"].setstate(_decode_random_state(rand_state))

    faker_state = snapshot.get("faker_state")
    faker_random = getattr(state.get("faker"), "random", None)
    if faker_state and isinstance(faker_random, random.Random):
        faker_random.setstate(_decode_random_state(faker_state))

    unique_sets = state.get("unique_sets", {})
    for path, values in snapshot.get("unique_sets", {}).items():
        if path not in unique_sets:
            unique_sets[path] = set()
        unique_sets[path] = set(map(str, values))

    composite_sets = state.get("composite_sets", {})
    for entry in snapshot.get("composite_sets", []):
        if not isinstance(entry, dict):
            continue
        paths = entry.get("paths")
        values = entry.get("values")
        if not isinstance(paths, list) or not isinstance(values, list):
            continue
        key = tuple(sorted(str(path) for path in paths))
        if key not in composite_sets:
            composite_sets[key] = set()
        composite_sets[key] = set(map(str, values))

    if snapshot.get("warning_emitted"):
        state["warning_emitted"] = True


def _write_checkpoint_payload(path: Path, *, generated: int, state: dict[str, Any]) -> None:
    payload = {
        "version": _CHECKPOINT_VERSION,
        "generated": int(generated),
        "state": _collect_state_snapshot(state),
    }
    path.write_text(json.dumps(payload))


from .rules import synthesize_rules, rules_from_profile as rules_module_from_profile
from .llm import LLMClient
from .utils import RNGConfig, stable_hash
from . import privacy as privacy_module
from .plugin_registry import PluginRegistry, registry as default_registry


def load_ruleset(path: Path) -> dict[str, Any]:
    """Load a ruleset specification from YAML."""

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Ruleset must be a mapping")
    return data


def rules_from_profile(profile: dict[str, Any], llm: Optional[LLMClient] = None) -> dict[str, Any]:
    """Derive a ruleset structure from a profile artifact."""

    return rules_module_from_profile(profile, llm=llm)


def iter_synthetic_data(
    *,
    profile: dict[str, Any],
    ruleset: dict[str, Any],
    count: int,
    rng: RNGConfig,
    cache_dir: Optional[Path],
    enforce_privacy: bool = True,
    registry: Optional[PluginRegistry] = None,
    default_array_cap: int = 5,
) -> Iterable[dict[str, Any]]:
    """Iterate over synthetic records instead of writing them to disk."""

    if count <= 0:
        return iter(())

    state = _setup_generation(
        profile=profile,
        ruleset=ruleset,
        rng=rng,
        enforce_privacy=enforce_privacy,
        registry=registry,
        default_array_cap=default_array_cap,
    )

    def _record_iterator() -> Iterable[dict[str, Any]]:
        for _ in range(count):
            record, pending = _generate_unique_record(state)
            _commit_unique_fingerprints(pending, state["unique_sets"], state["composite_sets"])
            yield record

    return _record_iterator()


def _setup_generation(
    *,
    profile: dict[str, Any],
    ruleset: dict[str, Any],
    rng: RNGConfig,
    enforce_privacy: bool,
    registry: Optional[PluginRegistry],
    default_array_cap: int,
) -> dict[str, Any]:
    effective_ruleset = copy.deepcopy(ruleset)
    if enforce_privacy:
        effective_ruleset = privacy_module.enforce_privacy_constraints(effective_ruleset, profile)

    field_rules: dict[str, Any] = effective_ruleset.get("fields", {})
    rand = rng.random()
    faker = Faker()
    faker.seed_instance(rng.seed or 0)

    active_registry = registry or default_registry

    root_node = ObjectNode()
    for path, rule in field_rules.items():
        segments = _parse_segments(path)
        if not segments:
            continue
        _insert_rule(root_node, segments, rule)

    pk_candidates = (profile.get("keys") or {}).get("pk_candidates", [])
    simple_unique_paths = {
        candidate["paths"][0]
        for candidate in pk_candidates
        if isinstance(candidate, dict)
        and isinstance(candidate.get("paths"), list)
        and len(candidate["paths"]) == 1
        and _is_simple_path(candidate["paths"][0])
        and candidate.get("confidence", 0.0) >= 0.6
    }
    composite_key_paths = [
        tuple(candidate["paths"])
        for candidate in pk_candidates
        if isinstance(candidate, dict)
        and isinstance(candidate.get("paths"), list)
        and len(candidate["paths"]) > 1
        and all(_is_simple_path(path) for path in candidate["paths"])
        and candidate.get("confidence", 0.0) >= 0.6
    ]

    unique_sets: dict[str, set[str]] = {path: set() for path in simple_unique_paths}
    composite_sets: dict[tuple[str, ...], set[str]] = {tuple(sorted(paths)): set() for paths in composite_key_paths}

    return {
        "root_node": root_node,
        "rand": rand,
        "faker": faker,
        "registry": active_registry,
        "default_array_cap": default_array_cap,
        "unique_sets": unique_sets,
        "composite_sets": composite_sets,
        "max_attempts": 25,
        "warning_emitted": False,
    }


def _generate_unique_record(
    state: dict[str, Any],
) -> tuple[
    dict[str, Any],
    tuple[list[tuple[str, str]], list[tuple[tuple[str, ...], str]]],
]:
    for _ in range(state["max_attempts"]):
        candidate = _generate_node(
            state["root_node"],
            state["rand"],
            state["faker"],
            state["registry"],
            state["default_array_cap"],
        )
        pending = _prepare_unique_fingerprints(candidate, state["unique_sets"], state["composite_sets"])
        if pending is not None:
            return candidate, pending

    candidate = _generate_node(
        state["root_node"],
        state["rand"],
        state["faker"],
        state["registry"],
        state["default_array_cap"],
    )
    pending = _prepare_unique_fingerprints(
        candidate,
        state["unique_sets"],
        state["composite_sets"],
        force=True,
    )
    if not state["warning_emitted"]:
        logger.warning(
            "Unable to satisfy uniqueness constraints after %s attempts; emitting best-effort record.",
            state["max_attempts"],
        )
        state["warning_emitted"] = True
    if pending is None:
        pending = ([], [])
    return candidate, pending


def generate_synthetic_data(
    *,
    profile: dict[str, Any],
    ruleset: dict[str, Any],
    output_handle: IO[str],
    count: int,
    rng: RNGConfig,
    cache_dir: Optional[Path],
    enforce_privacy: bool = True,
    registry: Optional[PluginRegistry] = None,
    default_array_cap: int = 5,
    checkpoint_path: Optional[Path] = None,
    checkpoint_interval: int = 1000,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> None:
    """Write synthetic JSONL records to the provided handle."""

    if count <= 0:
        return

    checkpoint_file: Optional[Path] = None
    checkpoint_payload: Optional[dict[str, Any]] = None
    if checkpoint_path is not None:
        checkpoint_file = checkpoint_path.expanduser().resolve()
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_payload = _load_checkpoint_payload(checkpoint_file)

    state = _setup_generation(
        profile=profile,
        ruleset=ruleset,
        rng=rng,
        enforce_privacy=enforce_privacy,
        registry=registry,
        default_array_cap=default_array_cap,
    )

    if checkpoint_payload and isinstance(checkpoint_payload.get("state"), dict):
        _apply_state_snapshot(state, checkpoint_payload["state"])

    if checkpoint_payload:
        try:
            generated = int(checkpoint_payload.get("generated", 0))
        except (TypeError, ValueError):
            generated = 0
    else:
        generated = 0
    if progress_callback:
        progress_callback(min(generated, count))

    if generated >= count:
        if checkpoint_file is not None:
            _write_checkpoint_payload(checkpoint_file, generated=generated, state=state)
        return

    effective_interval = checkpoint_interval if checkpoint_interval > 0 else None

    try:
        for _ in range(generated, count):
            record, pending_keys = _generate_unique_record(state)
            output_handle.write(json.dumps(record))
            output_handle.write("\n")
            _commit_unique_fingerprints(pending_keys, state["unique_sets"], state["composite_sets"])
            generated += 1
            if checkpoint_file and effective_interval and generated % effective_interval == 0:
                _write_checkpoint_payload(checkpoint_file, generated=generated, state=state)
            if progress_callback:
                progress_callback(min(generated, count))
    finally:
        if checkpoint_file is not None:
            _write_checkpoint_payload(checkpoint_file, generated=generated, state=state)
        if progress_callback:
            progress_callback(min(generated, count))


def _parse_segments(path: str) -> list[str]:
    if not path.startswith("$."):
        return []
    raw_segments = path[2:].split(".")
    segments: list[str] = []
    for segment in raw_segments:
        segment = segment.strip()
        if not segment:
            continue
        segments.append(segment)
    return segments


def _insert_rule(node: ObjectNode, segments: list[str], rule: dict[str, Any]) -> None:
    segment = segments[0]
    is_array = segment.endswith("[]")
    name = segment[:-2] if is_array else segment

    if len(segments) == 1:
        if is_array:
            array_spec = node.arrays.setdefault(name, ArraySpec())
            array_spec.element_rule = rule
        elif rule.get("type") == "array":
            array_spec = node.arrays.setdefault(name, ArraySpec())
            array_spec.rule = rule
        elif rule.get("type") == "object":
            child = node.objects.setdefault(name, ObjectNode())
            node.object_rules[name] = rule
        else:
            node.scalars[name] = rule
        return

    if is_array:
        array_spec = node.arrays.setdefault(name, ArraySpec())
        if array_spec.element_node is None:
            array_spec.element_node = ObjectNode()
        _insert_rule(array_spec.element_node, segments[1:], rule)
    else:
        child = node.objects.setdefault(name, ObjectNode())
        _insert_rule(child, segments[1:], rule)


def _generate_node(
    node: ObjectNode,
    rand: random.Random,
    faker: Faker,
    registry: PluginRegistry,
    default_array_cap: int,
) -> dict[str, Any]:
    data: dict[str, Any] = {}

    for name, rule in node.scalars.items():
        data[name] = _generate_value(rule, rand, faker, registry)

    for name, child in node.objects.items():
        data[name] = _generate_node(child, rand, faker, registry, default_array_cap)

    for name, array_spec in node.arrays.items():
        data[name] = _generate_array_value(array_spec, rand, faker, registry, default_array_cap)

    return data


def _generate_array_value(
    spec: ArraySpec,
    rand: random.Random,
    faker: Faker,
    registry: PluginRegistry,
    default_array_cap: int,
) -> list[Any]:
    length = _resolve_array_length(spec.rule, rand, default_array_cap)
    elements: list[Any] = []

    for _ in range(length):
        if spec.element_node is not None:
            elements.append(_generate_node(spec.element_node, rand, faker, registry, default_array_cap))
        elif spec.element_rule is not None:
            elements.append(_generate_value(spec.element_rule, rand, faker, registry))
        else:
            elements.append({})

    return elements


def _resolve_array_length(container_rule: Optional[dict[str, Any]], rand: random.Random, default_cap: int) -> int:
    if container_rule:
        array_cfg = container_rule.get("array")
        if array_cfg:
            min_items = max(int(array_cfg.get("min_items", 1)), 0)
            max_items = int(array_cfg.get("max_items", max(min_items, 1)))
            if max_items < min_items:
                max_items = min_items
            if min_items == max_items:
                return min_items
            return rand.randint(min_items, max_items)
    return rand.randint(1, max(default_cap, 1))


def _parse_datetime_bound(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            return None
    return None


def _generate_value(
    rule: dict[str, Any],
    rand: random.Random,
    faker: Faker,
    registry: PluginRegistry,
) -> Any:
    generator_spec = rule.get("generator")
    if generator_spec:
        plugin_value = _maybe_generate_via_plugin(generator_spec, registry, rule, rand, faker)
        if plugin_value is not _PluginSentinel:
            return plugin_value

    text_rules = rule.get("text_rules")
    if text_rules:
        text_value = _apply_text_rules(text_rules, rand, faker, rule)
        if text_value is not None:
            return text_value

    field_type = rule.get("type", "string")
    enums = rule.get("enum")
    if enums:
        return rand.choice(enums)

    semantic = rule.get("semantic_type") or rule.get("format")
    if isinstance(semantic, str):
        semantic = semantic.lower()

    regex_pattern = rule.get("regex")

    length_spec = rule.get("length", {})
    min_length = max(int(length_spec.get("min", 1)), 1)
    max_length = int(length_spec.get("max", max(min_length, 12)))

    if regex_pattern and field_type == "string":
        generated = _generate_string_with_regex(regex_pattern, rand, faker)
        if generated is not None:
            return generated

    if field_type == "boolean":
        return bool(rand.getrandbits(1))

    if field_type in {"number", "integer"}:
        numeric = rule.get("numeric", {})
        lower = numeric.get("min")
        upper = numeric.get("max")
        lower_bound = float(lower) if lower is not None else 0.0
        upper_bound = float(upper) if upper is not None else lower_bound + 1.0
        if math.isclose(lower_bound, upper_bound):
            value = lower_bound
        else:
            value = rand.uniform(lower_bound, upper_bound)

        dp_cfg = (rule.get("privacy") or {}).get("dp") or {}
        scale = float(dp_cfg.get("scale", 0.0)) if dp_cfg else 0.0
        if scale > 0:
            value += _laplace_noise(rand, scale)

        bucket_size = numeric.get("bucket")
        if bucket_size:
            size = float(bucket_size)
            if size > 0:
                value = round(value / size) * size

        if lower is not None:
            value = max(lower_bound, value)
        if upper is not None:
            value = min(upper_bound, value)

        if field_type == "integer":
            return int(round(value))
        return float(value)

    if field_type == "datetime":
        datetime_rule = rule.get("datetime", {})
        start = _parse_datetime_bound(datetime_rule.get("min"))
        end = _parse_datetime_bound(datetime_rule.get("max"))
        faker_kwargs: dict[str, Any] = {}
        if start:
            faker_kwargs["start_date"] = start
        if end:
            faker_kwargs["end_date"] = end
        granularity = datetime_rule.get("granularity", "second")
        dt_value = faker.date_time_between(**faker_kwargs)
        if granularity == "day":
            dt_value = dt_value.replace(hour=0, minute=0, second=0, microsecond=0)
        elif granularity == "hour":
            dt_value = dt_value.replace(minute=0, second=0, microsecond=0)
        elif granularity == "minute":
            dt_value = dt_value.replace(second=0, microsecond=0)
        return dt_value.isoformat()

    if semantic == "email":
        return faker.email()
    if semantic == "name":
        return faker.name()
    if semantic == "phone":
        return faker.phone_number()

    return faker.pystr(min_chars=min_length, max_chars=max_length)


def _generate_string_with_regex(pattern: str, rand: random.Random, faker: Faker) -> Optional[str]:
    if exrex is not None:
        try:
            return exrex.getone(pattern, randrange=rand.randrange)
        except Exception:  # pragma: no cover - fall back to heuristic attempts
            pass

    compiled = re.compile(pattern)
    for _ in range(50):
        candidate = faker.pystr(min_chars=1, max_chars=32)
        if compiled.fullmatch(candidate):
            return candidate
    return None


def _laplace_noise(rand: random.Random, scale: float) -> float:
    if scale <= 0:
        return 0.0
    u = rand.random() - 0.5
    if u == 0:
        return 0.0
    return -scale * math.copysign(math.log(1 - 2 * abs(u)), u)


_PluginSentinel = object()


def _maybe_generate_via_plugin(
    generator_spec: Any,
    registry: PluginRegistry,
    rule: dict[str, Any],
    rand: random.Random,
    faker: Faker,
) -> Any:
    if registry is None:
        return _PluginSentinel

    if isinstance(generator_spec, str):
        name = generator_spec
        config: dict[str, Any] = {}
    elif isinstance(generator_spec, dict):
        name = generator_spec.get("name")
        config = generator_spec.get("config", {})
    else:
        return _PluginSentinel

    if not name:
        return _PluginSentinel

    factory = registry.generators.get(name)
    if factory is None:
        return _PluginSentinel

    factory_payload: dict[str, Any] = {"rule": rule}
    if isinstance(config, dict):
        factory_payload.update(config)

    generator_fn = factory(factory_payload)
    return generator_fn({
        "random": rand,
        "faker": faker,
        "rule": rule,
    })


def _apply_text_rules(
    text_rules: dict[str, Any],
    rand: random.Random,
    faker: Faker,
    rule: dict[str, Any],
) -> Optional[str]:
    templates = text_rules.get("templates") or []
    if not templates:
        return None

    conditions = text_rules.get("conditions") or []
    required_rate = float(rule.get("required_rate", 1.0))
    for condition in conditions:
        if not isinstance(condition, dict):
            continue
        when = condition.get("when")
        template = condition.get("template")
        if when == "missing":
            probability = max(0.0, min(1.0, 1.0 - required_rate))
            if probability > 0 and rand.random() < probability:
                rendered = _render_template(template, rand, faker, rule)
                if rendered is not None:
                    return rendered

    selected_template = _choose_template(templates, rand)
    return _render_template(selected_template, rand, faker, rule)


def _choose_template(templates: list[Any], rand: random.Random) -> Any:
    weights: list[float] = []
    for template in templates:
        weight = 1.0
        if isinstance(template, dict) and isinstance(template.get("weight"), (int, float)):
            weight = max(float(template["weight"]), 0.0)
        weights.append(weight)

    total = sum(weights)
    if total <= 0:
        return templates[-1]

    pick = rand.random() * total
    for template, weight in zip(templates, weights):
        pick -= weight
        if pick <= 0:
            return template
    return templates[-1]


def _render_template(template: Any, rand: random.Random, faker: Faker, rule: dict[str, Any]) -> Optional[str]:
    if isinstance(template, str):
        return template
    if not isinstance(template, dict):
        return None

    kind = template.get("kind", "literal")
    value = template.get("value")

    if kind == "literal":
        return str(value) if value is not None else ""
    if kind == "faker" and isinstance(value, str):
        provider = getattr(faker, value, None)
        if callable(provider):
            return str(provider())
        return faker.pystr()
    if kind == "regex" and isinstance(value, str):
        generated = _generate_string_with_regex(value, rand, faker)
        return generated if generated is not None else faker.pystr()
    if kind == "pattern" and isinstance(value, str):
        return _render_pattern(value, rand, faker)

    return None


def _render_pattern(pattern: str, rand: random.Random, faker: Faker) -> str:
    def faker_replacer(match: re.Match[str]) -> str:
        method = match.group(1)
        provider = getattr(faker, method, None)
        if callable(provider):
            try:
                return str(provider())
            except Exception:
                return ""
        return ""

    result = re.sub(r"\{\{faker\.([a-zA-Z_]+)\}\}", faker_replacer, pattern)
    result = result.replace("{{digit}}", str(rand.randint(0, 9)))
    result = result.replace("{{letter}}", chr(rand.randint(65, 90)))
    result = result.replace("{{letter_lower}}", chr(rand.randint(97, 122)))
    result = result.replace("{{alnum}}", rand.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"))
    return result


def _prepare_unique_fingerprints(
    record: dict[str, Any],
    unique_sets: dict[str, set[str]],
    composite_sets: dict[tuple[str, ...], set[str]],
    *,
    force: bool = False,
) -> Optional[tuple[list[tuple[str, str]], list[tuple[tuple[str, ...], str]]]]:
    pending_uniques: list[tuple[str, str]] = []
    for path, seen in unique_sets.items():
        value = _resolve_simple_path(record, path)
        if value is None:
            continue
        fingerprint = _fingerprint_value(value)
        if not force and fingerprint in seen:
            return None
        pending_uniques.append((path, fingerprint))

    pending_composites: list[tuple[tuple[str, ...], str]] = []
    for paths, seen in composite_sets.items():
        fingerprints: list[str] = []
        for path in paths:
            value = _resolve_simple_path(record, path)
            if value is None:
                return None
            fingerprints.append(_fingerprint_value(value))
        composite_fingerprint = stable_hash(fingerprints)
        if not force and composite_fingerprint in seen:
            return None
        pending_composites.append((paths, composite_fingerprint))

    return pending_uniques, pending_composites


def _commit_unique_fingerprints(
    pending: tuple[list[tuple[str, str]], list[tuple[tuple[str, ...], str]]],
    unique_sets: dict[str, set[str]],
    composite_sets: dict[tuple[str, ...], set[str]],
) -> None:
    pending_uniques, pending_composites = pending
    for path, fingerprint in pending_uniques:
        unique_sets.setdefault(path, set()).add(fingerprint)
    for paths, fingerprint in pending_composites:
        composite_sets.setdefault(paths, set()).add(fingerprint)


def _resolve_simple_path(record: dict[str, Any], path: str) -> Any:
    if not path.startswith("$."):
        return None
    current: Any = record
    for segment in path[2:].split('.'):
        if not isinstance(current, dict):
            return None
        current = current.get(segment)
        if current is None:
            return None
    return current


def _fingerprint_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _is_simple_path(path: str) -> bool:
    return path.startswith("$.") and "[]" not in path and "*" not in path
