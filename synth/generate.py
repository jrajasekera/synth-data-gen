"""Synthetic data generation logic."""

from __future__ import annotations

import copy
import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, IO, Optional

from faker import Faker
import yaml

try:
    import exrex
except ImportError:  # pragma: no cover - exrex should be installed via dependencies
    exrex = None


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


from .rules import synthesize_rules, rules_from_profile as rules_module_from_profile
from .llm import LLMClient
from .utils import RNGConfig
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
) -> None:
    """Write synthetic JSONL records to the provided handle."""

    if count <= 0:
        return

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

    for _ in range(count):
        record = _generate_node(root_node, rand, faker, active_registry, default_array_cap)
        output_handle.write(json.dumps(record))
        output_handle.write("\n")


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

    if field_type == "number":
        numeric = rule.get("numeric", {})
        lower = float(numeric.get("min", 0.0))
        upper = float(numeric.get("max", lower + 1.0))
        if math.isclose(lower, upper):
            return lower
        return rand.uniform(lower, upper)

    if field_type == "datetime":
        datetime_rule = rule.get("datetime", {})
        start = datetime_rule.get("min")
        end = datetime_rule.get("max")
        faker_kwargs: dict[str, Any] = {}
        if start:
            faker_kwargs["start_date"] = start
        if end:
            faker_kwargs["end_date"] = end
        return faker.date_time_between(**faker_kwargs).isoformat()

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
