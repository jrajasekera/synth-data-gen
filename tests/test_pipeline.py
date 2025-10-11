import io
import json
import re
import tempfile
import unittest
from pathlib import Path

from synth import generate as generate_module
from synth import profile as profile_module
from synth import rules as rules_module
from synth import validate as validate_module
from synth import privacy as privacy_module
from synth import plugin_registry
from synth.io import ChunkingConfig, JSONStream
from synth.utils import RNGConfig


class PipelineIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.source_path = Path(self.tempdir.name) / "source.json"
        sample_records = [
            {
                "id": 1,
                "name": "Alice",
                "email": "alice@example.com",
                "tags": ["alpha", "beta"],
                "orders": [
                    {
                        "order_id": "A-1",
                        "items": [
                            {"sku": "SKU-1", "quantity": 1},
                            {"sku": "SKU-2", "quantity": 2},
                        ],
                    }
                ],
            },
            {
                "id": 2,
                "name": "Bob",
                "email": "bob@example.com",
                "tags": ["gamma"],
                "orders": [
                    {
                        "order_id": "B-1",
                        "items": [{"sku": "SKU-3", "quantity": 5}],
                    }
                ],
            },
        ]
        self.source_path.write_text(json.dumps(sample_records))

    def _profile_source(self) -> dict[str, object]:
        stream = JSONStream(self.source_path, ChunkingConfig(size=16, format="json_array"))
        return profile_module.profile_dataset(stream=stream, rng=RNGConfig(42), cache_dir=None)

    def test_profile_to_generate_roundtrip(self) -> None:
        profile = self._profile_source()
        ruleset = rules_module.synthesize_rules(profile)
        ruleset = privacy_module.enforce_privacy_constraints(ruleset, profile)

        buffer = io.StringIO()
        generate_module.generate_synthetic_data(
            profile=profile,
            ruleset=ruleset,
            output_handle=buffer,
            count=5,
            rng=RNGConfig(1234),
            cache_dir=None,
            default_array_cap=2,
        )

        lines = [json.loads(line) for line in buffer.getvalue().strip().splitlines()]
        self.assertEqual(len(lines), 5)
        for record in lines:
            self.assertIn("id", record)
            self.assertIn("name", record)
            self.assertIn("email", record)
            self.assertIsInstance(record["email"], str)
            self.assertIn("tags", record)
            self.assertTrue(record["tags"], "tags should not be empty")
            self.assertIsInstance(record["tags"], list)
            self.assertIn("orders", record)
            self.assertIsInstance(record["orders"], list)
            if record["orders"]:
                first_order = record["orders"][0]
                self.assertIn("order_id", first_order)
                self.assertIn("items", first_order)
                self.assertIsInstance(first_order["items"], list)

        synthetic_path = Path(self.tempdir.name) / "synthetic.jsonl"
        synthetic_path.write_text(buffer.getvalue())

        validation = validate_module.validate_dataset(
            synthetic_path=synthetic_path,
            source_path=self.source_path,
            ruleset=ruleset,
        )
        self.assertIn("summary", validation)
        self.assertFalse(validation["rule_violations"], msg="Expected no rule violations")
        array_delta = validation["field_comparisons"].get("$.orders", {}).get("array")
        self.assertIsNotNone(array_delta)

    def test_plugin_generator_override(self) -> None:
        profile = self._profile_source()
        ruleset = rules_module.synthesize_rules(profile)
        ruleset = privacy_module.enforce_privacy_constraints(ruleset, profile)

        plugin_name = "static_name"

        def _factory(config: dict[str, object]):
            value = config.get("value", "PluginValue")

            def _generator(_: dict[str, object]) -> str:
                return str(value)

            return _generator

        registry = plugin_registry.registry
        original = registry.generators.get(plugin_name)
        registry.register_generator(plugin_name, _factory)
        self.addCleanup(self._restore_generator, plugin_name, original)

        ruleset.setdefault("fields", {}).setdefault("$.name", {})["generator"] = {
            "name": plugin_name,
            "config": {"value": "Override"},
        }

        buffer = io.StringIO()
        generate_module.generate_synthetic_data(
            profile=profile,
            ruleset=ruleset,
            output_handle=buffer,
            count=3,
            rng=RNGConfig(99),
            cache_dir=None,
            registry=registry,
            default_array_cap=2,
        )

        lines = [json.loads(line) for line in buffer.getvalue().strip().splitlines()]
        self.assertTrue(lines)
        for record in lines:
            self.assertEqual(record["name"], "Override")

    def test_regex_generation_without_privacy(self) -> None:
        profile = self._profile_source()
        ruleset = rules_module.synthesize_rules(profile)
        regex = ruleset["fields"].get("$.email", {}).get("regex")
        self.assertIsNotNone(regex)
        pattern = re.compile(regex)

        buffer = io.StringIO()
        generate_module.generate_synthetic_data(
            profile=profile,
            ruleset=ruleset,
            output_handle=buffer,
            count=5,
            rng=RNGConfig(321),
            cache_dir=None,
            enforce_privacy=False,
            default_array_cap=2,
        )

        emails = [json.loads(line)["email"] for line in buffer.getvalue().strip().splitlines()]
        self.assertEqual(len(emails), 5)
        for email in emails:
            self.assertRegex(email, pattern)

    def _restore_generator(self, name: str, factory) -> None:
        registry = plugin_registry.registry
        if factory is None:
            registry.generators.pop(name, None)
        else:
            registry.generators[name] = factory


if __name__ == "__main__":
    unittest.main()
