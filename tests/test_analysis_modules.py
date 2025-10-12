import unittest

import pandas as pd

from synth import privacy as privacy_module
from synth.correlations import compute_correlations
from synth.temporal import analyze_temporal_patterns


class CorrelationModuleTests(unittest.TestCase):
    def test_compute_correlations_numeric(self) -> None:
        records = [
            {"a": 1, "b": 2, "category": "x"},
            {"a": 2, "b": 4, "category": "y"},
            {"a": 3, "b": 6, "category": "x"},
        ]
        result = compute_correlations(records)
        self.assertIn("pearson", result)
        self.assertAlmostEqual(result["pearson"]["a"]["b"], 1.0, places=4)

    def test_compute_correlations_categorical(self) -> None:
        records = [
            {"color": "red", "shape": "circle"},
            {"color": "red", "shape": "square"},
            {"color": "blue", "shape": "circle"},
            {"color": "blue", "shape": "circle"},
        ]
        result = compute_correlations(records)
        self.assertIn("cramers_v", result)
        self.assertIn("color", result["cramers_v"])
        self.assertIn("shape", result["cramers_v"]["color"])
        self.assertGreaterEqual(result["cramers_v"]["color"]["shape"], 0.0)


class TemporalModuleTests(unittest.TestCase):
    def test_analyze_temporal_patterns(self) -> None:
        timestamps = pd.date_range("2024-01-01", periods=5, freq="D")
        result = analyze_temporal_patterns(timestamps)
        self.assertIn(result["trend"], {"increasing", "stable"})
        self.assertIn(result["seasonality"], {"daily", "weekly", "none"})
        self.assertIn("weekday_distribution", result)
        self.assertIn("interarrival", result)


class PrivacyModuleTests(unittest.TestCase):
    def test_compute_k_anonymity(self) -> None:
        records = [
            {"name": "Alice", "zip": "12345"},
            {"name": "Bob", "zip": "12345"},
            {"name": "Alice", "zip": "67890"},
        ]
        profile = {
            "field_summaries": {
                "$.name": {"semantic_type": "name"},
                "$.zip": {"pii_likelihood": 0.8},
            }
        }
        qis = privacy_module.derive_quasi_identifiers(profile)
        self.assertIn("$.name", qis)
        result = privacy_module.compute_k_anonymity(records, qis, minimum_count=2)
        self.assertEqual(result["k"], 1)
        self.assertFalse(result.get("meets_threshold"))

    def test_enforce_privacy_generalizes_quasi_identifiers(self) -> None:
        profile = {
            "field_summaries": {
                "$.email": {
                    "semantic_type": "email",
                    "types": [{"name": "string", "confidence": 1.0}],
                },
                "$.age": {
                    "types": [{"name": "number", "confidence": 1.0}],
                    "numeric": {"min": 18, "max": 65, "std": 5},
                    "pii_likelihood": 0.7,
                },
            }
        }
        ruleset = {
            "fields": {
                "$.email": {"type": "string", "enum": ["a@example.com"]},
                "$.age": {"type": "number", "numeric": {"min": 18, "max": 65, "std": 5}},
            }
        }
        adjusted = privacy_module.enforce_privacy_constraints(ruleset, profile)
        email_rule = adjusted["fields"]["$.email"]
        self.assertNotIn("enum", email_rule)
        self.assertTrue(email_rule.get("privacy", {}).get("generalized"))
        age_rule = adjusted["fields"]["$.age"]
        self.assertIn("bucket", age_rule["numeric"])
        privacy_cfg = adjusted["privacy"]["k_anonymity"]
        self.assertIn("$.email", privacy_cfg["quasi_identifiers"])
        self.assertTrue(privacy_cfg["generalized_fields"])

    def test_enforce_privacy_applies_differential_privacy(self) -> None:
        ruleset = {
            "fields": {
                "$.score": {
                    "type": "number",
                    "numeric": {"min": 0, "max": 100, "std": 10},
                }
            },
            "privacy": {
                "differential_privacy": {
                    "enabled": True,
                    "epsilon": 0.5,
                }
            },
        }
        adjusted = privacy_module.enforce_privacy_constraints(ruleset, profile=None)
        score_rule = adjusted["fields"]["$.score"]
        dp_meta = score_rule.get("privacy", {}).get("dp")
        self.assertIsNotNone(dp_meta)
        self.assertGreater(score_rule["numeric"]["max"], 100)
        applied_fields = adjusted["privacy"]["differential_privacy"].get("applied_fields")
        self.assertIn("$.score", applied_fields)


if __name__ == "__main__":
    unittest.main()
