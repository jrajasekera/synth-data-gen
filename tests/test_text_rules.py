import unittest

from synth import text_rules


class TextRuleTests(unittest.TestCase):
    def test_heuristic_email_templates(self) -> None:
        summary = {
            "types": [{"name": "string", "confidence": 1.0}],
            "semantic_type": "email",
            "required_rate": 1.0,
        }
        rules = text_rules.heuristic_text_rules("$.email", summary)
        self.assertTrue(rules)
        templates = rules.get("templates")
        self.assertTrue(templates)
        first = templates[0]
        self.assertEqual(first["kind"], "faker")
        self.assertEqual(first["value"], "email")

    def test_merge_text_rules_deduplicates(self) -> None:
        base = {
            "templates": [{"kind": "literal", "value": "A", "weight": 1.0}],
            "conditions": [{"when": "missing", "template": {"kind": "literal", "value": "N/A"}}],
        }
        incoming = {
            "templates": [
                {"kind": "literal", "value": "A"},
                {"kind": "literal", "value": "B"},
            ],
            "conditions": [{"when": "missing", "template": {"kind": "literal", "value": "N/A"}}],
            "notes": ["note"],
        }
        merged = text_rules.merge_text_rules(base, incoming)
        self.assertEqual(len(merged["templates"]), 2)
        self.assertEqual(len(merged.get("conditions", [])), 1)
        self.assertIn("note", merged.get("notes", []))


if __name__ == "__main__":
    unittest.main()
