import io
import json
import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timezone

from synth import generate as generate_module
from synth.utils import RNGConfig


class GeneratorEnhancementsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = {
            "keys": {
                "pk_candidates": [
                    {"paths": ["$.id"], "confidence": 1.0},
                ]
            }
        }
        self.ruleset = {
            "fields": {
                "$.id": {
                    "type": "number",
                    "enum": [1, 2, 3, 4, 5],
                },
                "$.status": {
                    "type": "string",
                    "text_rules": {
                        "templates": [
                            {"kind": "literal", "value": "OK", "weight": 0.8},
                            {"kind": "literal", "value": "WARN", "weight": 0.2},
                        ],
                        "conditions": [
                            {
                                "when": "missing",
                                "template": {"kind": "literal", "value": "N/A"},
                            }
                        ],
                    },
                    "required_rate": 1.0,
                },
            }
        }
        self.ruleset.setdefault("fields", {}).setdefault("$.timestamp", {
            "type": "datetime",
            "datetime": {
                "min": "2025-09-28T16:17:17.543510-04:00",
                "max": "2025-10-02T08:00:00Z",
            },
        })

    def test_iter_synthetic_data(self) -> None:
        iterator = generate_module.iter_synthetic_data(
            profile=self.profile,
            ruleset=self.ruleset,
            count=3,
            rng=RNGConfig(123),
            cache_dir=None,
            enforce_privacy=False,
            default_array_cap=2,
        )
        records = list(iterator)
        self.assertEqual(len(records), 3)
        statuses = {record["status"] for record in records}
        self.assertTrue(statuses.issubset({"OK", "WARN", "N/A"}))

    def test_text_rule_generation(self) -> None:
        buffer = io.StringIO()
        generate_module.generate_synthetic_data(
            profile=self.profile,
            ruleset=self.ruleset,
            output_handle=buffer,
            count=2,
            rng=RNGConfig(7),
            cache_dir=None,
            enforce_privacy=False,
            default_array_cap=2,
        )
        buffer.seek(0)
        records = [json.loads(line) for line in buffer.read().strip().splitlines()]
        self.assertEqual(len(records), 2)
        for record in records:
            self.assertIn(record["status"], {"OK", "WARN", "N/A"})

    def test_datetime_generation_with_offset_bounds(self) -> None:
        buffer = io.StringIO()
        generate_module.generate_synthetic_data(
            profile=self.profile,
            ruleset=self.ruleset,
            output_handle=buffer,
            count=3,
            rng=RNGConfig(11),
            cache_dir=None,
            enforce_privacy=False,
            default_array_cap=2,
        )
        timestamps = [json.loads(line)["timestamp"] for line in buffer.getvalue().splitlines()]
        self.assertTrue(timestamps)
        for value in timestamps:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            lower = datetime.fromisoformat("2025-09-28T16:17:17.543510-04:00")
            upper = datetime.fromisoformat("2025-10-02T08:00:00+00:00")
            if parsed.tzinfo is None and lower.tzinfo is not None:
                parsed = parsed.replace(tzinfo=lower.tzinfo)
            if lower.tzinfo is None and parsed.tzinfo is not None:
                lower = lower.replace(tzinfo=parsed.tzinfo)
            if upper.tzinfo is None and parsed.tzinfo is not None:
                upper = upper.replace(tzinfo=parsed.tzinfo)
            self.assertGreaterEqual(parsed.timestamp(), lower.timestamp())
            self.assertLessEqual(parsed.timestamp(), upper.timestamp())

    def test_checkpoint_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "checkpoint.json"
            buffer = io.StringIO()
            generate_module.generate_synthetic_data(
                profile=self.profile,
                ruleset=self.ruleset,
                output_handle=buffer,
                count=5,
                rng=RNGConfig(21),
                cache_dir=None,
                enforce_privacy=False,
                default_array_cap=2,
                checkpoint_path=checkpoint,
                checkpoint_interval=2,
            )
            data = json.loads(checkpoint.read_text())
            self.assertEqual(data["generated"], 5)
            self.assertIn("state", data)

    def test_checkpoint_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "checkpoint.json"
            output_path = Path(tmpdir) / "synthetic.jsonl"

            class InterruptingWriter:
                def __init__(self, target: Path, limit: int) -> None:
                    self._handle = target.open("w", encoding="utf-8")
                    self._limit = limit
                    self._lines = 0

                def write(self, data: str) -> int:
                    if data == "\n":
                        self._lines += 1
                        if self._lines > self._limit:
                            raise RuntimeError("Simulated failure")
                    return self._handle.write(data)

                def flush(self) -> None:
                    self._handle.flush()

                def close(self) -> None:
                    self._handle.close()

            rng = RNGConfig(42)
            writer = InterruptingWriter(output_path, limit=3)
            with self.assertRaises(RuntimeError):
                generate_module.generate_synthetic_data(
                    profile=self.profile,
                    ruleset=self.ruleset,
                    output_handle=writer,
                    count=5,
                    rng=rng,
                    cache_dir=None,
                    enforce_privacy=False,
                    default_array_cap=2,
                    checkpoint_path=checkpoint,
                    checkpoint_interval=1,
                )
            writer.close()

            checkpoint_state = json.loads(checkpoint.read_text())
            self.assertGreaterEqual(checkpoint_state["generated"], 3)
            generated_before = checkpoint_state["generated"]
            existing_content = output_path.read_text()
            lines = existing_content.splitlines()
            trimmed = "\n".join(lines[:generated_before])
            if trimmed:
                trimmed += "\n"
            output_path.write_text(trimmed)

            with output_path.open("a", encoding="utf-8") as handle:
                generate_module.generate_synthetic_data(
                    profile=self.profile,
                    ruleset=self.ruleset,
                    output_handle=handle,
                    count=5,
                    rng=rng,
                    cache_dir=None,
                    enforce_privacy=False,
                    default_array_cap=2,
                    checkpoint_path=checkpoint,
                    checkpoint_interval=1,
                )

            records = [json.loads(line) for line in output_path.read_text().splitlines()]
            self.assertEqual(len(records), 5)
            self.assertEqual(len({record["id"] for record in records}), len(records))


if __name__ == "__main__":
    unittest.main()
