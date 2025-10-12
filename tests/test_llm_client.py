import json
import unittest
from typing import Any, List
from unittest.mock import patch

import httpx

from synth.llm import LLMClient, LLMConfig, LLMResponseError
from synth.schemas import PROFILE_SUMMARY_SCHEMA, REGEX_CANDIDATE_ARRAY_SCHEMA


def _make_response(content: Any) -> httpx.Response:
    """Helper to build a chat-completions style response."""

    if isinstance(content, str):
        payload = content
    else:
        payload = json.dumps(content)
    return httpx.Response(
        status_code=200,
        json={
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": payload,
                    },
                }
            ]
        },
        request=httpx.Request("POST", "http://127.0.0.1:8080/v1/chat/completions"),
    )


class LLMClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = LLMClient(config=LLMConfig(global_seed=123, batch_size=2, max_concurrent_requests=2))
        self.addCleanup(self.client.close)

    def test_map_reduce_assigns_deterministic_seeds(self) -> None:
        prompts = [
            {
                "messages": [
                    {"role": "system", "content": "Profile the dataset."},
                    {"role": "user", "content": json.dumps({"slice": i})},
                ]
            }
            for i in range(3)
        ]

        captured_payloads: List[dict[str, Any]] = []

        def _fake_post(_: str, json: dict[str, Any]) -> httpx.Response:
            captured_payloads.append(json)
            return _make_response({"field_summaries": {}})

        with patch.object(self.client._client, "post", side_effect=_fake_post):
            first_run = self.client.map_reduce(
                prompts,
                schema=PROFILE_SUMMARY_SCHEMA,
                parse_json=True,
            )

        self.assertEqual(len(first_run), 3)
        seeds = [payload.get("seed") for payload in captured_payloads]
        self.assertTrue(all(isinstance(seed, int) for seed in seeds))

        captured_payloads.clear()
        with patch.object(self.client._client, "post", side_effect=_fake_post):
            second_run = self.client.map_reduce(
                prompts,
                schema=PROFILE_SUMMARY_SCHEMA,
                parse_json=True,
            )

        self.assertEqual(len(second_run), 3)
        seeds_again = [payload.get("seed") for payload in captured_payloads]
        self.assertEqual(seeds, seeds_again, "Expected seeds to be deterministic across runs")

    def test_map_reduce_filters_invalid_json(self) -> None:
        prompt = {
            "messages": [
                {"role": "system", "content": "Profile records."},
                {"role": "user", "content": json.dumps({"records": []})},
            ]
        }

        with patch.object(self.client._client, "post", return_value=_make_response("not-json")):
            results = self.client.map_reduce(
                [prompt],
                schema=PROFILE_SUMMARY_SCHEMA,
                parse_json=True,
            )

        self.assertEqual(results, [], "Invalid JSON responses should be discarded")

    def test_generate_text_schema_validation(self) -> None:
        prompt = {
            "messages": [
                {"role": "system", "content": "Produce regex candidates."},
                {"role": "user", "content": json.dumps({"examples": ["abc"]})},
            ]
        }

        with patch.object(
            self.client._client,
            "post",
            return_value=_make_response(
                [
                    {"pattern": "^abc$", "support": 10, "generality": 0.5},
                    {"pattern": "^def$", "support": 5, "generality": 0.4},
                ]
            ),
        ):
            values = self.client.generate_text(
                prompt,
                schema=REGEX_CANDIDATE_ARRAY_SCHEMA,
                parse_json=True,
            )

        self.assertIsInstance(values, list)
        self.assertEqual(len(values), 2)

        with patch.object(
            self.client._client,
            "post",
            return_value=_make_response({"not": "an array"}),
        ):
            with self.assertRaises(LLMResponseError):
                self.client.generate_text(
                    prompt,
                    schema=REGEX_CANDIDATE_ARRAY_SCHEMA,
                    parse_json=True,
                )


if __name__ == "__main__":
    unittest.main()
