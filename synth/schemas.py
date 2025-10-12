"""JSON schema definitions for validating structured LLM responses."""

from __future__ import annotations

FIELD_SUMMARY_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "semantic_type": {"type": ["string", "null"]},
        "regex_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "support": {"type": ["number", "null"]},
                    "generality": {"type": ["number", "null"]},
                },
                "required": ["pattern"],
                "additionalProperties": True,
            },
        },
        "enum_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "value": {},
                    "count": {"type": ["number", "null"]},
                },
                "required": ["value"],
                "additionalProperties": True,
            },
        },
        "pii_likelihood": {"type": ["number", "null"], "minimum": 0.0, "maximum": 1.0},
        "anomalies": {"type": "array"},
        "confidence": {"type": ["number", "null"], "minimum": 0.0, "maximum": 1.0},
        "narratives": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "additionalProperties": True,
}

PROFILE_SUMMARY_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "field_summaries": {
            "type": "object",
            "additionalProperties": FIELD_SUMMARY_SCHEMA,
        },
        "keys": {"type": ["object", "null"]},
        "functional_dependencies": {"type": "array"},
        "temporal_patterns": {"type": "array"},
    },
    "required": ["field_summaries"],
}

REGEX_CANDIDATE_ARRAY_SCHEMA: dict[str, object] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "support": {"type": ["number", "null"]},
            "generality": {"type": ["number", "null"]},
        },
        "required": ["pattern"],
        "additionalProperties": True,
    },
}

TEXT_TEMPLATE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "kind": {
            "type": "string",
            "enum": ["literal", "faker", "regex", "pattern"],
        },
        "value": {"type": "string"},
        "weight": {"type": ["number", "null"]},
        "description": {"type": "string"},
    },
    "required": ["kind", "value"],
    "additionalProperties": True,
}

TEXT_CONDITION_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "when": {"type": "string"},
        "template": {
            "anyOf": [TEXT_TEMPLATE_SCHEMA, {"type": "string"}],
        },
        "weight": {"type": ["number", "null"]},
        "description": {"type": "string"},
    },
    "required": ["when", "template"],
    "additionalProperties": True,
}

TEXT_RULE_AUTHOR_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "templates": {
            "type": "array",
            "items": TEXT_TEMPLATE_SCHEMA,
            "minItems": 1,
        },
        "conditions": {
            "type": "array",
            "items": TEXT_CONDITION_SCHEMA,
        },
        "notes": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["templates"],
    "additionalProperties": True,
}

RULE_REFINEMENT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "fields": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "semantic_type": {"type": ["string", "null"]},
                    "regex": {"type": "string"},
                    "enum": {"type": "array"},
                    "pii_likelihood": {"type": ["number", "null"]},
                    "text_rules": TEXT_RULE_AUTHOR_SCHEMA,
                },
                "additionalProperties": True,
            },
        }
    },
    "required": ["fields"],
}
