"""Classify a completed request into one top-level execution mode."""

import json
import os
from dataclasses import fields, is_dataclass
from enum import Enum

from routing.request_understanding import CompletedRequest
from schemas.hypothesis_schema import ExecutionMode


DEFAULT_CLASSIFIER_MODEL = "gpt-5.6-luna"


def _to_json_compatible(value: object) -> object:
    """Recursively convert supported structured values to JSON-compatible data."""
    if isinstance(value, Enum):
        return _to_json_compatible(value.value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _to_json_compatible(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, (tuple, list)):
        return [_to_json_compatible(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _to_json_compatible(item)
            for key, item in value.items()
        }
    raise TypeError(
        f"Unsupported value type for JSON serialization: {type(value).__name__}."
    )


def _classifier_instructions() -> str:
    return """The request has already been semantically completed and canonicalized.
Do not reinterpret or expand it. Only classify execution mode. Return only the
required JSON.

Map the completed execution perspective to exactly one of:
- research: analyze realized or existing historical evidence, descriptive
  statistics, conditional results, or historical quantitative relationships.
- historical_replay: reconstruct or evaluate a prediction or decision at a
  historical point in time using only information available then.
- live_forecast: predict an unrealized future period from the current execution
  perspective.
- null: classification failed or the completed structured contract is
  unexpectedly ambiguous.

Use execution_perspective as the primary signal. normalized_request, time_scope,
delivery_scope, and requested_quantitative_views are supporting structured
context only. VALUE, PROBABILITY, DIFFERENCE, and CHANGE do not determine mode.
A historical predictive-value request is research when execution_perspective
calls for analysis of historical evidence; the word "predictive" does not imply
a live forecast. Do not infer variables, thresholds, targets, references, or
additional quantitative interpretations. Do not ask questions and do not solve
the request.
"""


def _classifier_input(request: CompletedRequest) -> str:
    """Serialize only mode-relevant completed semantics as compact JSON."""
    payload = {
        "normalized_request": request.normalized_request,
        "execution_perspective": request.execution_perspective,
        "time_scope": _to_json_compatible(request.time_scope),
        "delivery_scope": _to_json_compatible(request.delivery_scope),
        "decision_points": _to_json_compatible(
            getattr(request, "decision_points", ())
        ),
        "requested_quantitative_views": _to_json_compatible([
            {
                "kind": view.kind.value,
                "target": view.target,
                "reference": view.reference,
                "aggregation": view.aggregation,
            }
            for view in request.requested_quantitative_views
        ]),
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def _classify_with_llm(request: CompletedRequest) -> ExecutionMode | None:
    """Classify one completed semantic request, failing closed on API errors."""
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI

        client = OpenAI()
        response = client.responses.create(
            model=os.getenv("OPENAI_COMPILER_MODEL", DEFAULT_CLASSIFIER_MODEL),
            input=[
                {"role": "system", "content": _classifier_instructions()},
                {"role": "user", "content": _classifier_input(request)},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "execution_mode_classification",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "execution_mode": {
                                "enum": [
                                    "research",
                                    "historical_replay",
                                    "live_forecast",
                                    None,
                                ]
                            }
                        },
                        "required": ["execution_mode"],
                        "additionalProperties": False,
                    },
                }
            },
        )
        if not response.output_text:
            return None
        payload = json.loads(response.output_text)
        value = payload.get("execution_mode")
        return ExecutionMode(value) if value is not None else None
    except (ImportError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    except Exception:
        return None


def classify_execution_mode(
    completed_request: CompletedRequest,
) -> ExecutionMode | None:
    """Classify the execution perspective of an understood request."""
    if not isinstance(completed_request, CompletedRequest):
        raise TypeError("Execution-mode classification requires a CompletedRequest.")
    return _classify_with_llm(completed_request)
