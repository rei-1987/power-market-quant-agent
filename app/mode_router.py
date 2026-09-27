"""LLM-based top-level routing for Research / Historical Replay / Live Forecast."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os

try:
    from dotenv import load_dotenv
except ImportError:  # Optional convenience; environment variables still work without it.
    load_dotenv = None

from schemas.hypothesis_schema import ExecutionMode

DEFAULT_ROUTER_MODEL = "gpt-5.6-luna"


@dataclass(frozen=True)
class RouteDecision:
    mode: ExecutionMode
    reason: str


def _load_environment() -> None:
    if load_dotenv is not None:
        load_dotenv()


def _router_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": [
                    ExecutionMode.RESEARCH.value,
                    ExecutionMode.HISTORICAL_REPLAY.value,
                    ExecutionMode.LIVE_FORECAST.value,
                ],
            },
            "reason": {"type": "string"},
        },
        "required": ["mode", "reason"],
        "additionalProperties": False,
    }


def route_request(request: str) -> RouteDecision:
    """Use the LLM to choose the top-level execution mode.

    No keyword fallback is used. If the API is unavailable, the caller receives
    a real error rather than a fabricated routing result.
    """
    text = request.strip()
    if not text:
        raise ValueError("A non-empty natural-language request is required.")

    _load_environment()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. The real LLM router cannot run."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The OpenAI Python SDK is not installed. "
            "Run: python -m pip install openai"
        ) from exc

    instructions = """
You are the top-level mode router for a Power-Market Quant Research Agent.

Choose exactly one execution mode:

1. research
   Use when the user asks to investigate, test, compare, quantify, validate,
   or explore a historical/statistical relationship, hypothesis, signal,
   predictive value, conditional probability, or research question.

2. historical_replay
   Use when the user asks what the agent would have known, inferred, decided,
   or observed at a specific historical point in time, or asks to replay /
   step through a past market state using only information available then.

3. live_forecast
   Use when the answer depends on current/latest/live market information,
   today's or tonight's current state, monitoring, alerts, or newly arriving
   data.

Important:
- A request mentioning "delivery" is NOT automatically live.
- A historical relationship about delivery products is normally research.
- Do not route based on substring matches.
- Return a short factual reason for the classification.
"""

    client = OpenAI()
    response = client.responses.create(
        model=os.getenv("OPENAI_ROUTER_MODEL", DEFAULT_ROUTER_MODEL),
        input=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": text},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "power_market_mode_route",
                "strict": True,
                "schema": _router_schema(),
            }
        },
    )

    if not response.output_text:
        raise RuntimeError("The LLM router returned no output.")

    draft = json.loads(response.output_text)
    return RouteDecision(
        mode=ExecutionMode(draft["mode"]),
        reason=str(draft["reason"]),
    )
