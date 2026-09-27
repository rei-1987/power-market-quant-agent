"""Runtime bridge used by the UI.

This module connects the real LLM router to the existing Research compiler,
validator, state, and data-loading code. It does not fabricate statistical
results. Unimplemented pipelines are reported honestly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from app.agent_controller import AgentController
from app.mode_router import RouteDecision, route_request
from execution.historical_replay_pipeline import run_historical_replay
from execution.live_forecast_pipeline import run_live_forecast
from execution.research_pipeline import execute_research_specification
from schemas.hypothesis_schema import (
    ExecutionMode,
    SpecificationStatus,
)
from tools.data_source import resolve_master_dataset_path


@dataclass(frozen=True)
class RuntimeEvent:
    label: str
    status: str
    message: str


@dataclass
class RuntimeResult:
    request: str
    mode: ExecutionMode
    route_reason: str
    status: str
    message: str
    events: list[RuntimeEvent]
    agent_result: Any | None = None
    data_summary: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["mode"] = self.mode.value
        return payload


class AgentRuntime:
    """Stateful UI-facing runtime.

    Research mode is connected to the existing AgentController.
    Replay and Live are routed by the real LLM router but are not faked if
    their backend pipelines are not yet implemented.
    """

    def __init__(self) -> None:
        if load_dotenv is not None:
            load_dotenv()
        self.research_controller = AgentController()
        self.current_mode: ExecutionMode | None = None
        self.route_decision: RouteDecision | None = None
        self.last_request: str | None = None

    @property
    def pending_clarifications(self):
        return self.research_controller.state.pending_clarifications

    def _finalize_research(
        self,
        request: str,
        route: RouteDecision,
        agent_result: Any,
        events: list[RuntimeEvent],
    ) -> RuntimeResult:
        specification = agent_result.specification
        validation = agent_result.validation

        events.append(
            RuntimeEvent(
                "COMPILER",
                specification.status.value,
                f"Research specification status: {specification.status.value}.",
            )
        )
        events.append(
            RuntimeEvent(
                "VALIDATOR",
                "passed" if validation.is_valid else "failed",
                (
                    "Specification passed deterministic validation."
                    if validation.is_valid
                    else "; ".join(validation.errors)
                ),
            )
        )

        if specification.status == SpecificationStatus.NEEDS_CLARIFICATION:
            return RuntimeResult(
                request=request,
                mode=route.mode,
                route_reason=route.reason,
                status="needs_clarification",
                message="The backend needs clarification before research can continue.",
                events=events,
                agent_result=agent_result,
            )

        if specification.status == SpecificationStatus.UNSUPPORTED:
            return RuntimeResult(
                request=request,
                mode=route.mode,
                route_reason=route.reason,
                status="unsupported",
                message="The request requires concepts outside the current data/variable boundary.",
                events=events,
                agent_result=agent_result,
            )

        if not validation.is_valid:
            return RuntimeResult(
                request=request,
                mode=route.mode,
                route_reason=route.reason,
                status="validation_error",
                message="The compiled specification failed deterministic validation.",
                events=events,
                agent_result=agent_result,
            )

        try:
            data_path = resolve_master_dataset_path()
            pipeline_result = execute_research_specification(
                specification,
                validation,
                data_path,
            )
        except (RuntimeError, FileNotFoundError) as exc:
            events.append(
                RuntimeEvent(
                    "DATA",
                    "not_configured",
                    str(exc),
                )
            )
            return RuntimeResult(
                request=request,
                mode=route.mode,
                route_reason=route.reason,
                status="data_not_configured",
                message=(
                    "Routing, compilation, and validation are real and connected. "
                    "Set MARKET_DATA_CSV to continue into real data loading."
                ),
                events=events,
                agent_result=agent_result,
            )

        frame = pipeline_result.prepared_data
        quant = pipeline_result.quantitative_result
        data_summary = {
            "path": str(data_path),
            "original_question": specification.original_hypothesis,
            "rows_loaded": int(len(frame)),
            "columns_loaded": list(frame.columns),
            "required_columns": list(pipeline_result.required_source_columns),
            "quantitative_result": asdict(quant) if quant is not None else None,
            "resolved_study_period": (
                asdict(pipeline_result.resolved_study_period)
                if pipeline_result.resolved_study_period is not None
                else None
            ),
            "interpretation": pipeline_result.interpretation,
        }
        events.append(
            RuntimeEvent(
                "DATA",
                "loaded",
                f"Loaded {len(frame):,} rows from the real dataset.",
            )
        )

        events.append(
            RuntimeEvent(
                "QUANT TEST",
                "completed" if quant is not None else "not_implemented",
                (
                    "Conditional probability analysis completed."
                    if quant is not None
                    else "No Quant Core method is implemented for this specification."
                ),
            )
        )

        return RuntimeResult(
            request=request,
            mode=route.mode,
            route_reason=route.reason,
            status="research_complete" if quant is not None else "ready_for_quant_test",
            message=(
                "Real Research compilation, validation, data preparation, and "
                "quantitative analysis completed."
                if quant is not None
                else "Data preparation completed, but no Quant Core method is implemented."
            ),
            events=events,
            agent_result=agent_result,
            data_summary=data_summary,
        )

    def run(self, request: str) -> RuntimeResult:
        text = request.strip()
        if not text:
            raise ValueError("A non-empty request is required.")

        self.last_request = text
        route = route_request(text)
        self.route_decision = route
        self.current_mode = route.mode

        events = [
            RuntimeEvent(
                "ROUTER",
                "completed",
                f"LLM selected {route.mode.value}: {route.reason}",
            )
        ]

        if route.mode == ExecutionMode.RESEARCH:
            result = self.research_controller.process(text)
            return self._finalize_research(text, route, result, events)

        if route.mode == ExecutionMode.HISTORICAL_REPLAY:
            try:
                data_path = resolve_master_dataset_path()
                replay_result = run_historical_replay(text, data_path)
            except (RuntimeError, FileNotFoundError) as exc:
                events.append(RuntimeEvent("DATA", "not_configured", str(exc)))
                return RuntimeResult(
                    request=text,
                    mode=route.mode,
                    route_reason=route.reason,
                    status="data_not_configured",
                    message="Set MARKET_DATA_CSV to run Historical Replay.",
                    events=events,
                )
            events.append(
                RuntimeEvent(
                    "DATA",
                    "loaded",
                    "Loaded point-in-time Replay calibration and delivery data.",
                )
            )
            events.append(
                RuntimeEvent(
                    "REPLAY QUANT",
                    "completed",
                    "Historical sign-regime inference and separate ex-post check completed.",
                )
            )
            return RuntimeResult(
                request=text,
                mode=route.mode,
                route_reason=route.reason,
                status="replay_complete",
                message="Historical Replay completed without using replay-date outcomes in inference.",
                events=events,
                data_summary={
                    "path": str(data_path),
                    "replay_result": asdict(replay_result),
                },
            )

        try:
            data_path = resolve_master_dataset_path()
            live_result = run_live_forecast(text, data_path)
        except (RuntimeError, FileNotFoundError) as exc:
            events.append(RuntimeEvent("DATA", "not_configured", str(exc)))
            return RuntimeResult(
                request=text,
                mode=route.mode,
                route_reason=route.reason,
                status="data_not_configured",
                message="Set MARKET_DATA_CSV to run Live Forecast.",
                events=events,
            )
        events.append(
            RuntimeEvent(
                "DATA",
                "loaded",
                "Loaded pre-date calibration and latest-date A01/A40 data.",
            )
        )
        events.append(
            RuntimeEvent(
                "LIVE QUANT",
                "completed",
                "Simulated-live historical regime inference completed.",
            )
        )
        return RuntimeResult(
            request=text,
            mode=route.mode,
            route_reason=route.reason,
            status="live_complete",
            message="Simulated Live Forecast completed without current-date outcomes in inference.",
            events=events,
            data_summary={
                "path": str(data_path),
                "live_result": asdict(live_result),
            },
        )

    def answer_clarification(self, answer_text: str) -> RuntimeResult:
        if self.current_mode != ExecutionMode.RESEARCH:
            raise ValueError("Clarification continuation is currently supported only in Research mode.")
        if self.route_decision is None or self.last_request is None:
            raise ValueError("No active Research request exists.")

        result = self.research_controller.answer_clarification(answer_text)
        events = [
            RuntimeEvent(
                "CLARIFICATION",
                "received",
                "Clarification answer was applied to the active Research specification.",
            )
        ]
        return self._finalize_research(
            self.last_request,
            self.route_decision,
            result,
            events,
        )
