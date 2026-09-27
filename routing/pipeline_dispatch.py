"""Map a resolved execution mode to an independent downstream pipeline."""

from dataclasses import dataclass
from enum import Enum

from schemas.hypothesis_schema import ExecutionMode


class PipelineType(str, Enum):
    """Independent downstream pipeline implementation boundaries."""

    RESEARCH = "research"
    HISTORICAL_REPLAY = "historical_replay"
    LIVE_FORECAST = "live_forecast"


@dataclass(frozen=True)
class PipelineDispatch:
    """Resolved routing semantics paired with its downstream pipeline."""

    execution_mode: ExecutionMode
    pipeline: PipelineType


def dispatch_execution_mode(
    execution_mode: ExecutionMode,
) -> PipelineDispatch:
    """Map one resolved execution mode to its explicit pipeline boundary."""
    if not isinstance(execution_mode, ExecutionMode):
        raise TypeError("execution_mode must be an ExecutionMode instance.")

    pipelines = {
        ExecutionMode.RESEARCH: PipelineType.RESEARCH,
        ExecutionMode.HISTORICAL_REPLAY: PipelineType.HISTORICAL_REPLAY,
        ExecutionMode.LIVE_FORECAST: PipelineType.LIVE_FORECAST,
    }
    try:
        pipeline = pipelines[execution_mode]
    except KeyError as error:
        raise ValueError(
            f"Unsupported execution mode: {execution_mode.value}"
        ) from error
    return PipelineDispatch(
        execution_mode=execution_mode,
        pipeline=pipeline,
    )
