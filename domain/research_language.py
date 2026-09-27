"""Constrained Power Market Quant Research Specification Language v0.1."""

from dataclasses import dataclass
from enum import Enum


class OutputType(str, Enum):
    """Supported forms of research output."""

    VALUE = "value"
    CHANGE = "change"
    PROBABILITY = "probability"
    DIFFERENCE = "difference"


class Aggregation(str, Enum):
    """Supported aggregations for numerical observations."""

    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    EXACT = "exact"


class ChangeDirection(str, Enum):
    """Supported directional results for change requests."""

    INCREASE = "increase"
    DECREASE = "decrease"
    UNCHANGED = "unchanged"


class DifferenceSemantics(str, Enum):
    """Supported interpretations of a numerical difference."""

    GENERIC = "generic"
    REVISION = "revision"
    DEVIATION = "deviation"


@dataclass(frozen=True)
class OutputSemantics:
    """Meaning and field contract for one output type."""

    description: str
    required_fields: tuple[str, ...]
    optional_fields: tuple[str, ...]
    historical_meaning: str | None = None
    forecast_meaning: str | None = None
    supported_outcomes: tuple[ChangeDirection, ...] = ()


OUTPUT_SEMANTICS: dict[OutputType, OutputSemantics] = {
    OutputType.VALUE: OutputSemantics(
        description="A numerical value of a target variable.",
        historical_meaning="An observed or aggregated value from historical data.",
        forecast_meaning="A predicted future numerical value.",
        required_fields=("target",),
        optional_fields=("aggregation", "conditions", "filters"),
    ),
    OutputType.CHANGE: OutputSemantics(
        description="The direction of a target relative to a reference.",
        required_fields=("target", "reference"),
        optional_fields=("conditions", "filters"),
        supported_outcomes=(
            ChangeDirection.INCREASE,
            ChangeDirection.DECREASE,
            ChangeDirection.UNCHANGED,
        ),
    ),
    OutputType.PROBABILITY: OutputSemantics(
        description="The probability of a clearly defined event.",
        required_fields=("event",),
        optional_fields=("conditions", "filters", "baseline"),
    ),
    OutputType.DIFFERENCE: OutputSemantics(
        description="The numerical difference between two comparable quantities.",
        required_fields=("target", "reference"),
        optional_fields=(
            "aggregation",
            "difference_semantics",
            "conditions",
            "filters",
        ),
    ),
}
