"""Typed structures for compiled power-market research specifications."""

from dataclasses import dataclass
from enum import Enum

from domain.research_language import (
    Aggregation,
    ChangeDirection,
    DifferenceSemantics,
    OutputType,
)


class SpecificationStatus(str, Enum):
    """Compilation and point-in-time status of a research specification."""

    VALID = "valid"
    NEEDS_CLARIFICATION = "needs_clarification"
    UNSUPPORTED = "unsupported"


class ExecutionMode(str, Enum):
    """Top-level mode controlling how the agent executes a task."""

    RESEARCH = "research"
    HISTORICAL_REPLAY = "historical_replay"
    LIVE_FORECAST = "live_forecast"


class StudyPeriodType(str, Enum):
    """Supported representations of a historical research sample period."""

    ABSOLUTE_DATE_RANGE = "absolute_date_range"
    RELATIVE_LOOKBACK = "relative_lookback"
    ALL_AVAILABLE = "all_available"


class RelativePeriodUnit(str, Enum):
    """Calendar units supported by a relative historical lookback."""

    DAYS = "days"
    WEEKS = "weeks"
    MONTHS = "months"
    YEARS = "years"


@dataclass(frozen=True)
class StudyPeriod:
    """Historical dates included in a research sample."""

    type: StudyPeriodType
    start_date: str | None = None
    end_date: str | None = None
    lookback_value: int | None = None
    lookback_unit: RelativePeriodUnit | None = None


@dataclass(frozen=True)
class DeliveryWindow:
    """Market-local delivery interval requested by the user."""

    start_local: str | None = None
    end_local: str | None = None
    date: str | None = None


class DecisionPointType(str, Enum):
    """Supported ways to express when a decision is made."""

    ABSOLUTE_LOCAL_TIME = "absolute_local_time"
    RELATIVE_TO_DELIVERY = "relative_to_delivery"


@dataclass(frozen=True)
class DecisionPoint:
    """A market-local or delivery-relative decision point."""

    type: DecisionPointType
    local_time: str | None = None
    minutes_before_delivery: int | None = None
    label: str | None = None


ConditionValue = int | float | str | bool | tuple[object, ...]


@dataclass(frozen=True)
class Condition:
    """A machine-readable condition, event, or sample restriction."""

    variable: str
    operator: str
    value: ConditionValue


@dataclass(frozen=True)
class ClarificationOption:
    """Machine-readable value and user-facing label for a suggested answer."""

    value: str
    label: str


@dataclass(frozen=True)
class ClarificationRequest:
    """Structured follow-up question for an unresolved specification field."""

    field: str
    question: str
    options: tuple[ClarificationOption, ...] = ()
    allow_free_text: bool = True


@dataclass(frozen=True)
class ResearchSpecification:
    """Immutable structure produced by the research-hypothesis compiler."""

    original_hypothesis: str
    status: SpecificationStatus
    output_type: OutputType | None = None
    target: str | None = None
    reference: str | None = None
    event: Condition | None = None
    conditions: tuple[Condition, ...] = ()
    filters: tuple[Condition, ...] = ()
    baseline: tuple[Condition, ...] = ()
    aggregation: Aggregation | None = None
    change_direction: ChangeDirection | None = None
    difference_semantics: DifferenceSemantics | None = None
    ambiguities: tuple[str, ...] = ()
    unsupported_terms: tuple[str, ...] = ()
    compiler_notes: tuple[str, ...] = ()
    delivery_window: DeliveryWindow | None = None
    decision_points: tuple[DecisionPoint, ...] = ()
    study_period: StudyPeriod | None = None
    clarification_requests: tuple[ClarificationRequest, ...] = ()


@dataclass(frozen=True)
class ValidationResult:
    """Outcome returned by specification validation."""

    is_valid: bool
    errors: tuple[str, ...] = ()
