"""Understand and complete user requests before execution-mode classification."""

import json
import os
import re
from dataclasses import dataclass
from datetime import date
from enum import Enum

from compiler.variable_registry import DOMAIN_TERMS, VARIABLE_REGISTRY


DEFAULT_SCOPE_MODEL = "gpt-5.6-luna"


RequestValue = int | float | str | bool | tuple[object, ...]
_PREDICATE_OPERATORS = {">", "<", ">=", "<=", "=", "==", "between"}
_ISO_DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
_LOCAL_TIME_PATTERN = re.compile(r"(?:[01]\d|2[0-3]):[0-5]\d\Z")


class RequestViewKind(str, Enum):
    """Mode-neutral quantitative perspectives requested by the user."""

    VALUE = "value"
    PROBABILITY = "probability"
    DIFFERENCE = "difference"
    CHANGE = "change"


class RequestAggregation(str, Enum):
    """Supported aggregation semantics in a completed request."""

    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    EXACT = "exact"


class RequestTimeScopeType(str, Enum):
    """Supported forms of request-level calendar scope."""

    ABSOLUTE_DATE_RANGE = "absolute_date_range"
    RELATIVE_LOOKBACK = "relative_lookback"
    ALL_AVAILABLE = "all_available"


class RequestRelativePeriodUnit(str, Enum):
    """Supported relative calendar units."""

    DAYS = "days"
    WEEKS = "weeks"
    MONTHS = "months"
    YEARS = "years"


class RequestDecisionPointType(str, Enum):
    """Supported mode-neutral decision timing representations."""

    ABSOLUTE_LOCAL_TIME = "absolute_local_time"
    MINUTES_BEFORE_DELIVERY = "minutes_before_delivery"


class RequestChangeDirection(str, Enum):
    """Supported requested directions for change views."""

    INCREASE = "increase"
    DECREASE = "decrease"
    UNCHANGED = "unchanged"


class RequestDifferenceSemantics(str, Enum):
    """Supported meanings for requested numerical differences."""

    GENERIC = "generic"
    REVISION = "revision"
    DEVIATION = "deviation"


def _validate_iso_date(value: str | None, field_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not _ISO_DATE_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must use YYYY-MM-DD format.")
    try:
        date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field_name} must be a valid ISO date.") from error


def _validate_local_time(value: str | None, field_name: str) -> None:
    if value is not None and (
        not isinstance(value, str) or not _LOCAL_TIME_PATTERN.fullmatch(value)
    ):
        raise ValueError(f"{field_name} must use HH:MM format.")


@dataclass(frozen=True)
class RequestTimeScope:
    """Structured calendar scope without execution-date resolution."""

    scope_type: RequestTimeScopeType
    start_date: str | None = None
    end_date: str | None = None
    lookback_value: int | None = None
    lookback_unit: RequestRelativePeriodUnit | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scope_type, RequestTimeScopeType):
            raise ValueError("Request time scope has an unsupported scope_type.")
        _validate_iso_date(self.start_date, "Request time scope start_date")
        _validate_iso_date(self.end_date, "Request time scope end_date")
        if self.scope_type == RequestTimeScopeType.ABSOLUTE_DATE_RANGE:
            if self.start_date is None and self.end_date is None:
                raise ValueError("ABSOLUTE_DATE_RANGE requires at least one date.")
            if self.lookback_value is not None or self.lookback_unit is not None:
                raise ValueError("ABSOLUTE_DATE_RANGE cannot contain lookback fields.")
        elif self.scope_type == RequestTimeScopeType.RELATIVE_LOOKBACK:
            if (
                isinstance(self.lookback_value, bool)
                or not isinstance(self.lookback_value, int)
                or self.lookback_value <= 0
            ):
                raise ValueError("RELATIVE_LOOKBACK requires a positive lookback_value.")
            if not isinstance(self.lookback_unit, RequestRelativePeriodUnit):
                raise ValueError("RELATIVE_LOOKBACK requires a lookback_unit.")
            if self.start_date is not None or self.end_date is not None:
                raise ValueError("RELATIVE_LOOKBACK cannot contain absolute dates.")
        elif any(
            value is not None
            for value in (
                self.start_date,
                self.end_date,
                self.lookback_value,
                self.lookback_unit,
            )
        ):
            raise ValueError("ALL_AVAILABLE cannot contain date or lookback fields.")


@dataclass(frozen=True)
class RequestDeliveryScope:
    """Structured market-local delivery date or time window."""

    date: str | None = None
    start_local: str | None = None
    end_local: str | None = None

    def __post_init__(self) -> None:
        _validate_iso_date(self.date, "Request delivery scope date")
        _validate_local_time(self.start_local, "Request delivery scope start_local")
        _validate_local_time(self.end_local, "Request delivery scope end_local")
        if (self.start_local is None) != (self.end_local is None):
            raise ValueError("Delivery start_local and end_local must appear together.")
        if self.date is None and self.start_local is None:
            raise ValueError("Delivery scope requires a date or local time window.")


@dataclass(frozen=True)
class RequestDecisionPoint:
    """One structured mode-neutral decision time."""

    point_type: RequestDecisionPointType
    local_time: str | None = None
    minutes_before_delivery: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.point_type, RequestDecisionPointType):
            raise ValueError("Request decision point has an unsupported point_type.")
        _validate_local_time(self.local_time, "Request decision point local_time")
        if self.point_type == RequestDecisionPointType.ABSOLUTE_LOCAL_TIME:
            if self.local_time is None:
                raise ValueError("ABSOLUTE_LOCAL_TIME requires local_time.")
            if self.minutes_before_delivery is not None:
                raise ValueError(
                    "ABSOLUTE_LOCAL_TIME cannot contain minutes_before_delivery."
                )
        elif (
            isinstance(self.minutes_before_delivery, bool)
            or not isinstance(self.minutes_before_delivery, int)
            or self.minutes_before_delivery <= 0
        ):
            raise ValueError(
                "MINUTES_BEFORE_DELIVERY requires positive minutes_before_delivery."
            )
        elif self.local_time is not None:
            raise ValueError("MINUTES_BEFORE_DELIVERY cannot contain local_time.")


@dataclass(frozen=True)
class RequestPredicate:
    """Closed-world predicate understood before execution-mode selection."""

    variable: str
    operator: str
    value: RequestValue

    def __post_init__(self) -> None:
        if self.variable not in VARIABLE_REGISTRY:
            raise ValueError("Request predicate variable must be canonical.")
        if self.operator not in _PREDICATE_OPERATORS:
            raise ValueError("Request predicate has an unsupported operator.")
        if self.operator == "between" and (
            not isinstance(self.value, tuple) or len(self.value) != 2
        ):
            raise ValueError("A 'between' predicate requires two boundary values.")


@dataclass(frozen=True)
class RequestedQuantitativeView:
    """One requested numerical perspective without execution semantics."""

    kind: RequestViewKind
    target: str | None = None
    reference: str | None = None
    aggregation: RequestAggregation | None = None
    event: RequestPredicate | None = None
    conditions: tuple[RequestPredicate, ...] = ()
    baseline: tuple[RequestPredicate, ...] = ()
    change_direction: RequestChangeDirection | None = None
    difference_semantics: RequestDifferenceSemantics | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RequestViewKind):
            raise ValueError("Quantitative view has an unsupported kind.")
        for field_name, variable in (
            ("target", self.target),
            ("reference", self.reference),
        ):
            if variable is not None and variable not in VARIABLE_REGISTRY:
                raise ValueError(f"Quantitative view {field_name} must be canonical.")
        if self.aggregation is not None and not isinstance(
            self.aggregation, RequestAggregation
        ):
            raise ValueError("Quantitative view has an unsupported aggregation.")
        if self.event is not None and not isinstance(self.event, RequestPredicate):
            raise ValueError("Quantitative view event must be a request predicate.")
        if any(
            not isinstance(predicate, RequestPredicate)
            for predicate in (*self.conditions, *self.baseline)
        ):
            raise ValueError(
                "Quantitative view conditions and baseline must be request predicates."
            )
        if self.change_direction is not None and not isinstance(
            self.change_direction, RequestChangeDirection
        ):
            raise ValueError("Quantitative view has an unsupported change direction.")
        if self.difference_semantics is not None and not isinstance(
            self.difference_semantics, RequestDifferenceSemantics
        ):
            raise ValueError("Quantitative view has unsupported difference semantics.")
        if self.kind == RequestViewKind.VALUE and self.target is None:
            raise ValueError("VALUE view requires a target.")
        if self.kind == RequestViewKind.PROBABILITY and self.event is None:
            raise ValueError("PROBABILITY view requires an event.")
        if self.kind in {RequestViewKind.DIFFERENCE, RequestViewKind.CHANGE}:
            if self.target is None or self.reference is None:
                raise ValueError(
                    f"{self.kind.value.upper()} view requires target and reference."
                )


@dataclass(frozen=True)
class CompletedRequest:
    """Canonical semantic request produced before mode classification."""

    original_question: str
    normalized_request: str
    resolved_variables: tuple[str, ...] = ()
    filters: tuple[RequestPredicate, ...] = ()
    time_scope: RequestTimeScope | None = None
    delivery_scope: RequestDeliveryScope | None = None
    decision_points: tuple[RequestDecisionPoint, ...] = ()
    requested_quantitative_views: tuple[RequestedQuantitativeView, ...] = ()
    execution_perspective: str | None = None
    assumptions: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.requested_quantitative_views:
            raise ValueError("Completed request requires at least one quantitative view.")
        if len(set(self.resolved_variables)) != len(self.resolved_variables) or any(
            variable not in VARIABLE_REGISTRY for variable in self.resolved_variables
        ):
            raise ValueError(
                "Completed request resolved_variables must be unique canonical variables."
            )
        if any(not isinstance(predicate, RequestPredicate) for predicate in self.filters):
            raise ValueError("Completed request filters must be request predicates.")
        if self.time_scope is not None and not isinstance(
            self.time_scope, RequestTimeScope
        ):
            raise ValueError("Completed request time_scope must be structured.")
        if self.delivery_scope is not None and not isinstance(
            self.delivery_scope, RequestDeliveryScope
        ):
            raise ValueError("Completed request delivery_scope must be structured.")
        if any(
            not isinstance(point, RequestDecisionPoint)
            for point in self.decision_points
        ):
            raise ValueError("Completed request decision_points must be structured.")


@dataclass(frozen=True)
class ScopeClarificationRequest:
    """One material scope detail that the user still needs to provide."""

    field: str
    question: str
    options: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScopeClarificationTurn:
    """A user answer and the scope questions that were pending at the time."""

    requests: tuple[ScopeClarificationRequest, ...]
    answer_text: str


@dataclass(frozen=True)
class RequestUnderstandingState:
    """Immutable state for the scope-clarification conversation."""

    original_question: str
    pending_requests: tuple[ScopeClarificationRequest, ...] = ()
    clarification_history: tuple[ScopeClarificationTurn, ...] = ()
    is_complete: bool = False
    completed_request: CompletedRequest | None = None

    def __post_init__(self) -> None:
        if self.is_complete and self.completed_request is None:
            raise ValueError("A complete scope requires a completed request.")
        if self.is_complete and self.pending_requests:
            raise ValueError("A complete scope cannot contain clarification requests.")
        if not self.is_complete and self.completed_request is not None:
            raise ValueError("An incomplete scope cannot contain a completed request.")


_FALLBACK_REQUEST = ScopeClarificationRequest(
    field="scope",
    question="Could you provide a little more detail about the scope of the task?",
)


def _scope_checker_instructions() -> str:
    return """Determine only whether the user's request has enough material scope
and intent to proceed to execution-mode classification. Mark a request complete
only when an independent downstream classifier should be able to determine one
and only one execution behavior without asking the user another question.

Do not classify the request as research, historical replay, or live forecast.
Do not solve the user's problem and do not invent missing values. Ask only for
scope information whose absence materially prevents the request from being
understood and materially requires a user choice, such as a necessary time or
delivery scope, target, comparison, or requested quantity. Do not collect every
field an eventual compiler might need, and do not ask unnecessary questions
when the request is already sufficiently specific.

Completion requires enough understanding of the object or variable, the target
or comparison, the requested outcome or question, any materially necessary time
or delivery scope, and the execution-time perspective when it could change how
the request is carried out. This is not a rigid checklist. Ask only about gaps
that are material to this particular request or could change the downstream
pipeline.

Treat material execution-intent ambiguity as incomplete. In particular,
distinguish semantically between:
- analyzing existing historical evidence or whether a historical relationship
  has explanatory or predictive value;
- reconstructing predictions at historical decision points using only the
  information available then, often comparing them with later outcomes; and
- making a prediction now for an outcome that has not happened yet.

Do not output or ask the user to choose internal execution-mode labels. Ask a
natural question about the unresolved intent instead. For example, clarify
whether the user wants to study a historical predictive relationship or
recreate point-in-time predictions that could have been made during that
period. A statement such as "I want to use it for prediction" remains
incomplete when it does not say whether the user wants historical evidence,
historical point-in-time reconstruction, or a prediction of an unrealized
outcome. Do not infer this distinction merely from words such as "prediction"
or "predictive value".

Do not ask for information already available from known project or domain
metadata, execution context supplied later by the system, or deterministic
runtime resolution. Treat known domain terms and registered variable aliases in
the supplied project domain context as already understood. For example, do not
ask what ID1, A40, or total_wind_revision means when it is present there.

Relative periods such as "last 30 days", "last week", "last 3 months", "last
year", "past 2 years", and "all available history" are sufficiently specified
user scope. Do not ask for a calendar end date or concrete YYYY-MM-DD dates. A
later execution layer will resolve a relative period against an explicit as-of
date.

Distinguish user scope, which expresses what the user wants, from execution
details the system can determine later. User scope includes phrases such as
"last 3 months", "ID1", "average", "18:00 delivery", and "compare ID1 with
ID3". Execution details include the exact dates represented by a relative
period, the registered meaning of a term, physical source columns, derived
formulas, and data-availability boundaries. Never request an execution detail
merely to mark the scope complete.

Treat the original request and clarification dialogue as separate context.
Preserve information supplied in earlier answers. A user may answer only some
pending questions, so re-evaluate what material information is still missing
after every turn.

Clear requests must not be over-questioned. A historical average or explicitly
historical relationship analysis, an explicit reconstruction using information
available at past decision points, and an explicit prediction made now for a
future outcome are each sufficiently clear when their substantive scope is also
understood.

When scope is complete, produce a mode-neutral completed_request that preserves
the user's accumulated semantics. Canonicalize every clearly identified known
variable using the supplied registered variable names. Never invent a variable,
threshold, reference, period, or delivery window. Record execution_perspective
as a plain description such as "analyze realized historical evidence",
"reconstruct a historical point-in-time prediction", or "predict an unrealized
future period from now". Do not put an execution-mode or pipeline label in the
completed request.

Emit calendar, delivery, and decision timing only through the structured fields
defined by the output schema. Never leave phrases such as "last 3 months",
"17:00 to 21:00", or "three hours before delivery" as free-form scope values.
Each requested_quantitative_view must be self-contained: put its target,
reference, event, conditions, baseline, aggregation, change direction, and
difference semantics in that view. Request-level filters are shared sample
restrictions that apply to every view. A completed request must contain at least
one structurally complete quantitative view.

A sufficiently specified broad relationship question may expand into multiple
requested_quantitative_views when each view follows directly from the request.
For example, a stated negative-wind condition and requested ID1-ID3 effect may
support a mean VALUE view under that condition. Add a PROBABILITY view only when
the positive event was stated or is clearly part of the user's hypothesis. Add
DIFFERENCE or CHANGE only when the request supplies the necessary comparison.
Do not introduce relationship or hypothesis-test view kinds, and do not force a
clarification merely because more than one faithful quantitative view exists.

Before returning is_complete=true, ask internally: "If this accumulated request
were passed to an independent execution-mode classifier right now, should that
classifier be able to determine one and only one execution behavior without
asking the user anything else?" If no, return is_complete=false and ask for the
material missing information. If yes, return is_complete=true.

If the request is sufficiently specified, return is_complete=true, an empty
clarification_requests array, and a fully populated completed_request. Otherwise
return is_complete=false, at least one concise clarification request, and a null
completed_request. Never ask the user to choose an execution mode. Return only
the required JSON.
"""


def _known_domain_context() -> str:
    """Build concise reference knowledge from the closed-world registries."""
    lines = ["Named domain terms:"]
    for canonical, term in DOMAIN_TERMS.items():
        aliases = ", ".join(term.aliases) or "none"
        lines.append(
            f"- {canonical} ({term.name}); aliases: {aliases}; meaning: {term.meaning}"
        )

    lines.append("Registered variables:")
    for canonical, metadata in VARIABLE_REGISTRY.items():
        aliases = ", ".join(metadata.aliases) or "none"
        lines.append(
            f"- {canonical}; aliases: {aliases}; description: {metadata.description}"
        )
    return "\n".join(lines)


def _scope_json_schema() -> dict[str, object]:
    scalar = {
        "anyOf": [
            {"type": "number"},
            {"type": "string"},
            {"type": "boolean"},
        ]
    }
    predicate = {
        "type": "object",
        "properties": {
            "variable": {"type": "string"},
            "operator": {
                "type": "string",
                "enum": [">", "<", ">=", "<=", "=", "==", "between"],
            },
            "value": {
                "anyOf": [
                    *scalar["anyOf"],
                    {"type": "array", "items": scalar},
                ]
            },
        },
        "required": ["variable", "operator", "value"],
        "additionalProperties": False,
    }
    predicate_array = {"type": "array", "items": predicate}
    nullable_string = {"type": ["string", "null"]}
    nullable_integer = {"type": ["integer", "null"]}
    time_scope = {
        "type": "object",
        "properties": {
            "scope_type": {
                "type": "string",
                "enum": [
                    "absolute_date_range",
                    "relative_lookback",
                    "all_available",
                ],
            },
            "start_date": nullable_string,
            "end_date": nullable_string,
            "lookback_value": nullable_integer,
            "lookback_unit": {
                "enum": ["days", "weeks", "months", "years", None],
            },
        },
        "required": [
            "scope_type",
            "start_date",
            "end_date",
            "lookback_value",
            "lookback_unit",
        ],
        "additionalProperties": False,
    }
    delivery_scope = {
        "type": "object",
        "properties": {
            "date": nullable_string,
            "start_local": nullable_string,
            "end_local": nullable_string,
        },
        "required": ["date", "start_local", "end_local"],
        "additionalProperties": False,
    }
    decision_point = {
        "type": "object",
        "properties": {
            "point_type": {
                "type": "string",
                "enum": ["absolute_local_time", "minutes_before_delivery"],
            },
            "local_time": nullable_string,
            "minutes_before_delivery": nullable_integer,
        },
        "required": ["point_type", "local_time", "minutes_before_delivery"],
        "additionalProperties": False,
    }
    quantitative_view = {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["value", "probability", "difference", "change"],
            },
            "target": nullable_string,
            "reference": nullable_string,
            "aggregation": {
                "enum": ["mean", "median", "min", "max", "exact", None],
            },
            "event": {"anyOf": [predicate, {"type": "null"}]},
            "conditions": predicate_array,
            "baseline": predicate_array,
            "change_direction": {
                "enum": ["increase", "decrease", "unchanged", None],
            },
            "difference_semantics": {
                "enum": ["generic", "revision", "deviation", None],
            },
        },
        "required": [
            "kind",
            "target",
            "reference",
            "aggregation",
            "event",
            "conditions",
            "baseline",
            "change_direction",
            "difference_semantics",
        ],
        "additionalProperties": False,
    }
    completed_request = {
        "type": "object",
        "properties": {
            "original_question": {"type": "string"},
            "normalized_request": {"type": "string"},
            "resolved_variables": {
                "type": "array",
                "items": {"type": "string"},
            },
            "filters": predicate_array,
            "time_scope": {"anyOf": [time_scope, {"type": "null"}]},
            "delivery_scope": {"anyOf": [delivery_scope, {"type": "null"}]},
            "decision_points": {
                "type": "array",
                "items": decision_point,
            },
            "requested_quantitative_views": {
                "type": "array",
                "items": quantitative_view,
            },
            "execution_perspective": nullable_string,
            "assumptions": {
                "type": "array",
                "items": {"type": "string"},
            },
            "notes": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "original_question",
            "normalized_request",
            "resolved_variables",
            "filters",
            "time_scope",
            "delivery_scope",
            "decision_points",
            "requested_quantitative_views",
            "execution_perspective",
            "assumptions",
            "notes",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "is_complete": {"type": "boolean"},
            "clarification_requests": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "string"},
                        "question": {"type": "string"},
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["field", "question", "options"],
                    "additionalProperties": False,
                },
            },
            "completed_request": {
                "anyOf": [completed_request, {"type": "null"}],
            },
        },
        "required": [
            "is_complete",
            "clarification_requests",
            "completed_request",
        ],
        "additionalProperties": False,
    }


def _build_scope_context(
    original_question: str,
    history: tuple[ScopeClarificationTurn, ...],
) -> str:
    lines = [
        "Known project domain:",
        _known_domain_context(),
        "",
        "Original request:",
        original_question,
    ]
    if history:
        lines.extend(("", "Clarification dialogue:"))
        for turn in history:
            for request in turn.requests:
                lines.append(f"Q: {request.question}")
            lines.append(f"A: {turn.answer_text}")
            lines.append("")
    lines.append(
        "Determine whether any MATERIAL scope information is still missing."
    )
    return "\n".join(lines)


def _unresolved_fallback(
    original_question: str,
    history: tuple[ScopeClarificationTurn, ...],
) -> RequestUnderstandingState:
    return RequestUnderstandingState(
        original_question=original_question,
        pending_requests=(_FALLBACK_REQUEST,),
        clarification_history=history,
        is_complete=False,
    )


def _canonical_variable(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value not in VARIABLE_REGISTRY:
        raise ValueError(
            f"Completed request {field_name} must be a canonical registered variable."
        )
    return value


def _optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Completed request {field_name} must be non-empty text or null."
        )
    return value.strip()


def _text_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(
            f"Completed request {field_name} must be an array of non-empty strings."
        )
    return tuple(item.strip() for item in value)


def _predicate_from_payload(value: object) -> RequestPredicate:
    if not isinstance(value, dict) or set(value) != {
        "variable",
        "operator",
        "value",
    }:
        raise ValueError("Completed request predicate has an invalid structure.")
    variable = _canonical_variable(value["variable"], "predicate variable")
    if variable is None:
        raise ValueError("Completed request predicate requires a variable.")
    operator = value["operator"]
    raw_value = value["value"]
    if not isinstance(operator, str) or operator not in _PREDICATE_OPERATORS:
        raise ValueError("Completed request predicate has an unsupported operator.")
    if isinstance(raw_value, list):
        if any(
            not isinstance(item, (int, float, str, bool)) for item in raw_value
        ):
            raise ValueError("Completed request predicate has an invalid value.")
        predicate_value: RequestValue = tuple(raw_value)
    elif isinstance(raw_value, (int, float, str, bool)):
        predicate_value = raw_value
    else:
        raise ValueError("Completed request predicate has an invalid value.")
    if operator == "between" and (
        not isinstance(predicate_value, tuple) or len(predicate_value) != 2
    ):
        raise ValueError(
            "Completed request 'between' predicate requires two boundary values."
        )
    return RequestPredicate(
        variable=variable,
        operator=operator,
        value=predicate_value,
    )


def _predicate_tuple(value: object, field_name: str) -> tuple[RequestPredicate, ...]:
    if not isinstance(value, list):
        raise ValueError(f"Completed request {field_name} must be an array.")
    return tuple(_predicate_from_payload(item) for item in value)


def _optional_enum(value: object, enum_type: type[Enum], field_name: str) -> Enum | None:
    if value is None:
        return None
    try:
        return enum_type(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Completed request {field_name} is unsupported.") from error


def _time_scope_from_payload(value: object) -> RequestTimeScope | None:
    if value is None:
        return None
    expected = {
        "scope_type",
        "start_date",
        "end_date",
        "lookback_value",
        "lookback_unit",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("Completed request time_scope has an invalid structure.")
    try:
        scope_type = RequestTimeScopeType(value["scope_type"])
    except (TypeError, ValueError) as error:
        raise ValueError("Completed request time_scope has an unsupported type.") from error
    lookback_unit = _optional_enum(
        value["lookback_unit"], RequestRelativePeriodUnit, "lookback_unit"
    )
    return RequestTimeScope(
        scope_type=scope_type,
        start_date=value["start_date"],
        end_date=value["end_date"],
        lookback_value=value["lookback_value"],
        lookback_unit=lookback_unit,
    )


def _delivery_scope_from_payload(value: object) -> RequestDeliveryScope | None:
    if value is None:
        return None
    expected = {"date", "start_local", "end_local"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("Completed request delivery_scope has an invalid structure.")
    return RequestDeliveryScope(
        date=value["date"],
        start_local=value["start_local"],
        end_local=value["end_local"],
    )


def _decision_point_from_payload(value: object) -> RequestDecisionPoint:
    expected = {"point_type", "local_time", "minutes_before_delivery"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("Completed request decision point has an invalid structure.")
    try:
        point_type = RequestDecisionPointType(value["point_type"])
    except (TypeError, ValueError) as error:
        raise ValueError("Completed request decision point has an unsupported type.") from error
    return RequestDecisionPoint(
        point_type=point_type,
        local_time=value["local_time"],
        minutes_before_delivery=value["minutes_before_delivery"],
    )


def _view_from_payload(value: object) -> RequestedQuantitativeView:
    expected = {
        "kind",
        "target",
        "reference",
        "aggregation",
        "event",
        "conditions",
        "baseline",
        "change_direction",
        "difference_semantics",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("Quantitative view has an invalid structure.")
    try:
        kind = RequestViewKind(value["kind"])
    except (TypeError, ValueError) as error:
        raise ValueError("Quantitative view has an unsupported kind.") from error
    raw_event = value["event"]
    return RequestedQuantitativeView(
        kind=kind,
        target=_canonical_variable(value["target"], "view target"),
        reference=_canonical_variable(value["reference"], "view reference"),
        aggregation=_optional_enum(
            value["aggregation"], RequestAggregation, "view aggregation"
        ),
        event=(
            _predicate_from_payload(raw_event) if raw_event is not None else None
        ),
        conditions=_predicate_tuple(value["conditions"], "view conditions"),
        baseline=_predicate_tuple(value["baseline"], "view baseline"),
        change_direction=_optional_enum(
            value["change_direction"],
            RequestChangeDirection,
            "view change_direction",
        ),
        difference_semantics=_optional_enum(
            value["difference_semantics"],
            RequestDifferenceSemantics,
            "view difference_semantics",
        ),
    )


def _completed_request_from_payload(
    original_question: str,
    value: object,
) -> CompletedRequest:
    expected = {
        "original_question",
        "normalized_request",
        "resolved_variables",
        "filters",
        "time_scope",
        "delivery_scope",
        "decision_points",
        "requested_quantitative_views",
        "execution_perspective",
        "assumptions",
        "notes",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("Completed request has an invalid structure.")
    if value["original_question"] != original_question:
        raise ValueError("Completed request must preserve the original question.")
    normalized_request = value["normalized_request"]
    if not isinstance(normalized_request, str) or not normalized_request.strip():
        raise ValueError("Completed request requires a normalized_request.")

    resolved_variables = _text_tuple(
        value["resolved_variables"],
        "resolved_variables",
    )
    if len(set(resolved_variables)) != len(resolved_variables) or any(
        variable not in VARIABLE_REGISTRY for variable in resolved_variables
    ):
        raise ValueError(
            "Completed request resolved_variables must be unique canonical variables."
        )
    raw_views = value["requested_quantitative_views"]
    if not isinstance(raw_views, list):
        raise ValueError(
            "Completed request requested_quantitative_views must be an array."
        )
    return CompletedRequest(
        original_question=original_question,
        normalized_request=normalized_request.strip(),
        resolved_variables=resolved_variables,
        filters=_predicate_tuple(value["filters"], "filters"),
        time_scope=_time_scope_from_payload(value["time_scope"]),
        delivery_scope=_delivery_scope_from_payload(value["delivery_scope"]),
        decision_points=tuple(
            _decision_point_from_payload(item) for item in value["decision_points"]
        ),
        requested_quantitative_views=tuple(
            _view_from_payload(item) for item in raw_views
        ),
        execution_perspective=_optional_text(
            value["execution_perspective"],
            "execution_perspective",
        ),
        assumptions=_text_tuple(value["assumptions"], "assumptions"),
        notes=_text_tuple(value["notes"], "notes"),
    )


def _state_from_payload(
    original_question: str,
    history: tuple[ScopeClarificationTurn, ...],
    payload: object,
) -> RequestUnderstandingState:
    if not isinstance(payload, dict):
        raise ValueError("Scope checker output must be a JSON object.")
    is_complete = payload.get("is_complete")
    raw_requests = payload.get("clarification_requests")
    raw_completed_request = payload.get("completed_request")
    if (
        not isinstance(is_complete, bool)
        or not isinstance(raw_requests, list)
        or "completed_request" not in payload
    ):
        raise ValueError("Scope checker output has an invalid structure.")

    requests: list[ScopeClarificationRequest] = []
    for item in raw_requests:
        if not isinstance(item, dict):
            raise ValueError("Scope clarification requests must be objects.")
        field = item.get("field")
        question = item.get("question")
        options = item.get("options")
        if (
            not isinstance(field, str)
            or not field.strip()
            or not isinstance(question, str)
            or not question.strip()
            or not isinstance(options, list)
            or any(not isinstance(option, str) for option in options)
        ):
            raise ValueError("Scope clarification request has an invalid structure.")
        requests.append(
            ScopeClarificationRequest(
                field=field.strip(),
                question=question.strip(),
                options=tuple(options),
            )
        )

    if is_complete and requests:
        raise ValueError("A complete scope cannot contain clarification requests.")
    if not is_complete and not requests:
        raise ValueError("An incomplete scope requires a clarification request.")
    if is_complete and raw_completed_request is None:
        raise ValueError("A complete scope requires a completed request.")
    if not is_complete and raw_completed_request is not None:
        raise ValueError("An incomplete scope cannot contain a completed request.")
    completed_request = (
        _completed_request_from_payload(original_question, raw_completed_request)
        if is_complete
        else None
    )
    return RequestUnderstandingState(
        original_question=original_question,
        pending_requests=tuple(requests),
        clarification_history=history,
        is_complete=is_complete,
        completed_request=completed_request,
    )


def _check_scope(
    original_question: str,
    history: tuple[ScopeClarificationTurn, ...],
) -> RequestUnderstandingState:
    """Run the constrained semantic scope check, failing closed if unavailable."""
    if not os.getenv("OPENAI_API_KEY"):
        return _unresolved_fallback(original_question, history)
    try:
        from openai import OpenAI

        client = OpenAI()
        response = client.responses.create(
            model=os.getenv("OPENAI_COMPILER_MODEL", DEFAULT_SCOPE_MODEL),
            input=[
                {"role": "system", "content": _scope_checker_instructions()},
                {
                    "role": "user",
                    "content": _build_scope_context(original_question, history),
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "scope_resolution",
                    "strict": True,
                    "schema": _scope_json_schema(),
                }
            },
        )
        if not response.output_text:
            raise ValueError("The semantic scope checker returned no output text.")
        return _state_from_payload(
            original_question,
            history,
            json.loads(response.output_text),
        )
    except Exception:
        return _unresolved_fallback(original_question, history)


def start_request_understanding(question: str) -> RequestUnderstandingState:
    """Start semantic scope resolution for a new user request."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Scope resolution requires a non-empty question.")
    return _check_scope(question.strip(), ())


def answer_request_clarification(
    state: RequestUnderstandingState,
    answer_text: str,
) -> RequestUnderstandingState:
    """Record one answer and re-evaluate which material scope gaps remain."""
    if state.is_complete:
        raise ValueError("Scope is already complete.")
    if not state.pending_requests:
        raise ValueError("Cannot answer scope clarification when none is pending.")
    if not isinstance(answer_text, str) or not answer_text.strip():
        raise ValueError("Scope clarification answer must be a non-empty string.")

    history = (
        *state.clarification_history,
        ScopeClarificationTurn(
            requests=state.pending_requests,
            answer_text=answer_text,
        ),
    )
    return _check_scope(state.original_question, history)
