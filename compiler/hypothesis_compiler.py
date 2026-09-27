"""Hybrid deterministic/LLM compiler for research specifications."""

import json
import os
import re
from dataclasses import is_dataclass, replace
from enum import Enum
from typing import Any

from compiler.variable_registry import DOMAIN_TERMS, VARIABLE_REGISTRY
from domain.research_language import (
    Aggregation,
    ChangeDirection,
    DifferenceSemantics,
    OutputType,
)
from schemas.hypothesis_schema import (
    ClarificationOption,
    ClarificationRequest,
    Condition,
    DecisionPoint,
    DecisionPointType,
    DeliveryWindow,
    RelativePeriodUnit,
    ResearchSpecification,
    SpecificationStatus,
    StudyPeriod,
    StudyPeriodType,
)


DEFAULT_COMPILER_MODEL = "gpt-5.6-luna"


def _normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _contains_phrase(text: str, phrase: str) -> bool:
    pattern = rf"(?<!\w){re.escape(phrase)}(?!\w)"
    return re.search(pattern, text) is not None


def _parse_delivery_window(text: str) -> DeliveryWindow | None:
    if any(
        _contains_phrase(text, phrase)
        for phrase in (
            "evening delivery products",
            "evening delivery",
            "evening products",
        )
    ):
        return DeliveryWindow(start_local="17:00", end_local="21:00")

    pattern = (
        r"(?:(\d{4}-\d{2}-\d{2})\s+)?"
        r"(\d{1,2}:\d{2})\s*(?:-|–|to)\s*(\d{1,2}:\d{2})"
    )
    match = re.search(pattern, text)
    if match is None:
        return None
    nearby_context = text[
        max(0, match.start() - 32):min(len(text), match.end() + 32)
    ]
    if not any(
        _contains_phrase(nearby_context, phrase)
        for phrase in ("product", "delivery", "delivery window", "delivery interval")
    ):
        return None
    return DeliveryWindow(
        start_local=match.group(2),
        end_local=match.group(3),
        date=match.group(1),
    )


def _parse_decision_points(text: str) -> tuple[DecisionPoint, ...]:
    located: list[tuple[int, DecisionPoint]] = []
    absolute_patterns = (
        r"\bat\s+(\d{1,2}:\d{2})\s*,?\s*(?:predict|forecast|trade|decide)\b",
        r"\b(?:predict|forecast|decision|trade)\s+(?:at\s+)(\d{1,2}:\d{2})\b",
        r"\bas\s+of\s+(\d{1,2}:\d{2})\b",
    )
    for pattern in absolute_patterns:
        for match in re.finditer(pattern, text):
            located.append((
                match.start(),
                DecisionPoint(
                    type=DecisionPointType.ABSOLUTE_LOCAL_TIME,
                    local_time=match.group(1),
                ),
            ))

    relative_pattern = (
        r"\b(\d+)\s*(hours?|hrs?|h|minutes?|mins?|min)\s+before\s+delivery\b"
    )
    for match in re.finditer(relative_pattern, text):
        amount = int(match.group(1))
        unit = match.group(2)
        minutes = amount * 60 if unit.startswith(("h", "hour")) else amount
        located.append((
            match.start(),
            DecisionPoint(
                type=DecisionPointType.RELATIVE_TO_DELIVERY,
                minutes_before_delivery=minutes,
            ),
        ))

    horizon_mentions: list[tuple[int, str]] = []
    horizon_patterns = (
        r"\b(?:trade|buy)\s+around\s+(?:the\s+)?(id[13])\b",
        r"\baround\s+(?:the\s+)?(id[13])\s+horizon\b",
        r"\b(id[13])\s+(?:timing|trading\s+horizon)\b",
    )
    for pattern in horizon_patterns:
        horizon_mentions.extend(
            (match.start(1), match.group(1)) for match in re.finditer(pattern, text)
        )
    paired_horizons = re.finditer(
        r"(?:around\s+(?:the\s+)?)?(id[13])\s+horizon\s+"
        r"(?:or|and|versus|vs\.?)\s+"
        r"(?:around\s+)?(?:the\s+)?(id[13])\s+horizon",
        text,
    )
    for match in paired_horizons:
        horizon_mentions.extend(
            (match.start(index), match.group(index)) for index in (1, 2)
        )
    grouped_timing = re.search(
        r"\b(id[13])\b\s+(?:or|and|versus|vs\.?)\s+"
        r"\b(id[13])\b\s+(?:timing|trading\s+horizons?)\b",
        text,
    )
    if grouped_timing:
        horizon_mentions.extend((grouped_timing.start(index), grouped_timing.group(index)) for index in (1, 2))

    seen_horizons: set[tuple[int, str]] = set()
    for position, term in horizon_mentions:
        if (position, term) in seen_horizons:
            continue
        seen_horizons.add((position, term))
        minutes = 60 if term == "id1" else 180
        located.append((
            position,
            DecisionPoint(
                type=DecisionPointType.RELATIVE_TO_DELIVERY,
                minutes_before_delivery=minutes,
                label=f"around {term.upper()} horizon",
            ),
        ))

    located.sort(key=lambda item: item[0])
    points: list[DecisionPoint] = []
    for _, point in located:
        if point not in points:
            points.append(point)
    return tuple(points)


def _variable_phrases() -> list[tuple[str, str]]:
    phrases: set[tuple[str, str]] = set()
    for canonical, metadata in VARIABLE_REGISTRY.items():
        phrases.add((canonical, canonical))
        phrases.add((canonical.replace("_", " "), canonical))
        phrases.update((alias.casefold(), canonical) for alias in metadata.aliases)
    return sorted(phrases, key=lambda item: len(item[0]), reverse=True)


def _resolve_variables(text: str) -> tuple[str, ...]:
    occupied: list[tuple[int, int]] = []
    resolved: list[str] = []
    for phrase, canonical in _variable_phrases():
        pattern = rf"(?<!\w){re.escape(phrase)}(?!\w)"
        for match in re.finditer(pattern, text):
            span = match.span()
            if any(span[0] < end and start < span[1] for start, end in occupied):
                continue
            occupied.append(span)
            if canonical not in resolved:
                resolved.append(canonical)
    return tuple(resolved)


def _parse_study_period(text: str) -> StudyPeriod | None:
    """Parse explicit historical sample periods without resolving calendar dates."""
    date = r"(\d{4}-\d{2}-\d{2})"
    range_patterns = (
        rf"\bfrom\s+{date}\s+to\s+{date}\b",
        rf"\bbetween\s+{date}\s+and\s+{date}\b",
        rf"\b{date}\s+to\s+{date}\b",
        rf"\b{date}\s+-\s+{date}\b",
    )
    for pattern in range_patterns:
        match = re.search(pattern, text)
        if match:
            return StudyPeriod(
                type=StudyPeriodType.ABSOLUTE_DATE_RANGE,
                start_date=match.group(1),
                end_date=match.group(2),
            )

    lookback = re.search(
        r"\b(?:last|past|previous)\s+(\d+)\s+"
        r"(days?|weeks?|months?|years?)\b",
        text,
    )
    if lookback:
        units = {
            "day": RelativePeriodUnit.DAYS,
            "week": RelativePeriodUnit.WEEKS,
            "month": RelativePeriodUnit.MONTHS,
            "year": RelativePeriodUnit.YEARS,
        }
        unit = lookback.group(2).removesuffix("s")
        return StudyPeriod(
            type=StudyPeriodType.RELATIVE_LOOKBACK,
            lookback_value=int(lookback.group(1)),
            lookback_unit=units[unit],
        )

    if any(
        _contains_phrase(text, phrase)
        for phrase in (
            "all available history",
            "all available data",
            "full available history",
            "entire available history",
        )
    ):
        return StudyPeriod(type=StudyPeriodType.ALL_AVAILABLE)
    return None


def _resolve_aggregation(text: str) -> Aggregation | None:
    phrases = (
        (("average", "mean"), Aggregation.MEAN),
        (("median",), Aggregation.MEDIAN),
        (("minimum", "min"), Aggregation.MIN),
        (("maximum", "max"), Aggregation.MAX),
        (("exact",), Aggregation.EXACT),
    )
    for aliases, aggregation in phrases:
        if any(_contains_phrase(text, alias) for alias in aliases):
            return aggregation
    return None


def _id1_id3_event(text: str) -> Condition | None:
    if re.search(
        r"\bid1\b(?:\s+\w+){0,5}\s+(?:above|exceed|exceeds)\s+\bid3\b",
        text,
    ):
        return Condition("id1_minus_id3", ">", 0)
    relations = (
        ((">", "above", "higher than", "exceed", "exceeds"), ">"),
        (("<", "below", "lower than"), "<"),
        (("==", "=", "equal to", "equals"), "="),
    )
    for phrases, operator in relations:
        for phrase in phrases:
            escaped = re.escape(phrase)
            if re.search(rf"\bid1\b\s*(?:is\s+)?{escaped}\s*\bid3\b", text):
                return Condition("id1_minus_id3", operator, 0)
    return None


def _polarity_conditions(text: str) -> tuple[Condition, ...]:
    conditions: list[Condition] = []
    directional_wind = _directional_wind_revision_condition(text)
    if directional_wind is not None:
        conditions.append(directional_wind)
    for phrase, canonical in _variable_phrases():
        escaped = re.escape(phrase)
        negative = (
            rf"(?<!\w){escaped}(?!\w)\s+is\s+negative",
            rf"(?<!\w)negative\s+{escaped}(?!\w)",
            rf"(?<!\w)downward\s+{escaped}(?!\w)",
        )
        positive = (
            rf"(?<!\w){escaped}(?!\w)\s+is\s+positive",
            rf"(?<!\w)positive\s+{escaped}(?!\w)",
        )
        condition = None
        if any(re.search(pattern, text) for pattern in negative):
            condition = Condition(canonical, "<", 0)
        elif any(re.search(pattern, text) for pattern in positive):
            condition = Condition(canonical, ">", 0)
        if condition is not None and condition not in conditions:
            conditions.append(condition)
    return tuple(conditions)


def _directional_wind_revision_condition(text: str) -> Condition | None:
    """Resolve explicit directional wind-revision language for the MVP."""
    match = re.search(
        r"\b(?:downward|negative)\s+"
        r"(?:(onshore|offshore|total)\s+)?"
        r"wind(?:\s+forecast)?\s+revisions?\b",
        text,
    )
    if match is None:
        return None
    variable = {
        "onshore": "wind_onshore_revision",
        "offshore": "wind_offshore_revision",
        "total": "total_wind_revision",
        None: "total_wind_revision",
    }[match.group(1)]
    return Condition(variable=variable, operator="<", value=0)


def _explicit_comparisons(text: str) -> tuple[Condition, ...]:
    conditions: list[Condition] = []
    value_pattern = r"(?:-?\d+(?:\.\d+)?|true|false|[a-z][\w-]*)"
    for phrase, canonical in _variable_phrases():
        pattern = rf"(?<!\w){re.escape(phrase)}(?!\w)\s*(>=|<=|==|=|>|<)\s*({value_pattern})"
        for match in re.finditer(pattern, text):
            raw_value = match.group(2)
            right_is_domain_term = raw_value in DOMAIN_TERMS or any(
                raw_value in term.aliases for term in DOMAIN_TERMS.values()
            )
            if _resolve_variables(raw_value) or right_is_domain_term:
                continue
            if re.fullmatch(r"-?\d+(?:\.\d+)?", raw_value):
                value: int | float | str | bool = (
                    float(raw_value) if "." in raw_value else int(raw_value)
                )
            elif raw_value in {"true", "false"}:
                value = raw_value == "true"
            else:
                value = raw_value
            condition = Condition(canonical, match.group(1), value)
            if condition not in conditions:
                conditions.append(condition)
    return tuple(conditions)


def _delivery_hour_filter(text: str) -> Condition | None:
    if not re.search(r"\bdelivery\s+(?:time|hour|hours)\b", text):
        return None
    patterns = (
        r"between\s+(\d{1,2})(?::00)?\s+and\s+(\d{1,2})(?::00)?",
        r"(\d{1,2}):00\s*[-–]\s*(\d{1,2}):00",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return Condition(
                variable="delivery_hour",
                operator="between",
                value=(int(match.group(1)), int(match.group(2))),
            )
    return None


def _generic_wind_revision_is_ambiguous(text: str) -> bool:
    generic_wind = (
        _contains_phrase(text, "wind revision")
        or _contains_phrase(text, "wind forecast revision")
        or _contains_phrase(text, "wind revisions")
        or _contains_phrase(text, "wind forecast revisions")
    )
    specific_wind = any(
        _contains_phrase(text, phrase)
        for phrase in ("onshore wind", "offshore wind", "total wind")
    )
    directional_wind = _directional_wind_revision_condition(text)
    return generic_wind and not specific_wind and directional_wind is None


def _wind_clarification() -> ClarificationRequest:
    return ClarificationRequest(
        field="conditions.variable",
        question="Which wind revision do you mean?",
        options=(
            ClarificationOption(
                value="total_wind_revision",
                label="Total wind revision",
            ),
            ClarificationOption(
                value="wind_onshore_revision",
                label="Onshore wind revision",
            ),
            ClarificationOption(
                value="wind_offshore_revision",
                label="Offshore wind revision",
            ),
        ),
        allow_free_text=True,
    )


def _study_period_clarification() -> ClarificationRequest:
    return ClarificationRequest(
        field="study_period",
        question="Which historical period should be analyzed?",
        options=(
            ClarificationOption(value="last_1_month", label="Last 1 month"),
            ClarificationOption(value="last_3_months", label="Last 3 months"),
            ClarificationOption(value="last_1_year", label="Last 1 year"),
            ClarificationOption(
                value="all_available",
                label="All available history",
            ),
            ClarificationOption(value="custom", label="Custom period"),
        ),
        allow_free_text=True,
    )


def _single_historical_point_lookup(text: str) -> bool:
    dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)
    if len(dates) != 1:
        return False
    sample_language = (
        "average",
        "median",
        "probability",
        "how often",
        "more likely",
        "relationship",
        "related to",
        "evidence",
        "when ",
    )
    return not any(phrase in text for phrase in sample_language)


def _research_needs_study_period(
    text: str,
) -> bool:
    return not _single_historical_point_lookup(text)


def _merge_clarification_requests(
    *groups: tuple[ClarificationRequest, ...],
) -> tuple[ClarificationRequest, ...]:
    merged: list[ClarificationRequest] = []
    seen: set[tuple[str, str]] = set()
    for group in groups:
        for request in group:
            key = (request.field, request.question)
            if key not in seen:
                seen.add(key)
                merged.append(request)
    return tuple(merged)


def _apply_research_context(
    specification: ResearchSpecification,
    text: str,
    study_period: StudyPeriod | None,
) -> ResearchSpecification:
    """Preserve parsed context and add deterministic research clarifications."""
    ambiguities = list(specification.ambiguities)
    deterministic_requests: list[ClarificationRequest] = []

    if _generic_wind_revision_is_ambiguous(text):
        diagnostic = "Specify whether wind revision means onshore, offshore, or total wind."
        if diagnostic not in ambiguities:
            ambiguities.append(diagnostic)
        deterministic_requests.append(_wind_clarification())

    effective_study_period = specification.study_period or study_period
    if (
        _research_needs_study_period(text)
        and effective_study_period is None
    ):
        diagnostic = "study period is unspecified"
        if diagnostic not in ambiguities:
            ambiguities.append(diagnostic)
        deterministic_requests.append(_study_period_clarification())

    status = specification.status
    if ambiguities and status != SpecificationStatus.UNSUPPORTED:
        status = SpecificationStatus.NEEDS_CLARIFICATION

    return replace(
        specification,
        status=status,
        study_period=effective_study_period,
        ambiguities=tuple(ambiguities),
        clarification_requests=_merge_clarification_requests(
            specification.clarification_requests,
            tuple(deterministic_requests),
        ),
    )


def _known_ambiguity(
    original: str,
    text: str,
    delivery_window: DeliveryWindow | None,
    decision_points: tuple[DecisionPoint, ...],
) -> ResearchSpecification | None:
    if _generic_wind_revision_is_ambiguous(text):
        event = _id1_id3_event(text)
        probability_requested = any(
            _contains_phrase(text, phrase)
            for phrase in ("probability", "chance", "how often", "more likely")
        )
        return ResearchSpecification(
            original_hypothesis=original,
            status=SpecificationStatus.NEEDS_CLARIFICATION,
            output_type=(
                OutputType.PROBABILITY
                if probability_requested and event is not None
                else None
            ),
            event=event,
            ambiguities=(
                "Specify whether wind revision means onshore, offshore, or total wind.",
            ),
            clarification_requests=(_wind_clarification(),),
            delivery_window=delivery_window,
            decision_points=decision_points,
        )
    return None


def _known_unsupported(
    original: str,
    text: str,
    delivery_window: DeliveryWindow | None,
    decision_points: tuple[DecisionPoint, ...],
) -> ResearchSpecification | None:
    unsupported = tuple(
        term
        for term in ("gas price", "system load", "outages", "order book")
        if _contains_phrase(text, term)
    )
    if not unsupported:
        return None
    return ResearchSpecification(
        original_hypothesis=original,
        status=SpecificationStatus.UNSUPPORTED,
        unsupported_terms=unsupported,
        delivery_window=delivery_window,
        decision_points=decision_points,
    )


def _try_deterministic_parse(
    original: str,
    text: str,
    delivery_window: DeliveryWindow | None,
    decision_points: tuple[DecisionPoint, ...],
) -> ResearchSpecification | None:
    event = _id1_id3_event(text)
    probability_requested = any(
        _contains_phrase(text, phrase)
        for phrase in ("probability", "chance", "how often", "more likely")
    )
    conditions = list(_polarity_conditions(text))
    conditions.extend(
        condition
        for condition in _explicit_comparisons(text)
        if condition not in conditions and condition != event
    )
    delivery_filter = _delivery_hour_filter(text)
    filters = (delivery_filter,) if delivery_filter else ()

    if probability_requested and event is not None:
        return ResearchSpecification(
            original_hypothesis=original,
            status=SpecificationStatus.VALID,
            output_type=OutputType.PROBABILITY,
            event=event,
            conditions=tuple(conditions),
            filters=filters,
            delivery_window=delivery_window,
            decision_points=decision_points,
        )

    aggregation = _resolve_aggregation(text)
    value_requested = aggregation is not None or any(
        _contains_phrase(text, phrase)
        for phrase in ("value", "how much", "what is")
    )
    resolved = _resolve_variables(text)
    if value_requested and len(resolved) == 1:
        return ResearchSpecification(
            original_hypothesis=original,
            status=SpecificationStatus.VALID,
            output_type=OutputType.VALUE,
            target=resolved[0],
            conditions=tuple(conditions),
            filters=filters,
            aggregation=aggregation,
            delivery_window=delivery_window,
            decision_points=decision_points,
        )
    return None


def _condition_schema() -> dict[str, Any]:
    scalar = {"anyOf": [
        {"type": "number"},
        {"type": "string"},
        {"type": "boolean"},
    ]}
    return {
        "type": "object",
        "properties": {
            "variable": {"type": "string"},
            "operator": {"type": "string"},
            "value": {
                "anyOf": [
                    {"type": "number"},
                    {"type": "string"},
                    {"type": "boolean"},
                    {"type": "array", "items": scalar},
                ]
            },
        },
        "required": ["variable", "operator", "value"],
        "additionalProperties": False,
    }


def _build_llm_json_schema() -> dict[str, Any]:
    nullable_string = {"type": ["string", "null"]}
    condition = _condition_schema()
    condition_array = {"type": "array", "items": condition}
    string_array = {"type": "array", "items": {"type": "string"}}
    delivery_window = {
        "type": "object",
        "properties": {
            "start_local": nullable_string,
            "end_local": nullable_string,
            "date": nullable_string,
        },
        "required": ["start_local", "end_local", "date"],
        "additionalProperties": False,
    }
    decision_point = {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": ["absolute_local_time", "relative_to_delivery"],
            },
            "local_time": nullable_string,
            "minutes_before_delivery": {"type": ["integer", "null"]},
            "label": nullable_string,
        },
        "required": [
            "type",
            "local_time",
            "minutes_before_delivery",
            "label",
        ],
        "additionalProperties": False,
    }
    study_period = {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": [
                    "absolute_date_range",
                    "relative_lookback",
                    "all_available",
                ],
            },
            "start_date": nullable_string,
            "end_date": nullable_string,
            "lookback_value": {"type": ["integer", "null"]},
            "lookback_unit": {
                "enum": ["days", "weeks", "months", "years", None],
            },
        },
        "required": [
            "type",
            "start_date",
            "end_date",
            "lookback_value",
            "lookback_unit",
        ],
        "additionalProperties": False,
    }
    clarification_option = {
        "type": "object",
        "properties": {
            "value": {"type": "string"},
            "label": {"type": "string"},
        },
        "required": ["value", "label"],
        "additionalProperties": False,
    }
    clarification_request = {
        "type": "object",
        "properties": {
            "field": {"type": "string"},
            "question": {"type": "string"},
            "options": {"type": "array", "items": clarification_option},
            "allow_free_text": {"type": "boolean"},
        },
        "required": ["field", "question", "options", "allow_free_text"],
        "additionalProperties": False,
    }
    properties: dict[str, Any] = {
        "status": {
            "type": "string",
            "enum": ["valid", "needs_clarification", "unsupported"],
        },
        "output_type": {
            "enum": ["value", "change", "probability", "difference", None],
        },
        "target": nullable_string,
        "reference": nullable_string,
        "event": {"anyOf": [condition, {"type": "null"}]},
        "conditions": condition_array,
        "filters": condition_array,
        "baseline": condition_array,
        "aggregation": {"enum": ["mean", "median", "min", "max", "exact", None]},
        "change_direction": {"enum": ["increase", "decrease", "unchanged", None]},
        "difference_semantics": {"enum": ["generic", "revision", "deviation", None]},
        "ambiguities": string_array,
        "unsupported_terms": string_array,
        "compiler_notes": string_array,
        "delivery_window": {
            "anyOf": [delivery_window, {"type": "null"}],
        },
        "decision_points": {"type": "array", "items": decision_point},
        "study_period": {"anyOf": [study_period, {"type": "null"}]},
        "clarification_requests": {
            "type": "array",
            "items": clarification_request,
        },
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _build_registry_context() -> str:
    variables = []
    for name, metadata in VARIABLE_REGISTRY.items():
        aliases = ", ".join(metadata.aliases)
        variables.append(
            f"- {name}: {metadata.description} "
            f"[category={metadata.category}; unit={metadata.unit}; aliases={aliases}]"
        )
    terms = [
        f"- {term.name}: {term.meaning}"
        for term in DOMAIN_TERMS.values()
    ]
    return "DOMAIN TERMS\n" + "\n".join(terms) + "\n\nVARIABLES\n" + "\n".join(variables)


def _build_llm_instructions() -> str:
    return f"""You translate user questions into the constrained Power Market Quant
Research Specification Language. Return only the requested JSON structure.

This compiler is already inside the Research pipeline. Do not classify or route the request.
Compile historical evidence, descriptive statistics, conditional
probabilities, relationships, associations, hypotheses, and historical
predictive-value questions as Research specifications.

Allowed output types: value, change, probability, difference.
VALUE is a numerical target value. CHANGE is the direction of a target relative
to a reference. PROBABILITY requires a true/false event. DIFFERENCE is target
minus reference. Difference semantics are generic, revision, or deviation.
Revision means A40 - A01. Deviation means actual - forecast.
Conditions are research signals; filters are sample restrictions; baseline holds
comparison conditions. ID1 above/below ID3 maps to id1_minus_id3 >/< 0.
A40 is the delivery-day 08:00 forecast; A01 is the day-ahead forecast. Actual
generation is ex-post. Do not assess point-in-time validity.

A delivery_window is the product's market-local delivery interval. A
decision_point is when a prediction or trading decision is made, either at an
absolute local time such as 08:00 or a number of minutes before delivery.
Delivery time and decision time are different. Never infer information_cutoff,
assess time legality, or invent a missing date or time. In explicit trading-
timing context only, "around ID1 horizon" may mean approximately 60 minutes
before delivery and "around ID3 horizon" may mean approximately 180 minutes
before delivery. This is only a decision-horizon interpretation: never redefine
the ID1 or ID3 volume-weighted price indices as point-in-time prices.

A study_period is the historical sample date range; it is separate from the
electricity product's delivery_window. Preserve explicit absolute ISO date
ranges, calendar-relative lookbacks, and requests for all available history.
Never assume a missing historical period. When a sample period materially
affects a RESEARCH answer and none is supplied, leave study_period null, return
needs_clarification, and add a clarification request for study_period.

Use only the closed world below. Never invent variables, methods, or concepts.
If generic wind revision is unresolved among onshore, offshore, and total wind,
return needs_clarification. If a required concept is outside the registry, return
unsupported and list it. Do not guess missing meaning.

{_build_registry_context()}"""


def _condition_from_dict(data: dict[str, Any]) -> Condition:
    value = data["value"]
    if isinstance(value, list):
        value = tuple(value)
    return Condition(
        variable=str(data["variable"]),
        operator=str(data["operator"]),
        value=value,
    )


def _draft_to_specification(
    original: str,
    draft: dict[str, Any],
) -> ResearchSpecification:
    event = _condition_from_dict(draft["event"]) if draft["event"] else None
    conditions = tuple(_condition_from_dict(item) for item in draft["conditions"])
    filters = tuple(_condition_from_dict(item) for item in draft["filters"])
    baseline = tuple(_condition_from_dict(item) for item in draft["baseline"])
    delivery_data = draft["delivery_window"]
    delivery_window = (
        DeliveryWindow(
            start_local=delivery_data["start_local"],
            end_local=delivery_data["end_local"],
            date=delivery_data["date"],
        )
        if delivery_data is not None
        else None
    )
    decision_points = tuple(
        DecisionPoint(
            type=DecisionPointType(item["type"]),
            local_time=item["local_time"],
            minutes_before_delivery=item["minutes_before_delivery"],
            label=item["label"],
        )
        for item in draft["decision_points"]
    )
    study_data = draft["study_period"]
    study_period = (
        StudyPeriod(
            type=StudyPeriodType(study_data["type"]),
            start_date=study_data["start_date"],
            end_date=study_data["end_date"],
            lookback_value=study_data["lookback_value"],
            lookback_unit=(
                RelativePeriodUnit(study_data["lookback_unit"])
                if study_data["lookback_unit"]
                else None
            ),
        )
        if study_data is not None
        else None
    )
    clarification_requests = tuple(
        ClarificationRequest(
            field=str(item["field"]),
            question=str(item["question"]),
            options=tuple(
                ClarificationOption(
                    value=str(option["value"]),
                    label=str(option["label"]),
                )
                for option in item["options"]
            ),
            allow_free_text=bool(item["allow_free_text"]),
        )
        for item in draft["clarification_requests"]
    )

    target = draft["target"]
    reference = draft["reference"]
    unsupported_terms = list(draft["unsupported_terms"])
    variables = [target, reference]
    variables.extend(
        condition.variable
        for condition in ((event,) if event else ()) + conditions + filters + baseline
    )
    for variable in variables:
        if variable is not None and variable not in VARIABLE_REGISTRY:
            if variable not in unsupported_terms:
                unsupported_terms.append(variable)

    ambiguities = tuple(str(item) for item in draft["ambiguities"])
    if unsupported_terms:
        status = SpecificationStatus.UNSUPPORTED
    elif ambiguities:
        status = SpecificationStatus.NEEDS_CLARIFICATION
    else:
        status = SpecificationStatus(draft["status"])

    return ResearchSpecification(
        original_hypothesis=original,
        status=status,
        output_type=OutputType(draft["output_type"]) if draft["output_type"] else None,
        target=target,
        reference=reference,
        event=event,
        conditions=conditions,
        filters=filters,
        baseline=baseline,
        aggregation=Aggregation(draft["aggregation"]) if draft["aggregation"] else None,
        change_direction=(
            ChangeDirection(draft["change_direction"])
            if draft["change_direction"] else None
        ),
        difference_semantics=(
            DifferenceSemantics(draft["difference_semantics"])
            if draft["difference_semantics"] else None
        ),
        ambiguities=ambiguities,
        unsupported_terms=tuple(str(item) for item in unsupported_terms),
        compiler_notes=tuple(str(item) for item in draft["compiler_notes"]),
        delivery_window=delivery_window,
        decision_points=decision_points,
        study_period=study_period,
        clarification_requests=_merge_clarification_requests(
            clarification_requests,
        ),
    )


def _fallback_unavailable(
    original: str,
    note: str,
    delivery_window: DeliveryWindow | None,
    decision_points: tuple[DecisionPoint, ...],
) -> ResearchSpecification:
    return ResearchSpecification(
        original_hypothesis=original,
        status=SpecificationStatus.NEEDS_CLARIFICATION,
        ambiguities=("The request could not be interpreted deterministically.",),
        compiler_notes=(note,),
        delivery_window=delivery_window,
        decision_points=decision_points,
    )


def _parse_with_llm(
    original: str,
    delivery_window: DeliveryWindow | None,
    decision_points: tuple[DecisionPoint, ...],
) -> ResearchSpecification:
    if not os.getenv("OPENAI_API_KEY"):
        return _fallback_unavailable(
            original,
            "Semantic fallback is unavailable because OPENAI_API_KEY is not set.",
            delivery_window,
            decision_points,
        )
    try:
        from openai import OpenAI

        client = OpenAI()
        response = client.responses.create(
            model=os.getenv("OPENAI_COMPILER_MODEL", DEFAULT_COMPILER_MODEL),
            input=[
                {"role": "system", "content": _build_llm_instructions()},
                {"role": "user", "content": original},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "research_specification_draft",
                    "strict": True,
                    "schema": _build_llm_json_schema(),
                }
            },
        )
        if not response.output_text:
            raise ValueError("The semantic parser returned no output text.")
        draft = json.loads(response.output_text)
        return _draft_to_specification(original, draft)
    except ImportError:
        return _fallback_unavailable(
            original,
            "Semantic fallback is unavailable because the OpenAI SDK is not installed.",
            delivery_window,
            decision_points,
        )
    except Exception as error:
        return _fallback_unavailable(
            original,
            f"Semantic fallback failed: {type(error).__name__}.",
            delivery_window,
            decision_points,
        )


def _serialize_compiler_context(value: Any) -> Any:
    """Convert typed compiler values to canonical JSON-compatible values."""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            field.name: _serialize_compiler_context(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, (tuple, list)):
        return [_serialize_compiler_context(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _serialize_compiler_context(item)
            for key, item in value.items()
        }
    return value


def _parse_clarification_study_period(text: str) -> StudyPeriod | None:
    """Parse concise study-period answers and compiler option values."""
    normalized_options = {
        "last_1_month": StudyPeriod(
            type=StudyPeriodType.RELATIVE_LOOKBACK,
            lookback_value=1,
            lookback_unit=RelativePeriodUnit.MONTHS,
        ),
        "last_3_months": StudyPeriod(
            type=StudyPeriodType.RELATIVE_LOOKBACK,
            lookback_value=3,
            lookback_unit=RelativePeriodUnit.MONTHS,
        ),
        "last_1_year": StudyPeriod(
            type=StudyPeriodType.RELATIVE_LOOKBACK,
            lookback_value=1,
            lookback_unit=RelativePeriodUnit.YEARS,
        ),
        "all_available": StudyPeriod(type=StudyPeriodType.ALL_AVAILABLE),
    }
    for option, study_period in normalized_options.items():
        if _contains_phrase(text, option):
            return study_period

    singular_units = {
        "day": RelativePeriodUnit.DAYS,
        "week": RelativePeriodUnit.WEEKS,
        "month": RelativePeriodUnit.MONTHS,
        "year": RelativePeriodUnit.YEARS,
    }
    match = re.search(r"\b(?:last|past|previous)\s+(day|week|month|year)\b", text)
    if match:
        return StudyPeriod(
            type=StudyPeriodType.RELATIVE_LOOKBACK,
            lookback_value=1,
            lookback_unit=singular_units[match.group(1)],
        )
    return _parse_study_period(text)


def _resolve_wind_revision_answer(text: str) -> str | None:
    """Resolve an unambiguous generic-wind clarification answer."""
    candidates = {
        canonical
        for canonical, phrases in {
            "total_wind_revision": (
                "total wind",
                "total wind revision",
                "total_wind_revision",
            ),
            "wind_onshore_revision": (
                "onshore wind",
                "onshore wind revision",
                "wind_onshore_revision",
            ),
            "wind_offshore_revision": (
                "offshore wind",
                "offshore wind revision",
                "wind_offshore_revision",
            ),
        }.items()
        if any(_contains_phrase(text, phrase) for phrase in phrases)
    }
    return next(iter(candidates)) if len(candidates) == 1 else None


def _generic_wind_operator(original_hypothesis: str) -> str | None:
    """Recover the safely expressed direction of a generic wind revision."""
    text = _normalize_text(original_hypothesis)
    negative_patterns = (
        r"\bdownward\s+wind(?:\s+forecast)?\s+revision\b",
        r"\bnegative\s+wind(?:\s+forecast)?\s+revision\b",
        r"\bwind(?:\s+forecast)?\s+revision\s+is\s+negative\b",
        r"\bwind(?:\s+forecast)?\s+revision\s*<\s*0\b",
    )
    positive_patterns = (
        r"\bupward\s+wind(?:\s+forecast)?\s+revision\b",
        r"\bpositive\s+wind(?:\s+forecast)?\s+revision\b",
        r"\bwind(?:\s+forecast)?\s+revision\s+is\s+positive\b",
        r"\bwind(?:\s+forecast)?\s+revision\s*>\s*0\b",
    )
    if any(re.search(pattern, text) for pattern in negative_patterns):
        return "<"
    if any(re.search(pattern, text) for pattern in positive_patterns):
        return ">"
    return None


def _without_resolved_ambiguities(
    ambiguities: tuple[str, ...],
    resolved_fields: set[str],
) -> tuple[str, ...]:
    """Remove only diagnostics corresponding to deterministically resolved fields."""
    retained: list[str] = []
    for ambiguity in ambiguities:
        normalized = ambiguity.casefold()
        if "study_period" in resolved_fields and "study period" in normalized:
            continue
        if (
            "conditions.variable" in resolved_fields
            and "wind revision" in normalized
        ):
            continue
        retained.append(ambiguity)
    return tuple(retained)


def _resume_status(
    previous_status: SpecificationStatus,
    ambiguities: tuple[str, ...],
    clarification_requests: tuple[ClarificationRequest, ...],
    unsupported_terms: tuple[str, ...],
) -> SpecificationStatus:
    if unsupported_terms or previous_status == SpecificationStatus.UNSUPPORTED:
        return SpecificationStatus.UNSUPPORTED
    if ambiguities or clarification_requests:
        return SpecificationStatus.NEEDS_CLARIFICATION
    return SpecificationStatus.VALID


def _preserve_previous_resolution(
    previous: ResearchSpecification,
    candidate: ResearchSpecification,
    pending_clarifications: tuple[ClarificationRequest, ...],
) -> ResearchSpecification:
    """Allow the LLM to change only fields represented by pending requests."""
    allowed_fields = {
        request.field.split(".", maxsplit=1)[0]
        for request in pending_clarifications
        if request.field
    }
    preservable_fields = (
        "output_type",
        "target",
        "reference",
        "event",
        "conditions",
        "filters",
        "baseline",
        "aggregation",
        "change_direction",
        "difference_semantics",
        "delivery_window",
        "decision_points",
        "study_period",
    )
    updates = {
        field_name: getattr(previous, field_name)
        for field_name in preservable_fields
        if field_name not in allowed_fields
    }
    return replace(
        candidate,
        original_hypothesis=previous.original_hypothesis,
        **updates,
    )


def _copy_research_fields(
    specification: ResearchSpecification,
) -> ResearchSpecification:
    """Copy only fields owned by Research compilation and diagnostics."""
    return ResearchSpecification(
        original_hypothesis=specification.original_hypothesis,
        status=specification.status,
        output_type=specification.output_type,
        target=specification.target,
        reference=specification.reference,
        event=specification.event,
        conditions=specification.conditions,
        filters=specification.filters,
        baseline=specification.baseline,
        aggregation=specification.aggregation,
        change_direction=specification.change_direction,
        difference_semantics=specification.difference_semantics,
        ambiguities=specification.ambiguities,
        unsupported_terms=specification.unsupported_terms,
        compiler_notes=specification.compiler_notes,
        delivery_window=specification.delivery_window,
        decision_points=specification.decision_points,
        study_period=specification.study_period,
        clarification_requests=specification.clarification_requests,
    )


def _resume_fallback_unavailable(
    specification: ResearchSpecification,
    note: str,
) -> ResearchSpecification:
    notes = specification.compiler_notes
    if note not in notes:
        notes = (*notes, note)
    return replace(
        _copy_research_fields(specification),
        status=SpecificationStatus.NEEDS_CLARIFICATION,
        compiler_notes=notes,
    )


def _resume_with_llm(
    previous_specification: ResearchSpecification,
    pending_clarifications: tuple[ClarificationRequest, ...],
    answer_text: str,
) -> ResearchSpecification:
    """Ask the semantic parser to update, rather than replace, a partial draft."""
    if not os.getenv("OPENAI_API_KEY"):
        return _resume_fallback_unavailable(
            previous_specification,
            "Semantic resume is unavailable because OPENAI_API_KEY is not set.",
        )
    context = {
        "original_question": previous_specification.original_hypothesis,
        "previous_specification": _serialize_compiler_context(
            previous_specification
        ),
        "pending_clarifications": _serialize_compiler_context(
            pending_clarifications
        ),
        "latest_answer": answer_text,
    }
    resume_instructions = """
Update the previous research specification using the latest clarification answer.
The answer is not a standalone research question. Preserve fields already
resolved in previous_specification unless the answer clearly corrects them. Use
the answer primarily for the listed pending clarifications. Do not assume every
pending question was answered, do not invent missing information, and retain or
create clarification requests for anything still unresolved. Return the same
complete structured draft used by initial compilation.
"""
    try:
        from openai import OpenAI

        client = OpenAI()
        response = client.responses.create(
            model=os.getenv("OPENAI_COMPILER_MODEL", DEFAULT_COMPILER_MODEL),
            input=[
                {
                    "role": "system",
                    "content": _build_llm_instructions() + resume_instructions,
                },
                {
                    "role": "user",
                    "content": json.dumps(context, ensure_ascii=False),
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "research_specification_resume_draft",
                    "strict": True,
                    "schema": _build_llm_json_schema(),
                }
            },
        )
        if not response.output_text:
            raise ValueError("The semantic resume parser returned no output text.")
        candidate = _draft_to_specification(
            previous_specification.original_hypothesis,
            json.loads(response.output_text),
        )
        return _preserve_previous_resolution(
            previous_specification,
            candidate,
            pending_clarifications,
        )
    except ImportError:
        return _resume_fallback_unavailable(
            previous_specification,
            "Semantic resume is unavailable because the OpenAI SDK is not installed.",
        )
    except Exception as error:
        return _resume_fallback_unavailable(
            previous_specification,
            f"Semantic resume failed: {type(error).__name__}.",
        )


def resume_hypothesis_compilation(
    previous_specification: ResearchSpecification,
    pending_clarifications: tuple[ClarificationRequest, ...],
    answer_text: str,
) -> ResearchSpecification:
    """Resume compilation using an answer to the currently pending questions."""
    if not isinstance(answer_text, str) or not answer_text.strip():
        raise ValueError("Clarification answer must be a non-empty string.")
    if not pending_clarifications:
        raise ValueError(
            "Cannot resume compilation when no clarification is pending."
        )

    normalized_answer = _normalize_text(answer_text)
    resolved_fields: set[str] = set()
    unresolved: list[ClarificationRequest] = []
    conditions = list(previous_specification.conditions)
    study_period = previous_specification.study_period
    requires_generic_fallback = False

    for request in pending_clarifications:
        if request.field == "study_period":
            parsed_period = _parse_clarification_study_period(normalized_answer)
            if parsed_period is None:
                unresolved.append(request)
            else:
                study_period = parsed_period
                resolved_fields.add(request.field)
            continue

        if request.field == "conditions.variable" and (
            _generic_wind_revision_is_ambiguous(
                _normalize_text(previous_specification.original_hypothesis)
            )
            or any("wind revision" in item.casefold() for item in previous_specification.ambiguities)
        ):
            variable = _resolve_wind_revision_answer(normalized_answer)
            operator = _generic_wind_operator(
                previous_specification.original_hypothesis
            )
            if variable is None or operator is None:
                unresolved.append(request)
            else:
                condition = Condition(variable=variable, operator=operator, value=0)
                if condition not in conditions:
                    conditions.append(condition)
                resolved_fields.add(request.field)
            continue

        unresolved.append(request)
        requires_generic_fallback = True

    ambiguities = _without_resolved_ambiguities(
        previous_specification.ambiguities,
        resolved_fields,
    )
    pending = tuple(unresolved)
    merged = replace(
        _copy_research_fields(previous_specification),
        status=_resume_status(
            previous_specification.status,
            ambiguities,
            pending,
            previous_specification.unsupported_terms,
        ),
        conditions=tuple(conditions),
        study_period=study_period,
        ambiguities=ambiguities,
        clarification_requests=pending,
    )

    if requires_generic_fallback or not resolved_fields:
        return _resume_with_llm(merged, pending, answer_text)
    return merged


def compile_hypothesis(hypothesis: str) -> ResearchSpecification:
    """Compile natural language into the constrained research specification."""
    original = hypothesis.strip()
    if not original:
        return ResearchSpecification(
            original_hypothesis=original,
            status=SpecificationStatus.NEEDS_CLARIFICATION,
            ambiguities=("A research question is required.",),
        )

    normalized = _normalize_text(original)
    delivery_window = _parse_delivery_window(normalized)
    decision_points = _parse_decision_points(normalized)
    study_period = _parse_study_period(normalized)
    known_ambiguity = _known_ambiguity(
        original,
        normalized,
        delivery_window,
        decision_points,
    )
    if known_ambiguity is not None:
        return _apply_research_context(
            known_ambiguity,
            normalized,
            study_period,
        )
    known_unsupported = _known_unsupported(
        original,
        normalized,
        delivery_window,
        decision_points,
    )
    if known_unsupported is not None:
        return _apply_research_context(
            known_unsupported,
            normalized,
            study_period,
        )
    deterministic = _try_deterministic_parse(
        original,
        normalized,
        delivery_window,
        decision_points,
    )
    if deterministic is not None:
        return _apply_research_context(
            deterministic,
            normalized,
            study_period,
        )
    semantic = _parse_with_llm(original, delivery_window, decision_points)
    semantic = replace(
        semantic,
        delivery_window=semantic.delivery_window or delivery_window,
        decision_points=semantic.decision_points or decision_points,
    )
    return _apply_research_context(
        semantic,
        normalized,
        study_period,
    )
