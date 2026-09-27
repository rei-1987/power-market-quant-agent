"""Deterministic validation for compiled research specifications."""

from datetime import date
import re

from compiler.variable_registry import (
    DOMAIN_TERMS,
    VARIABLE_REGISTRY,
    VariableRole,
)
from domain.research_language import OUTPUT_SEMANTICS, OutputType
from schemas.hypothesis_schema import (
    ClarificationOption,
    ClarificationRequest,
    Condition,
    DecisionPointType,
    DeliveryWindow,
    RelativePeriodUnit,
    ResearchSpecification,
    SpecificationStatus,
    StudyPeriod,
    StudyPeriodType,
    ValidationResult,
)


ALLOWED_CONDITION_OPERATORS = {">", "<", ">=", "<=", "=", "==", "between"}
LOCAL_TIME_PATTERN = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")
ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _valid_local_time(value: str | None) -> bool:
    return value is not None and LOCAL_TIME_PATTERN.fullmatch(value) is not None


def _valid_iso_date(value: str | None) -> bool:
    if value is None or ISO_DATE_PATTERN.fullmatch(value) is None:
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _validate_study_period(
    study_period: StudyPeriod | None,
    errors: list[str],
) -> None:
    """Validate study-period structure without resolving relative dates."""
    if study_period is None:
        return

    if study_period.type == StudyPeriodType.ABSOLUTE_DATE_RANGE:
        start_valid = _valid_iso_date(study_period.start_date)
        end_valid = _valid_iso_date(study_period.end_date)
        if not start_valid:
            errors.append(
                "Absolute study period requires a valid start_date in YYYY-MM-DD "
                "format."
            )
        if not end_valid:
            errors.append(
                "Absolute study period requires a valid end_date in YYYY-MM-DD "
                "format."
            )
        if (
            start_valid
            and end_valid
            and date.fromisoformat(study_period.start_date)  # type: ignore[arg-type]
            > date.fromisoformat(study_period.end_date)  # type: ignore[arg-type]
        ):
            errors.append("Study period start_date must not be after end_date.")
        if study_period.lookback_value is not None:
            errors.append(
                "Absolute study period must not set lookback_value."
            )
        if study_period.lookback_unit is not None:
            errors.append("Absolute study period must not set lookback_unit.")
        return

    if study_period.type == StudyPeriodType.RELATIVE_LOOKBACK:
        if (
            not isinstance(study_period.lookback_value, int)
            or isinstance(study_period.lookback_value, bool)
            or study_period.lookback_value <= 0
        ):
            errors.append(
                "Relative study period requires a positive integer lookback_value."
            )
        if not isinstance(study_period.lookback_unit, RelativePeriodUnit):
            errors.append("Relative study period requires a lookback_unit.")
        if study_period.start_date is not None:
            errors.append("Relative study period must not set start_date.")
        if study_period.end_date is not None:
            errors.append("Relative study period must not set end_date.")
        return

    if study_period.type == StudyPeriodType.ALL_AVAILABLE:
        if study_period.start_date is not None:
            errors.append("ALL_AVAILABLE study period must not set start_date.")
        if study_period.end_date is not None:
            errors.append("ALL_AVAILABLE study period must not set end_date.")
        if study_period.lookback_value is not None:
            errors.append("ALL_AVAILABLE study period must not set lookback_value.")
        if study_period.lookback_unit is not None:
            errors.append("ALL_AVAILABLE study period must not set lookback_unit.")
        return

    errors.append(f"Unsupported study-period type: {study_period.type}")


def _validate_clarification_option(
    option: ClarificationOption,
    request_index: int,
    option_index: int,
    errors: list[str],
) -> None:
    prefix = f"Clarification request {request_index} option {option_index}"
    if not isinstance(option.value, str) or not option.value.strip():
        errors.append(f"{prefix} requires a non-empty value.")
    if not isinstance(option.label, str) or not option.label.strip():
        errors.append(f"{prefix} requires a non-empty label.")


def _validate_clarification_requests(
    specification: ResearchSpecification,
    errors: list[str],
) -> None:
    seen_requests: set[tuple[str, str]] = set()
    for request_index, request in enumerate(
        specification.clarification_requests,
        start=1,
    ):
        if not isinstance(request, ClarificationRequest):
            errors.append(
                f"Clarification request {request_index} has an invalid structure."
            )
            continue
        if not isinstance(request.field, str) or not request.field.strip():
            errors.append(
                f"Clarification request {request_index} requires a non-empty field."
            )
        if not isinstance(request.question, str) or not request.question.strip():
            errors.append(
                f"Clarification request {request_index} requires a non-empty question."
            )

        key = (request.field, request.question)
        if key in seen_requests:
            errors.append(
                "Duplicate clarification request for "
                f"field '{request.field}' and question '{request.question}'."
            )
        seen_requests.add(key)

        seen_option_values: set[str] = set()
        for option_index, option in enumerate(request.options, start=1):
            if not isinstance(option, ClarificationOption):
                errors.append(
                    f"Clarification request {request_index} option {option_index} "
                    "has an invalid structure."
                )
                continue
            _validate_clarification_option(
                option,
                request_index,
                option_index,
                errors,
            )
            if option.value in seen_option_values:
                errors.append(
                    f"Clarification request {request_index} contains duplicate "
                    f"option value '{option.value}'."
                )
            seen_option_values.add(option.value)

    if (
        specification.status == SpecificationStatus.VALID
        and specification.clarification_requests
    ):
        errors.append(
            "VALID specification must not contain unresolved clarification requests."
        )
    if specification.study_period is not None and any(
        request.field == "study_period"
        for request in specification.clarification_requests
        if isinstance(request, ClarificationRequest)
    ):
        errors.append(
            "Study period is already resolved but a study-period clarification "
            "is still pending."
        )


def _research_requires_study_period(
    specification: ResearchSpecification,
) -> bool:
    """Return whether structured RESEARCH fields clearly describe a sample task."""
    return (
        specification.output_type == OutputType.PROBABILITY
        or specification.aggregation is not None
        or bool(specification.conditions)
        or bool(specification.baseline)
    )


def _validate_research_execution(
    specification: ResearchSpecification,
    errors: list[str],
) -> None:
    """Validate consistency required by complete RESEARCH specifications."""
    if (
        specification.status == SpecificationStatus.VALID
        and _research_requires_study_period(specification)
        and specification.study_period is None
    ):
        errors.append("Sample-based RESEARCH requires a resolved study period.")


def _validate_delivery_window(
    delivery_window: DeliveryWindow | None,
    errors: list[str],
) -> None:
    if delivery_window is None:
        return
    if delivery_window.start_local is not None and not _valid_local_time(
        delivery_window.start_local
    ):
        errors.append("Delivery start time must use valid HH:MM format.")
    if delivery_window.end_local is not None and not _valid_local_time(
        delivery_window.end_local
    ):
        errors.append("Delivery end time must use valid HH:MM format.")
    if delivery_window.end_local is not None and delivery_window.start_local is None:
        errors.append("Delivery end time requires a delivery start time.")
    if delivery_window.date is not None and not _valid_iso_date(delivery_window.date):
        errors.append("Delivery date must use valid YYYY-MM-DD format.")


def _validate_decision_points(
    specification: ResearchSpecification,
    errors: list[str],
) -> None:
    for index, point in enumerate(specification.decision_points, start=1):
        prefix = f"Decision point {index}"
        if point.type == DecisionPointType.ABSOLUTE_LOCAL_TIME:
            if not _valid_local_time(point.local_time):
                errors.append(f"{prefix} requires a valid local_time in HH:MM format.")
            if point.minutes_before_delivery is not None:
                errors.append(
                    f"{prefix} with ABSOLUTE_LOCAL_TIME must not set "
                    "minutes_before_delivery."
                )
        elif point.type == DecisionPointType.RELATIVE_TO_DELIVERY:
            if (
                point.minutes_before_delivery is None
                or point.minutes_before_delivery <= 0
            ):
                errors.append(
                    f"{prefix} requires positive minutes_before_delivery."
                )
            if point.local_time is not None:
                errors.append(
                    f"{prefix} with RELATIVE_TO_DELIVERY must not set local_time."
                )
            if (
                specification.delivery_window is None
                or specification.delivery_window.start_local is None
            ):
                errors.append("Relative decision point requires a delivery start time.")
        else:
            errors.append(f"{prefix} has an unsupported decision-point type.")


def _validate_time_structures(
    specification: ResearchSpecification,
    errors: list[str],
) -> None:
    _validate_delivery_window(specification.delivery_window, errors)
    _validate_decision_points(specification, errors)


def _conditions_with_locations(
    specification: ResearchSpecification,
) -> tuple[tuple[str, Condition], ...]:
    located: list[tuple[str, Condition]] = []
    if specification.event is not None:
        located.append(("Event", specification.event))
    located.extend(("Condition", item) for item in specification.conditions)
    located.extend(("Filter", item) for item in specification.filters)
    located.extend(("Baseline", item) for item in specification.baseline)
    return tuple(located)


def _research_quantities(
    specification: ResearchSpecification,
) -> tuple[tuple[str, str], ...]:
    quantities: list[tuple[str, str]] = []
    if specification.target is not None:
        quantities.append(("Target", specification.target))
    if specification.reference is not None:
        quantities.append(("Reference", specification.reference))
    if specification.event is not None:
        quantities.append(("Event", specification.event.variable))
    quantities.extend(("Condition", item.variable) for item in specification.conditions)
    quantities.extend(("Baseline", item.variable) for item in specification.baseline)
    return tuple(quantities)


def _all_variables(
    specification: ResearchSpecification,
) -> tuple[str, ...]:
    variables = [name for _, name in _research_quantities(specification)]
    variables.extend(item.variable for item in specification.filters)
    return tuple(dict.fromkeys(variables))


def _known_registry_terms() -> frozenset[str]:
    terms: set[str] = set()
    for canonical, metadata in VARIABLE_REGISTRY.items():
        terms.add(canonical.casefold())
        terms.add(canonical.replace("_", " ").casefold())
        terms.update(alias.casefold() for alias in metadata.aliases)
    for key, term in DOMAIN_TERMS.items():
        terms.add(key.casefold())
        terms.add(term.name.casefold())
        terms.update(alias.casefold() for alias in term.aliases)
    return frozenset(terms)


def _validate_status_consistency(
    specification: ResearchSpecification,
    errors: list[str],
) -> bool:
    if specification.status == SpecificationStatus.NEEDS_CLARIFICATION:
        if not specification.ambiguities:
            errors.append(
                "NEEDS_CLARIFICATION status requires at least one ambiguity."
            )
        if specification.unsupported_terms:
            errors.append(
                "NEEDS_CLARIFICATION status must not contain unsupported terms."
            )
        return False

    if specification.status == SpecificationStatus.UNSUPPORTED:
        if not specification.unsupported_terms:
            errors.append("UNSUPPORTED status requires at least one unsupported term.")
        known_terms = _known_registry_terms()
        for term in specification.unsupported_terms:
            if term.casefold().strip() in known_terms:
                errors.append(
                    f"Unsupported term is already registered: {term}"
                )
        return False

    if specification.status == SpecificationStatus.VALID:
        if specification.ambiguities:
            errors.append("VALID specification must not contain ambiguities.")
        if specification.unsupported_terms:
            errors.append("VALID specification must not contain unsupported terms.")
        return True

    errors.append(f"Unknown specification status: {specification.status}")
    return False


def _validate_research_language(
    specification: ResearchSpecification,
    errors: list[str],
) -> None:
    if specification.output_type is None:
        errors.append("VALID specification requires an output type.")
        return

    semantics = OUTPUT_SEMANTICS[specification.output_type]
    for field_name in semantics.required_fields:
        if getattr(specification, field_name) is None:
            errors.append(
                f"{specification.output_type.name} output requires a {field_name}."
            )

    if (
        specification.change_direction is not None
        and specification.output_type != OutputType.CHANGE
    ):
        errors.append("change_direction is only valid for CHANGE output.")
    if (
        specification.difference_semantics is not None
        and specification.output_type != OutputType.DIFFERENCE
    ):
        errors.append("difference_semantics is only valid for DIFFERENCE output.")
    if (
        specification.aggregation is not None
        and specification.output_type not in {OutputType.VALUE, OutputType.DIFFERENCE}
    ):
        errors.append("aggregation is only valid for VALUE or DIFFERENCE output.")


def _validate_closed_world(
    specification: ResearchSpecification,
    errors: list[str],
) -> None:
    for variable in _all_variables(specification):
        if variable not in VARIABLE_REGISTRY:
            errors.append(f"Unknown registry variable: {variable}")


def _validate_condition_operators(
    specification: ResearchSpecification,
    errors: list[str],
) -> None:
    for _, condition in _conditions_with_locations(specification):
        if condition.operator not in ALLOWED_CONDITION_OPERATORS:
            errors.append(f"Unsupported condition operator: {condition.operator}")
        elif condition.operator == "between":
            if not isinstance(condition.value, tuple) or len(condition.value) != 2:
                errors.append(
                    f"Condition on '{condition.variable}' uses 'between' "
                    "and requires a two-value tuple."
                )


def _require_role(
    location: str,
    variable: str,
    role: VariableRole,
    errors: list[str],
) -> None:
    metadata = VARIABLE_REGISTRY.get(variable)
    if metadata is not None and role not in metadata.allowed_roles:
        errors.append(
            f"{location} variable '{variable}' is not allowed in the {role.name} role."
        )


def _validate_roles(
    specification: ResearchSpecification,
    errors: list[str],
) -> None:
    for location, variable in _research_quantities(specification):
        _require_role(
            location,
            variable,
            VariableRole.HISTORICAL_ANALYSIS,
            errors,
        )

    for condition in specification.filters:
        _require_role(
            "Filter", condition.variable, VariableRole.FILTER, errors
        )


def validate_specification(
    specification: ResearchSpecification,
) -> ValidationResult:
    """Check a compiled specification without changing or reinterpreting it."""
    errors: list[str] = []
    requires_full_validation = _validate_status_consistency(specification, errors)
    _validate_closed_world(specification, errors)
    _validate_condition_operators(specification, errors)
    _validate_time_structures(specification, errors)
    _validate_study_period(specification.study_period, errors)
    _validate_clarification_requests(specification, errors)
    _validate_research_execution(specification, errors)

    if requires_full_validation:
        _validate_research_language(specification, errors)
        _validate_roles(specification, errors)

    return ValidationResult(is_valid=not errors, errors=tuple(errors))
