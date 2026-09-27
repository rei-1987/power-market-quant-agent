import pytest

from compiler.compiler_validator import validate_specification
from compiler.hypothesis_compiler import compile_hypothesis
from app.agent_controller import AgentController
from domain.research_language import Aggregation, OutputType
from schemas.hypothesis_schema import (
    Condition,
    DeliveryWindow,
    RelativePeriodUnit,
    SpecificationStatus,
    StudyPeriodType,
)


def test_deterministic_historical_probability() -> None:
    specification = compile_hypothesis(
        "Historically, when total wind revision is negative, what is the "
        "probability that ID1 is above ID3 over the last 3 months?"
    )

    assert specification.status == SpecificationStatus.VALID
    assert specification.output_type == OutputType.PROBABILITY
    assert specification.event is not None
    assert specification.event.variable == "id1_minus_id3"
    assert specification.event.operator == ">"
    assert specification.event.value == 0

    assert len(specification.conditions) == 1
    condition = specification.conditions[0]
    assert condition.variable == "total_wind_revision"
    assert condition.operator == "<"
    assert condition.value == 0
    assert not any(
        item.variable == "id1_price_eur_mwh"
        and item.operator == ">"
        and item.value == "id3"
        for item in specification.conditions
    )
    assert specification.study_period is not None
    assert specification.study_period.type == StudyPeriodType.RELATIVE_LOOKBACK
    assert specification.study_period.lookback_value == 3
    assert specification.study_period.lookback_unit == RelativePeriodUnit.MONTHS

    validation = validate_specification(specification)

    assert validation.is_valid is True
    assert validation.errors == ()


def test_deterministic_wind_revision_ambiguity() -> None:
    specification = compile_hypothesis("Does wind revision affect ID1?")

    assert specification.status == SpecificationStatus.NEEDS_CLARIFICATION
    assert specification.ambiguities
    assert specification.unsupported_terms == ()

    validation = validate_specification(specification)

    assert validation.is_valid is True
    assert validation.errors == ()


def test_deterministic_unsupported_request() -> None:
    specification = compile_hypothesis("Does gas price affect ID1?")

    assert specification.status == SpecificationStatus.UNSUPPORTED
    assert "gas price" in specification.unsupported_terms

    validation = validate_specification(specification)

    assert validation.is_valid is True
    assert validation.errors == ()


def test_deterministic_historical_average_value() -> None:
    specification = compile_hypothesis(
        "Historically, what is the average ID1 price over the last 3 months?"
    )

    assert specification.status == SpecificationStatus.VALID
    assert specification.output_type == OutputType.VALUE
    assert specification.target == "id1_price_eur_mwh"
    assert specification.aggregation == Aggregation.MEAN
    assert specification.study_period is not None
    assert specification.study_period.type == StudyPeriodType.RELATIVE_LOOKBACK
    assert specification.study_period.lookback_value == 3
    assert specification.study_period.lookback_unit == RelativePeriodUnit.MONTHS

    validation = validate_specification(specification)

    assert validation.is_valid is True
    assert validation.errors == ()


@pytest.mark.parametrize(
    "phrase",
    [
        "downward wind forecast revisions",
        "downward wind revisions",
        "negative wind forecast revision",
        "negative wind revisions",
    ],
)
def test_directional_generic_wind_revision_maps_to_total_wind(phrase: str) -> None:
    specification = compile_hypothesis(
        f"Historically, do {phrase} make ID1 more likely to exceed ID3 over the last 1 year?"
    )

    assert specification.status == SpecificationStatus.VALID
    assert specification.conditions == (
        Condition(variable="total_wind_revision", operator="<", value=0),
    )


@pytest.mark.parametrize(
    ("phrase", "expected_variable"),
    [
        ("downward onshore wind revisions", "wind_onshore_revision"),
        ("negative offshore wind forecast revisions", "wind_offshore_revision"),
    ],
)
def test_directional_specific_wind_revision_is_not_mapped_to_total(
    phrase: str,
    expected_variable: str,
) -> None:
    specification = compile_hypothesis(
        f"Historically, do {phrase} make ID1 more likely to exceed ID3 over the last 1 year?"
    )

    assert specification.status == SpecificationStatus.VALID
    assert specification.conditions == (
        Condition(variable=expected_variable, operator="<", value=0),
    )


@pytest.mark.parametrize(
    "phrase",
    ["evening delivery products", "evening delivery", "evening products"],
)
def test_evening_delivery_language_maps_to_fixed_local_window(phrase: str) -> None:
    specification = compile_hypothesis(
        f"Historically, what is the probability that ID1 exceeds ID3 for {phrase} over the last year?"
    )

    assert specification.delivery_window == DeliveryWindow(
        start_local="17:00",
        end_local="21:00",
    )


def test_full_demo_request_preserves_condition_and_evening_window() -> None:
    specification = compile_hypothesis(
        "Do downward wind forecast revisions make ID1 more likely to exceed ID3 "
        "for evening delivery products?"
    )

    assert specification.status == SpecificationStatus.NEEDS_CLARIFICATION
    assert specification.event == Condition("id1_minus_id3", ">", 0)
    assert specification.conditions == (
        Condition("total_wind_revision", "<", 0),
    )
    assert specification.delivery_window == DeliveryWindow("17:00", "21:00")
    assert tuple(item.field for item in specification.clarification_requests) == (
        "study_period",
    )


def test_study_period_clarification_preserves_demo_research_semantics() -> None:
    controller = AgentController()
    controller.process(
        "Do downward wind forecast revisions make ID1 more likely to exceed ID3 "
        "for evening delivery products?"
    )

    result = controller.answer_clarification("Last 1 year")
    specification = result.specification

    assert specification.status == SpecificationStatus.VALID
    assert specification.event == Condition("id1_minus_id3", ">", 0)
    assert specification.conditions == (
        Condition("total_wind_revision", "<", 0),
    )
    assert specification.delivery_window == DeliveryWindow("17:00", "21:00")
    assert specification.study_period is not None
    assert specification.study_period.type == StudyPeriodType.RELATIVE_LOOKBACK
    assert specification.study_period.lookback_value == 1
    assert specification.study_period.lookback_unit == RelativePeriodUnit.YEARS
