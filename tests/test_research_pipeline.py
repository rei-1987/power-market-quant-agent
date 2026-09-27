import pandas as pd

from domain.research_language import OutputType
from execution.research_pipeline import execute_research_specification
from schemas.hypothesis_schema import (
    Condition,
    DeliveryWindow,
    RelativePeriodUnit,
    ResearchSpecification,
    SpecificationStatus,
    StudyPeriod,
    StudyPeriodType,
    ValidationResult,
)


def test_probability_pipeline_uses_dataset_time_and_complement_baseline() -> None:
    local_times = [
        "2026-09-12T17:00:00+02:00",
        "2026-09-12T17:15:00+02:00",
        "2026-09-12T17:30:00+02:00",
        "2026-09-12T17:45:00+02:00",
        "2026-09-12T18:00:00+02:00",
        "2026-09-12T18:15:00+02:00",
        "2026-09-12T18:30:00+02:00",
        "2026-09-12T18:45:00+02:00",
        "2026-09-12T21:00:00+02:00",
        "2024-09-12T18:00:00+02:00",
    ]
    dataframe = pd.DataFrame(
        {
            "delivery_start_local": local_times,
            "delivery_start_utc": [
                "2026-09-12T15:00:00Z",
                "2026-09-12T15:15:00Z",
                "2026-09-12T15:30:00Z",
                "2026-09-12T15:45:00Z",
                "2026-09-12T16:00:00Z",
                "2026-09-12T16:15:00Z",
                "2026-09-12T16:30:00Z",
                "2026-09-12T16:45:00Z",
                "2026-09-12T19:00:00Z",
                "2024-09-12T16:00:00Z",
            ],
            "id1_price_eur_mwh": [2, 2, 2, -1, 2, -1, -1, -1, 2, 2],
            "id3_price_eur_mwh": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "wind_onshore_a40": [0, 0, 0, 0, 2, 2, 2, 2, 0, 0],
            "wind_offshore_a40": [0] * 10,
            "wind_onshore_a01": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "wind_offshore_a01": [0] * 10,
        }
    )
    specification = ResearchSpecification(
        original_hypothesis="demo",
        status=SpecificationStatus.VALID,
        output_type=OutputType.PROBABILITY,
        event=Condition("id1_minus_id3", ">", 0),
        conditions=(Condition("total_wind_revision", "<", 0),),
        delivery_window=DeliveryWindow("17:00", "21:00"),
        study_period=StudyPeriod(
            type=StudyPeriodType.RELATIVE_LOOKBACK,
            lookback_value=1,
            lookback_unit=RelativePeriodUnit.YEARS,
        ),
    )

    result = execute_research_specification(
        specification,
        ValidationResult(is_valid=True),
        dataframe,
    )

    assert result.latest_dataset_timestamp == pd.Timestamp("2026-09-12T19:00:00Z")
    assert result.resolved_study_period is not None
    assert result.resolved_study_period.start_date.isoformat() == "2025-09-12"
    assert result.resolved_study_period.end_date.isoformat() == "2026-09-12"
    assert result.research_samples is not None
    assert result.research_samples.base_row_count == 8
    assert result.quantitative_result is not None
    assert result.quantitative_result.valid_sample_size == 8
    assert result.quantitative_result.condition_sample_size == 4
    assert result.quantitative_result.baseline_sample_size == 4
    assert result.quantitative_result.condition_probability == 0.75
    assert result.quantitative_result.baseline_probability == 0.25
    assert result.quantitative_result.probability_difference_percentage_points == 50.0
    assert "association" in result.interpretation
