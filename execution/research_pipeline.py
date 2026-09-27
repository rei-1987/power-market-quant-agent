"""Orchestrate Research compilation, data preparation, and quantitative analysis."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from compiler.compiler_validator import validate_specification
from compiler.hypothesis_compiler import compile_hypothesis
from execution.research_analysis import ResearchAnalysisResult, analyze_research
from execution.research_context import ResolvedStudyPeriod, resolve_study_period
from execution.research_sample_builder import (
    ResearchSampleSet,
    build_research_samples,
)
from schemas.hypothesis_schema import (
    ResearchSpecification,
    SpecificationStatus,
    ValidationResult,
)
from domain.research_language import OutputType
from tools.data_source import latest_available_timestamp, load_master_dataset
from tools.data_tools import (
    materialize_variables,
    normalize_datetime_variable,
    required_source_columns,
    specification_variables,
)
from tools.quant_tools import ConditionalProbabilityResult, conditional_probability


MarketDataSource = str | Path | pd.DataFrame


@dataclass(frozen=True)
class ResearchPipelineResult:
    """Structured result of one executable Research request."""

    specification: ResearchSpecification
    validation: ValidationResult
    resolved_study_period: ResolvedStudyPeriod | None = None
    required_variables: tuple[str, ...] = ()
    required_source_columns: tuple[str, ...] = ()
    prepared_data: pd.DataFrame | None = None
    research_samples: ResearchSampleSet | None = None
    analysis_result: ResearchAnalysisResult | None = None
    quantitative_result: ConditionalProbabilityResult | None = None
    latest_dataset_timestamp: pd.Timestamp | None = None
    interpretation: str | None = None


def _execution_variables(
    specification: ResearchSpecification,
) -> tuple[str, ...]:
    variables = list(specification_variables(specification))
    for temporal_axis in ("delivery_start_local", "delivery_start_utc"):
        if temporal_axis not in variables:
            variables.append(temporal_axis)
    return tuple(variables)


def _interpret_probability(result: ConditionalProbabilityResult) -> str:
    difference = result.probability_difference_percentage_points
    if difference < 0:
        return (
            "In this historical sample, the event probability was "
            f"{abs(difference):.2f} percentage points lower when the condition "
            "held than in its complement baseline. This is an association, not "
            "evidence of causality, and the observed direction is opposite to the "
            "proposed higher-probability relationship."
        )
    if difference > 0:
        return (
            "In this historical sample, the event probability was "
            f"{difference:.2f} percentage points higher when the condition held "
            "than in its complement baseline. This describes association, not "
            "causality."
        )
    return (
        "In this historical sample, the event probability was the same under the "
        "condition and its complement baseline. This describes association, not "
        "causality."
    )


def execute_research_specification(
    specification: ResearchSpecification,
    validation: ValidationResult,
    market_data_source: MarketDataSource | None = None,
) -> ResearchPipelineResult:
    """Execute one already-compiled and validated Research specification."""
    initial = ResearchPipelineResult(specification=specification, validation=validation)
    if specification.status != SpecificationStatus.VALID or not validation.is_valid:
        return initial

    variables = _execution_variables(specification)
    source_columns = required_source_columns(specification)
    if market_data_source is None:
        prepared_data = load_master_dataset(variables=variables, materialize=True)
    elif isinstance(market_data_source, (str, Path)):
        prepared_data = load_master_dataset(
            variables=variables,
            path_override=market_data_source,
            materialize=True,
        )
    elif isinstance(market_data_source, pd.DataFrame):
        prepared_data = materialize_variables(market_data_source, list(variables))
    else:
        raise ValueError("market_data_source must be a CSV path or pandas DataFrame.")

    prepared_data = normalize_datetime_variable(
        prepared_data, "delivery_start_local"
    )
    latest_timestamp = latest_available_timestamp(
        prepared_data, "delivery_start_utc"
    )
    resolved_period = (
        resolve_study_period(specification.study_period, latest_timestamp.date())
        if specification.study_period is not None
        else None
    )
    samples = build_research_samples(
        specification=specification,
        prepared_data=prepared_data,
        resolved_study_period=resolved_period,
    )

    quantitative_result = None
    analysis_result = None
    interpretation = None
    if (
        specification.output_type == OutputType.PROBABILITY
        and specification.event is not None
        and len(specification.conditions) == 1
    ):
        event = specification.event
        condition = specification.conditions[0]
        quantitative_result = conditional_probability(
            samples.base_sample,
            event_variable=event.variable,
            event_operator=event.operator,
            event_value=event.value,
            condition_variable=condition.variable,
            condition_operator=condition.operator,
            condition_value=condition.value,
        )
        interpretation = _interpret_probability(quantitative_result)
    else:
        analysis_result = analyze_research(specification, samples)

    return ResearchPipelineResult(
        specification=specification,
        validation=validation,
        resolved_study_period=resolved_period,
        required_variables=variables,
        required_source_columns=source_columns,
        prepared_data=prepared_data,
        research_samples=samples,
        analysis_result=analysis_result,
        quantitative_result=quantitative_result,
        latest_dataset_timestamp=latest_timestamp,
        interpretation=interpretation,
    )


def run_research_pipeline(
    completed_request_context: str,
    as_of_date: date | None = None,
    market_data_source: MarketDataSource | None = None,
) -> ResearchPipelineResult:
    """Compile, validate, and execute one routed Research request.

    ``as_of_date`` is retained for call compatibility; relative periods are
    anchored to the dataset's latest available UTC observation.
    """
    if (
        not isinstance(completed_request_context, str)
        or not completed_request_context.strip()
    ):
        raise ValueError("Research pipeline requires non-empty request context.")

    specification = compile_hypothesis(completed_request_context)
    validation = validate_specification(specification)
    return execute_research_specification(
        specification,
        validation,
        market_data_source,
    )
