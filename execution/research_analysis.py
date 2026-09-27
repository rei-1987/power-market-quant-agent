"""Calculate defined v0.1 statistics from constructed Research samples."""

from dataclasses import dataclass

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from domain.research_language import Aggregation, OutputType
from execution.research_sample_builder import ResearchSampleSet
from schemas.hypothesis_schema import ResearchSpecification


@dataclass(frozen=True)
class ResearchAnalysisResult:
    """Numerical Research result before interpretation or reasoning."""

    output_type: OutputType
    statistic: str
    value: float | None
    sample_size: int
    valid_observation_count: int
    numerator_count: int | None = None
    denominator_count: int | None = None


def _analyze_value(
    specification: ResearchSpecification,
    samples: ResearchSampleSet,
) -> ResearchAnalysisResult:
    target = specification.target
    if target is None:
        raise ValueError("VALUE analysis requires a target variable.")
    if target not in samples.analysis_sample.columns:
        raise ValueError(
            f"VALUE target column '{target}' is absent from analysis_sample."
        )

    aggregation = specification.aggregation
    if aggregation == Aggregation.EXACT:
        raise NotImplementedError(
            "EXACT aggregation is not implemented in Research Analysis v0.1."
        )
    supported = {
        Aggregation.MEAN,
        Aggregation.MEDIAN,
        Aggregation.MIN,
        Aggregation.MAX,
    }
    if aggregation not in supported:
        raise NotImplementedError(
            "VALUE analysis requires MEAN, MEDIAN, MIN, or MAX aggregation "
            "in Research Analysis v0.1."
        )

    observations = samples.analysis_sample[target]
    sample_size = len(observations)
    valid_observations = observations.dropna()
    valid_count = len(valid_observations)
    if valid_count == 0:
        return ResearchAnalysisResult(
            output_type=OutputType.VALUE,
            statistic=aggregation.value,
            value=None,
            sample_size=sample_size,
            valid_observation_count=0,
        )
    if not is_numeric_dtype(valid_observations.dtype) or is_bool_dtype(
        valid_observations.dtype
    ):
        raise TypeError("VALUE analysis requires numerical target observations.")

    operations = {
        Aggregation.MEAN: valid_observations.mean,
        Aggregation.MEDIAN: valid_observations.median,
        Aggregation.MIN: valid_observations.min,
        Aggregation.MAX: valid_observations.max,
    }
    value = float(operations[aggregation]())
    return ResearchAnalysisResult(
        output_type=OutputType.VALUE,
        statistic=aggregation.value,
        value=value,
        sample_size=sample_size,
        valid_observation_count=valid_count,
    )


def _analyze_probability(
    samples: ResearchSampleSet,
) -> ResearchAnalysisResult:
    event_mask = samples.event_mask
    if event_mask is None:
        raise ValueError("PROBABILITY analysis requires an event_mask.")
    if not event_mask.index.equals(samples.analysis_sample.index):
        raise ValueError("event_mask must align with analysis_sample.")
    if not is_bool_dtype(event_mask.dtype):
        raise TypeError("event_mask must contain boolean or nullable boolean values.")

    denominator = int(event_mask.count())
    numerator = int(event_mask.sum(skipna=True))
    value = numerator / denominator if denominator else None
    return ResearchAnalysisResult(
        output_type=OutputType.PROBABILITY,
        statistic="probability",
        value=value,
        sample_size=len(samples.analysis_sample),
        valid_observation_count=denominator,
        numerator_count=numerator,
        denominator_count=denominator,
    )


def analyze_research(
    specification: ResearchSpecification,
    samples: ResearchSampleSet,
) -> ResearchAnalysisResult:
    """Analyze one supported output without rebuilding Research samples."""
    if not isinstance(specification, ResearchSpecification):
        raise ValueError("specification must be a ResearchSpecification.")
    if not isinstance(samples, ResearchSampleSet):
        raise ValueError("samples must be a ResearchSampleSet.")
    if specification.baseline or samples.baseline_sample is not None:
        raise NotImplementedError(
            "Baseline comparison analysis is not implemented in Research "
            "Analysis v0.1."
        )

    if specification.output_type == OutputType.VALUE:
        return _analyze_value(specification, samples)
    if specification.output_type == OutputType.PROBABILITY:
        return _analyze_probability(samples)
    if specification.output_type == OutputType.CHANGE:
        raise NotImplementedError(
            "CHANGE analysis is not implemented in Research Analysis v0.1."
        )
    if specification.output_type == OutputType.DIFFERENCE:
        raise NotImplementedError(
            "DIFFERENCE analysis is not implemented in Research Analysis v0.1."
        )
    raise NotImplementedError(
        "The requested output type is not implemented in Research Analysis v0.1."
    )
