"""Construct Research samples from already prepared market data."""

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import pandas as pd

from execution.research_context import ResolvedStudyPeriod
from schemas.hypothesis_schema import Condition, DeliveryWindow, ResearchSpecification


_DELIVERY_TIMESTAMP = "delivery_start_local"
_SUPPORTED_OPERATORS = {">", "<", ">=", "<=", "=", "==", "between"}


@dataclass(frozen=True)
class ResearchSampleSet:
    """Selected Research cohorts and an optional event outcome mask."""

    base_sample: pd.DataFrame
    analysis_sample: pd.DataFrame
    baseline_sample: pd.DataFrame | None
    event_mask: pd.Series | None
    input_row_count: int
    base_row_count: int
    analysis_row_count: int
    baseline_row_count: int | None


def _delivery_timestamps(dataframe: pd.DataFrame) -> pd.Series:
    if _DELIVERY_TIMESTAMP not in dataframe.columns:
        raise ValueError(
            "Research time selection requires a 'delivery_start_local' column."
        )
    return pd.to_datetime(dataframe[_DELIVERY_TIMESTAMP], errors="coerce")


def _parse_time(value: str | None, field_name: str) -> time:
    if not isinstance(value, str):
        raise ValueError(f"DeliveryWindow requires {field_name} in HH:MM format.")
    try:
        return datetime.strptime(value, "%H:%M").time()
    except ValueError as error:
        raise ValueError(
            f"DeliveryWindow requires a valid {field_name} in HH:MM format."
        ) from error


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(
            "DeliveryWindow requires a valid date in YYYY-MM-DD format."
        ) from error


def _study_period_mask(
    dataframe: pd.DataFrame,
    period: ResolvedStudyPeriod,
) -> pd.Series:
    timestamps = _delivery_timestamps(dataframe)
    delivery_dates = timestamps.dt.date
    mask = timestamps.notna() & (delivery_dates <= period.end_date)
    if period.start_date is not None:
        mask &= delivery_dates >= period.start_date
    return mask.fillna(False)


def _delivery_window_mask(
    dataframe: pd.DataFrame,
    window: DeliveryWindow,
) -> pd.Series:
    timestamps = _delivery_timestamps(dataframe)
    valid = timestamps.notna()
    start = _parse_time(window.start_local, "start_local")
    end = _parse_time(window.end_local, "end_local")
    if start == end:
        raise ValueError("DeliveryWindow start_local and end_local must differ.")

    mask = pd.Series(False, index=dataframe.index, dtype=bool)
    if window.date is not None:
        start_date = _parse_date(window.date)
        start_timestamp = pd.Timestamp(datetime.combine(start_date, start))
        end_date = start_date if start < end else start_date + timedelta(days=1)
        end_timestamp = pd.Timestamp(datetime.combine(end_date, end))
        mask.loc[valid] = (
            (timestamps.loc[valid] >= start_timestamp)
            & (timestamps.loc[valid] < end_timestamp)
        )
        return mask

    local_times = timestamps.loc[valid].dt.time
    if start < end:
        mask.loc[valid] = (local_times >= start) & (local_times < end)
    else:
        mask.loc[valid] = (local_times >= start) | (local_times < end)
    return mask


def _operator_result(series: pd.Series, condition: Condition) -> pd.Series:
    operator = condition.operator
    if operator not in _SUPPORTED_OPERATORS:
        raise ValueError(f"Unsupported condition operator: {operator}")
    if operator == ">":
        return series > condition.value
    if operator == "<":
        return series < condition.value
    if operator == ">=":
        return series >= condition.value
    if operator == "<=":
        return series <= condition.value
    if operator in {"=", "=="}:
        return series == condition.value

    boundaries = condition.value
    if not isinstance(boundaries, tuple) or len(boundaries) != 2:
        raise ValueError(
            f"Condition on '{condition.variable}' uses 'between' and requires "
            "a two-value tuple."
        )
    lower, upper = boundaries
    return (series >= lower) & (series <= upper)


def _predicate_mask(
    dataframe: pd.DataFrame,
    condition: Condition,
    *,
    preserve_missing: bool,
) -> pd.Series:
    if condition.variable not in dataframe.columns:
        raise ValueError(
            f"Predicate variable '{condition.variable}' is absent from prepared_data."
        )

    values = dataframe[condition.variable]
    available = values.notna()
    if preserve_missing:
        mask = pd.Series(pd.NA, index=dataframe.index, dtype="boolean")
        mask.loc[available] = _operator_result(
            values.loc[available], condition
        ).astype("boolean")
        return mask

    mask = pd.Series(False, index=dataframe.index, dtype=bool)
    mask.loc[available] = _operator_result(
        values.loc[available], condition
    ).fillna(False).astype(bool)
    return mask


def _combined_predicate_mask(
    dataframe: pd.DataFrame,
    conditions: tuple[Condition, ...],
) -> pd.Series:
    mask = pd.Series(True, index=dataframe.index, dtype=bool)
    for condition in conditions:
        mask &= _predicate_mask(
            dataframe,
            condition,
            preserve_missing=False,
        )
    return mask


def build_research_samples(
    specification: ResearchSpecification,
    prepared_data: pd.DataFrame,
    resolved_study_period: ResolvedStudyPeriod | None,
) -> ResearchSampleSet:
    """Build base, analysis, baseline, and event samples in v0.1 order."""
    if not isinstance(specification, ResearchSpecification):
        raise ValueError("specification must be a ResearchSpecification.")
    if not isinstance(prepared_data, pd.DataFrame):
        raise ValueError("prepared_data must be a pandas DataFrame.")

    selected = prepared_data.copy()
    if resolved_study_period is not None:
        selected = selected.loc[
            _study_period_mask(selected, resolved_study_period)
        ].copy()
    if specification.delivery_window is not None:
        selected = selected.loc[
            _delivery_window_mask(selected, specification.delivery_window)
        ].copy()
    if specification.filters:
        selected = selected.loc[
            _combined_predicate_mask(selected, specification.filters)
        ].copy()
    base_sample = selected.copy()

    if specification.conditions:
        analysis_sample = base_sample.loc[
            _combined_predicate_mask(base_sample, specification.conditions)
        ].copy()
    else:
        analysis_sample = base_sample.copy()

    if specification.baseline:
        baseline_sample = base_sample.loc[
            _combined_predicate_mask(base_sample, specification.baseline)
        ].copy()
    else:
        baseline_sample = None

    event_mask = (
        _predicate_mask(
            analysis_sample,
            specification.event,
            preserve_missing=True,
        )
        if specification.event is not None
        else None
    )

    return ResearchSampleSet(
        base_sample=base_sample,
        analysis_sample=analysis_sample,
        baseline_sample=baseline_sample,
        event_mask=event_mask,
        input_row_count=len(prepared_data),
        base_row_count=len(base_sample),
        analysis_row_count=len(analysis_sample),
        baseline_row_count=(
            len(baseline_sample) if baseline_sample is not None else None
        ),
    )
