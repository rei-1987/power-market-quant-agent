"""Small deterministic quantitative primitives shared by execution pipelines."""

from dataclasses import dataclass
from math import erfc, sqrt
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


_SUPPORTED_OPERATORS = frozenset({">", "<", ">=", "<=", "==", "=", "between"})


@dataclass(frozen=True)
class ConditionalProbabilityResult:
    """Conditional event probability compared with its complement baseline."""

    valid_sample_size: int
    dropped_missing_count: int
    condition_sample_size: int
    baseline_sample_size: int
    condition_event_count: int
    baseline_event_count: int
    condition_probability: float
    baseline_probability: float
    probability_difference_percentage_points: float
    confidence_interval_95_percentage_points: tuple[float, float]
    p_value: float


@dataclass(frozen=True)
class DescriptiveStatsResult:
    """Minimal numeric summary for one dataframe column."""

    sample_size: int
    missing_count: int
    mean: float
    median: float
    std: float
    min: float
    max: float


def apply_condition(
    dataframe: pd.DataFrame,
    variable: str,
    operator: str,
    value: Any,
) -> pd.Series:
    """Return a boolean mask for one explicit dataframe condition."""
    if variable not in dataframe.columns:
        raise ValueError(f"Condition variable column is missing: {variable}")
    if operator not in _SUPPORTED_OPERATORS:
        raise ValueError(f"Unsupported condition operator: {operator}")

    series = dataframe[variable]
    try:
        if operator == ">":
            mask = series > value
        elif operator == "<":
            mask = series < value
        elif operator == ">=":
            mask = series >= value
        elif operator == "<=":
            mask = series <= value
        elif operator in {"==", "="}:
            mask = series == value
        else:
            if (
                not isinstance(value, (tuple, list))
                or len(value) != 2
            ):
                raise ValueError(
                    "The 'between' operator requires a two-item tuple or list."
                )
            lower, upper = value
            mask = series.between(lower, upper, inclusive="both")
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("The 'between'"):
            raise
        raise ValueError(
            f"Could not apply condition: {variable} {operator} {value!r}."
        ) from exc

    return mask.fillna(False).astype(bool)


def conditional_probability(
    dataframe: pd.DataFrame,
    *,
    event_variable: str,
    event_operator: str,
    event_value: Any,
    condition_variable: str,
    condition_operator: str,
    condition_value: Any,
) -> ConditionalProbabilityResult:
    """Compare an event probability under a condition and its complement."""
    for variable in dict.fromkeys((event_variable, condition_variable)):
        if variable not in dataframe.columns:
            raise ValueError(f"Required calculation column is missing: {variable}")

    required = list(dict.fromkeys((event_variable, condition_variable)))
    valid_mask = dataframe[required].notna().all(axis=1)
    valid_sample_size = int(valid_mask.sum())
    dropped_missing_count = int((~valid_mask).sum())
    if valid_sample_size == 0:
        raise ValueError("Conditional probability has no valid rows after dropping missing values.")

    valid_data = dataframe.loc[valid_mask]
    event_mask = apply_condition(
        valid_data, event_variable, event_operator, event_value
    )
    condition_mask = apply_condition(
        valid_data, condition_variable, condition_operator, condition_value
    )
    baseline_mask = ~condition_mask

    condition_sample_size = int(condition_mask.sum())
    baseline_sample_size = int(baseline_mask.sum())
    if condition_sample_size == 0:
        raise ValueError("Conditional probability condition group is empty.")
    if baseline_sample_size == 0:
        raise ValueError("Conditional probability baseline group is empty.")

    condition_event_count = int((event_mask & condition_mask).sum())
    baseline_event_count = int((event_mask & baseline_mask).sum())
    condition_probability = condition_event_count / condition_sample_size
    baseline_probability = baseline_event_count / baseline_sample_size
    difference = condition_probability - baseline_probability

    unpooled_standard_error = sqrt(
        condition_probability * (1.0 - condition_probability) / condition_sample_size
        + baseline_probability * (1.0 - baseline_probability) / baseline_sample_size
    )
    margin = 1.96 * unpooled_standard_error

    pooled_probability = (
        condition_event_count + baseline_event_count
    ) / valid_sample_size
    pooled_standard_error = sqrt(
        pooled_probability
        * (1.0 - pooled_probability)
        * (1.0 / condition_sample_size + 1.0 / baseline_sample_size)
    )
    if pooled_standard_error == 0.0:
        p_value = 1.0 if difference == 0.0 else 0.0
    else:
        z_score = difference / pooled_standard_error
        p_value = erfc(abs(z_score) / sqrt(2.0))

    return ConditionalProbabilityResult(
        valid_sample_size=valid_sample_size,
        dropped_missing_count=dropped_missing_count,
        condition_sample_size=condition_sample_size,
        baseline_sample_size=baseline_sample_size,
        condition_event_count=condition_event_count,
        baseline_event_count=baseline_event_count,
        condition_probability=condition_probability,
        baseline_probability=baseline_probability,
        probability_difference_percentage_points=difference * 100.0,
        confidence_interval_95_percentage_points=(
            (difference - margin) * 100.0,
            (difference + margin) * 100.0,
        ),
        p_value=p_value,
    )


def descriptive_stats(
    dataframe: pd.DataFrame,
    variable: str,
) -> DescriptiveStatsResult:
    """Return deterministic descriptive statistics for one numeric column."""
    if variable not in dataframe.columns:
        raise ValueError(f"Descriptive-statistics column is missing: {variable}")
    series = dataframe[variable]
    if not is_numeric_dtype(series.dtype) or is_bool_dtype(series.dtype):
        raise ValueError(f"Descriptive statistics require a numeric column: {variable}")

    missing_count = int(series.isna().sum())
    valid = series.dropna()
    sample_size = int(len(valid))
    if sample_size == 0:
        raise ValueError("Descriptive statistics have no valid numeric observations.")

    return DescriptiveStatsResult(
        sample_size=sample_size,
        missing_count=missing_count,
        mean=float(valid.mean()),
        median=float(valid.median()),
        std=float(valid.std(ddof=1)),
        min=float(valid.min()),
        max=float(valid.max()),
    )
