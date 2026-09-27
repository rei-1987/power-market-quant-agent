"""Deterministic calculations for registered power-market variables."""

import pandas as pd

from compiler.variable_registry import DataType, VARIABLE_REGISTRY


_ADDITIONS: dict[str, tuple[str, ...]] = {
    "total_wind_a01": ("wind_onshore_a01", "wind_offshore_a01"),
    "total_wind_a40": ("wind_onshore_a40", "wind_offshore_a40"),
    "total_wind_actual": (
        "wind_onshore_actual_mw",
        "wind_offshore_actual_mw",
    ),
    "wind_solar_a01": (
        "solar_a01",
        "wind_onshore_a01",
        "wind_offshore_a01",
    ),
    "wind_solar_a40": (
        "solar_a40",
        "wind_onshore_a40",
        "wind_offshore_a40",
    ),
    "wind_solar_actual": (
        "solar_actual_mw",
        "wind_onshore_actual_mw",
        "wind_offshore_actual_mw",
    ),
}

_SUBTRACTIONS: dict[str, tuple[str, str]] = {
    "id1_minus_id3": ("id1_price_eur_mwh", "id3_price_eur_mwh"),
    "solar_revision": ("solar_a40", "solar_a01"),
    "wind_onshore_revision": ("wind_onshore_a40", "wind_onshore_a01"),
    "wind_offshore_revision": ("wind_offshore_a40", "wind_offshore_a01"),
    "total_wind_revision": ("total_wind_a40", "total_wind_a01"),
    "wind_solar_revision": ("wind_solar_a40", "wind_solar_a01"),
    "solar_a01_deviation": ("solar_actual_mw", "solar_a01"),
    "solar_a40_deviation": ("solar_actual_mw", "solar_a40"),
    "wind_onshore_a01_deviation": (
        "wind_onshore_actual_mw",
        "wind_onshore_a01",
    ),
    "wind_onshore_a40_deviation": (
        "wind_onshore_actual_mw",
        "wind_onshore_a40",
    ),
    "wind_offshore_a01_deviation": (
        "wind_offshore_actual_mw",
        "wind_offshore_a01",
    ),
    "wind_offshore_a40_deviation": (
        "wind_offshore_actual_mw",
        "wind_offshore_a40",
    ),
    "total_wind_a01_deviation": ("total_wind_actual", "total_wind_a01"),
    "total_wind_a40_deviation": ("total_wind_actual", "total_wind_a40"),
    "wind_solar_a01_deviation": ("wind_solar_actual", "wind_solar_a01"),
    "wind_solar_a40_deviation": ("wind_solar_actual", "wind_solar_a40"),
}

_TIME_VARIABLES = {
    "delivery_hour",
    "quarter_hour",
    "weekday",
    "month",
    "year",
    "season",
}


def _column_name(variable: str) -> str:
    """Return the physical column for raw data or canonical derived name."""
    metadata = VARIABLE_REGISTRY[variable]
    if metadata.data_type == DataType.RAW:
        if metadata.source_column is None:
            raise ValueError(f"Raw variable '{variable}' has no source column.")
        return metadata.source_column
    return variable


def _series(dataframe: pd.DataFrame, variable: str) -> pd.Series:
    """Return an already resolved variable column."""
    column = _column_name(variable)
    if column not in dataframe.columns:
        raise ValueError(
            f"Required input column '{column}' for variable '{variable}' is absent."
        )
    return dataframe[column]


def _calculate_time_variable(dataframe: pd.DataFrame, variable: str) -> pd.Series:
    """Calculate one registered market-local time dimension."""
    source = _series(dataframe, "delivery_start_local")
    parsed = pd.to_datetime(source, errors="coerce")
    invalid = source.notna() & parsed.isna()
    if invalid.any():
        raise ValueError(
            "Cannot calculate "
            f"'{variable}': delivery_start_local contains unparseable non-null values."
        )

    if variable == "delivery_hour":
        return parsed.dt.hour
    if variable == "quarter_hour":
        minutes = parsed.dt.minute
        invalid_minutes = minutes.notna() & ~minutes.isin((0, 15, 30, 45))
        if invalid_minutes.any():
            raise ValueError(
                "Cannot calculate 'quarter_hour': delivery_start_local minute values "
                "must be 0, 15, 30, or 45."
            )
        return minutes
    if variable == "weekday":
        return parsed.dt.weekday
    if variable == "month":
        return parsed.dt.month
    if variable == "year":
        return parsed.dt.year

    month_to_season = {
        1: "winter",
        2: "winter",
        3: "spring",
        4: "spring",
        5: "spring",
        6: "summer",
        7: "summer",
        8: "summer",
        9: "autumn",
        10: "autumn",
        11: "autumn",
        12: "winter",
    }
    return parsed.dt.month.map(month_to_season)


def _calculate_derived(dataframe: pd.DataFrame, variable: str) -> pd.Series:
    """Execute one explicitly supported deterministic registry calculation."""
    if variable in _ADDITIONS:
        dependencies = _ADDITIONS[variable]
        result = _series(dataframe, dependencies[0])
        for dependency in dependencies[1:]:
            result = result + _series(dataframe, dependency)
        return result

    if variable in _SUBTRACTIONS:
        left, right = _SUBTRACTIONS[variable]
        return _series(dataframe, left) - _series(dataframe, right)

    if variable in _TIME_VARIABLES:
        return _calculate_time_variable(dataframe, variable)

    raise ValueError(
        f"No deterministic calculation is implemented for derived variable '{variable}'."
    )


def calculate_variables(
    dataframe: pd.DataFrame,
    variables: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    """Calculate registered variables and dependencies in stable requested order."""
    result = dataframe.copy()
    completed: set[str] = set()
    visiting: list[str] = []

    def calculate(variable: str) -> None:
        if variable in completed:
            return
        if variable not in VARIABLE_REGISTRY:
            raise ValueError(f"Unknown registry variable: {variable}")
        if variable in visiting:
            cycle_start = visiting.index(variable)
            cycle = " -> ".join((*visiting[cycle_start:], variable))
            raise ValueError(f"Variable dependency cycle detected: {cycle}")

        metadata = VARIABLE_REGISTRY[variable]
        if metadata.data_type == DataType.RAW:
            _series(result, variable)
            completed.add(variable)
            return

        visiting.append(variable)
        try:
            for dependency in metadata.dependencies:
                calculate(dependency)
            result[variable] = _calculate_derived(result, variable)
        finally:
            visiting.pop()
        completed.add(variable)

    for requested_variable in variables:
        calculate(requested_variable)
    return result


def calculate_variable(dataframe: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Calculate one registered variable without mutating the caller's DataFrame."""
    return calculate_variables(dataframe, (variable,))


# Forecast Calculation layer placeholder: model-based calculations will be added
# separately after the deterministic calculation contract is stable.
