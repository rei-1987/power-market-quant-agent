"""Shared configuration and loading boundary for the master market dataset."""

import os
from pathlib import Path

import pandas as pd

from tools.data_tools import (
    load_market_data,
    materialize_variables,
    normalize_datetime_variable,
    required_source_columns_for_variables,
)


_TEMPORAL_AXES = frozenset({"delivery_start_utc", "delivery_start_local"})


def resolve_master_dataset_path(
    *,
    path_override: str | Path | None = None,
) -> Path:
    """Resolve and validate the configured master CSV path."""
    if path_override is not None:
        if isinstance(path_override, str) and not path_override.strip():
            raise ValueError("path_override must not be blank.")
        configured: str | Path = path_override
    else:
        configured_value = os.getenv("MARKET_DATA_CSV")
        if configured_value is None or not configured_value.strip():
            raise RuntimeError(
                "MARKET_DATA_CSV is not set or is blank. Configure the master "
                "market dataset path or supply path_override."
            )
        configured = configured_value.strip()

    path = Path(configured).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Master market dataset file does not exist: {path}"
        )
    return path


def load_master_dataset(
    variables: tuple[str, ...] | list[str] | None = None,
    *,
    path_override: str | Path | None = None,
    materialize: bool = True,
) -> pd.DataFrame:
    """Load all raw master data or only dependencies for requested variables."""
    path = resolve_master_dataset_path(path_override=path_override)
    if variables is None:
        return load_market_data(path)
    if not isinstance(variables, (tuple, list)) or not variables:
        raise ValueError("variables must be a non-empty tuple or list when supplied.")
    requested = tuple(dict.fromkeys(variables))
    columns = required_source_columns_for_variables(requested)
    dataframe = load_market_data(path, columns)
    return (
        materialize_variables(dataframe, requested)
        if materialize
        else dataframe
    )


def latest_available_timestamp(
    dataframe: pd.DataFrame,
    datetime_variable: str = "delivery_start_utc",
) -> pd.Timestamp:
    """Return the latest available observation on a registered temporal axis."""
    if datetime_variable not in _TEMPORAL_AXES:
        raise ValueError(
            "datetime_variable must be delivery_start_utc or delivery_start_local."
        )
    if dataframe.empty:
        raise ValueError("Cannot determine the latest timestamp from an empty DataFrame.")
    normalized = normalize_datetime_variable(dataframe, datetime_variable)
    available = normalized[datetime_variable].dropna()
    if available.empty:
        raise ValueError(
            f"Cannot determine the latest timestamp because '{datetime_variable}' "
            "contains only missing values."
        )
    return available.max()
