"""Deterministic simulated-live sign-regime forecast pipeline."""

from dataclasses import dataclass
from datetime import date, time
from pathlib import Path

import pandas as pd

from tools.data_source import load_master_dataset
from tools.data_tools import (
    filter_delivery_time_window,
    materialize_variables,
    normalize_datetime_variable,
)
from tools.quant_tools import conditional_probability


_DELIVERY_START = "17:00"
_DELIVERY_END = "21:00"
_VARIABLES = (
    "total_wind_revision",
    "id1_minus_id3",
    "delivery_start_local",
    "delivery_start_utc",
)


@dataclass(frozen=True)
class LiveForecastResult:
    """Historically calibrated inference at a simulated live decision time."""

    original_request: str
    simulated_current_local: pd.Timestamp
    current_delivery_date: date
    delivery_window: tuple[str, str]
    historical_calibration_observations: int
    current_product_count: int
    current_mean_total_wind_revision: float
    signal_regime: str
    inferred_probability: float
    comparison_probability: float
    applicable_difference_percentage_points: float
    applicable_confidence_interval_95_percentage_points: tuple[float, float]
    p_value: float
    dropped_missing_calibration_rows: int
    interpretation: str
    ex_post_demo_event_rate: float | None
    ex_post_demo_valid_observations: int


def _load_live_data(
    market_data_source: str | Path | pd.DataFrame | None,
) -> pd.DataFrame:
    if market_data_source is None:
        dataframe = load_master_dataset(variables=list(_VARIABLES), materialize=True)
    elif isinstance(market_data_source, (str, Path)):
        dataframe = load_master_dataset(
            variables=list(_VARIABLES),
            path_override=market_data_source,
            materialize=True,
        )
    elif isinstance(market_data_source, pd.DataFrame):
        dataframe = materialize_variables(market_data_source, list(_VARIABLES))
    else:
        raise ValueError("market_data_source must be a CSV path or pandas DataFrame.")
    return normalize_datetime_variable(dataframe, "delivery_start_local")


def run_live_forecast(
    request: str,
    market_data_source: str | Path | pd.DataFrame | None = None,
) -> LiveForecastResult:
    """Run the latest-date 08:00 simulated Live Forecast MVP."""
    if not isinstance(request, str) or not request.strip():
        raise ValueError("Live Forecast requires a non-empty request.")
    dataframe = _load_live_data(market_data_source)
    available_local = dataframe["delivery_start_local"].dropna()
    if available_local.empty:
        raise ValueError("Live Forecast requires available local delivery timestamps.")

    current_date = available_local.max().date()
    simulated_current = pd.Timestamp.combine(current_date, time(8, 0))
    local_dates = dataframe["delivery_start_local"].dt.date

    calibration = dataframe.loc[local_dates < current_date].copy()
    calibration = filter_delivery_time_window(
        calibration,
        "delivery_start_local",
        start_local=_DELIVERY_START,
        end_local=_DELIVERY_END,
    )
    calibration_result = conditional_probability(
        calibration,
        event_variable="id1_minus_id3",
        event_operator=">",
        event_value=0,
        condition_variable="total_wind_revision",
        condition_operator="<",
        condition_value=0,
    )

    current_products = dataframe.loc[local_dates == current_date].copy()
    current_products = filter_delivery_time_window(
        current_products,
        "delivery_start_local",
        start_local=_DELIVERY_START,
        end_local=_DELIVERY_END,
    )
    revisions = current_products["total_wind_revision"].dropna()
    if revisions.empty:
        raise ValueError("Current evening products have no available A01/A40 wind revisions.")
    mean_revision = float(revisions.mean())
    negative_regime = mean_revision < 0
    regime = "negative" if negative_regime else "non-negative"
    inferred_probability = (
        calibration_result.condition_probability
        if negative_regime
        else calibration_result.baseline_probability
    )
    comparison_probability = (
        calibration_result.baseline_probability
        if negative_regime
        else calibration_result.condition_probability
    )
    raw_difference = calibration_result.probability_difference_percentage_points
    raw_interval = calibration_result.confidence_interval_95_percentage_points
    applicable_difference = raw_difference if negative_regime else -raw_difference
    applicable_interval = (
        raw_interval
        if negative_regime
        else (-raw_interval[1], -raw_interval[0])
    )

    outcomes = current_products["id1_minus_id3"].dropna()
    ex_post_rate = float((outcomes > 0).mean()) if not outcomes.empty else None
    interpretation = (
        f"At the simulated current time {simulated_current:%Y-%m-%d %H:%M} "
        f"market-local, the latest-date evening mean total-wind revision was "
        f"{mean_revision:.2f} MW ({regime}). The historically calibrated "
        f"probability of ID1 exceeding ID3 for this regime was "
        f"{inferred_probability:.2%}. This is a non-causal regime inference. "
        "The MVP treats the latest historical delivery date as simulated live "
        "time and does not use current-date final prices or actual generation."
    )

    return LiveForecastResult(
        original_request=request,
        simulated_current_local=simulated_current,
        current_delivery_date=current_date,
        delivery_window=(_DELIVERY_START, _DELIVERY_END),
        historical_calibration_observations=calibration_result.valid_sample_size,
        current_product_count=len(current_products),
        current_mean_total_wind_revision=mean_revision,
        signal_regime=regime,
        inferred_probability=inferred_probability,
        comparison_probability=comparison_probability,
        applicable_difference_percentage_points=applicable_difference,
        applicable_confidence_interval_95_percentage_points=applicable_interval,
        p_value=calibration_result.p_value,
        dropped_missing_calibration_rows=calibration_result.dropped_missing_count,
        interpretation=interpretation,
        ex_post_demo_event_rate=ex_post_rate,
        ex_post_demo_valid_observations=len(outcomes),
    )
