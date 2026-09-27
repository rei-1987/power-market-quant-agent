"""Deterministic historical replay pipeline for the sign-regime MVP."""

import re
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
from tools.quant_tools import ConditionalProbabilityResult, conditional_probability


_REPLAY_PATTERN = re.compile(
    r"\breplay\s+(\d{4}-\d{2}-\d{2})\s+from\s+(\d{1,2}:\d{2})\b",
    re.IGNORECASE,
)
_DELIVERY_START = "17:00"
_DELIVERY_END = "21:00"
_VARIABLES = (
    "total_wind_revision",
    "id1_minus_id3",
    "delivery_start_local",
    "delivery_start_utc",
)


@dataclass(frozen=True)
class HistoricalReplayResult:
    """Result of one point-in-time historical replay."""

    original_request: str
    replay_timestamp_local: pd.Timestamp
    delivery_window: tuple[str, str]
    historical_calibration_observations: int
    replay_product_count: int
    replay_mean_total_wind_revision: float
    signal_regime: str
    inferred_probability: float
    comparison_probability: float
    applicable_difference_percentage_points: float
    applicable_confidence_interval_95_percentage_points: tuple[float, float]
    p_value: float
    dropped_missing_calibration_rows: int
    inference: str
    ex_post_event_rate: float
    ex_post_valid_observations: int


def parse_replay_timestamp(request: str) -> pd.Timestamp:
    """Parse the explicit market-local replay timestamp from a request."""
    if not isinstance(request, str) or not request.strip():
        raise ValueError("Historical Replay requires a non-empty request.")
    match = _REPLAY_PATTERN.search(request)
    if match is None:
        raise ValueError(
            "Historical Replay requires 'Replay YYYY-MM-DD from HH:MM'."
        )
    try:
        replay_timestamp = pd.Timestamp(f"{match.group(1)} {match.group(2)}")
    except ValueError as exc:
        raise ValueError("Historical Replay contains an invalid date or time.") from exc
    if replay_timestamp.tzinfo is not None:
        raise ValueError("Historical Replay timestamp must be market-local wall-clock time.")
    return replay_timestamp


def _load_replay_data(
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


def run_historical_replay(
    request: str,
    market_data_source: str | Path | pd.DataFrame | None = None,
) -> HistoricalReplayResult:
    """Run the deterministic 08:00 sign-regime historical replay."""
    replay_timestamp = parse_replay_timestamp(request)
    if replay_timestamp.time() != time(8, 0):
        raise ValueError("Historical Replay MVP currently supports replay time 08:00 only.")

    dataframe = _load_replay_data(market_data_source)
    local_dates = dataframe["delivery_start_local"].dt.date
    replay_date = replay_timestamp.date()

    calibration = dataframe.loc[local_dates < replay_date].copy()
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

    replay_products = dataframe.loc[local_dates == replay_date].copy()
    replay_products = filter_delivery_time_window(
        replay_products,
        "delivery_start_local",
        start_local=_DELIVERY_START,
        end_local=_DELIVERY_END,
    )
    revisions = replay_products["total_wind_revision"].dropna()
    if revisions.empty:
        raise ValueError("Replay-date evening products have no available A01/A40 wind revisions.")
    mean_revision = float(revisions.mean())
    negative_regime = mean_revision < 0
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
    regime = "negative" if negative_regime else "non-negative"
    raw_difference = calibration_result.probability_difference_percentage_points
    raw_interval = calibration_result.confidence_interval_95_percentage_points
    applicable_difference = raw_difference if negative_regime else -raw_difference
    applicable_interval = (
        raw_interval
        if negative_regime
        else (-raw_interval[1], -raw_interval[0])
    )

    outcomes = replay_products["id1_minus_id3"].dropna()
    if outcomes.empty:
        raise ValueError("Replay-date evening products have no ex-post ID1/ID3 outcomes.")
    ex_post_event_rate = float((outcomes > 0).mean())
    inference = (
        f"At {replay_timestamp:%Y-%m-%d %H:%M} market-local, the replay-date "
        f"evening mean total-wind revision was {mean_revision:.2f} MW ({regime}). "
        f"The pre-date historical probability for that regime was "
        f"{inferred_probability:.2%}. This is a non-causal historical-regime "
        "inference; replay-date final prices and actual generation were not used."
    )

    return HistoricalReplayResult(
        original_request=request,
        replay_timestamp_local=replay_timestamp,
        delivery_window=(_DELIVERY_START, _DELIVERY_END),
        historical_calibration_observations=calibration_result.valid_sample_size,
        replay_product_count=len(replay_products),
        replay_mean_total_wind_revision=mean_revision,
        signal_regime=regime,
        inferred_probability=inferred_probability,
        comparison_probability=comparison_probability,
        applicable_difference_percentage_points=applicable_difference,
        applicable_confidence_interval_95_percentage_points=applicable_interval,
        p_value=calibration_result.p_value,
        dropped_missing_calibration_rows=calibration_result.dropped_missing_count,
        inference=inference,
        ex_post_event_rate=ex_post_event_rate,
        ex_post_valid_observations=len(outcomes),
    )
