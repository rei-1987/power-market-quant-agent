import pandas as pd
import pytest

import app.agent_runtime as runtime_module
from app.agent_runtime import AgentRuntime
from app.mode_router import RouteDecision
from execution.live_forecast_pipeline import run_live_forecast
from schemas.hypothesis_schema import ExecutionMode


REQUEST = "What does the latest wind forecast revision imply for tonight's intraday market?"


def _live_data(*, current_revision: float, current_events: list[bool]) -> pd.DataFrame:
    local_values = []
    for day in ("2026-09-11", "2026-09-12"):
        local_values.extend(f"{day}T{hour:02d}:00:00+02:00" for hour in (17, 18, 19, 20))
    local_values.extend(
        f"2026-09-13T{hour:02d}:00:00+02:00" for hour in (17, 18, 19, 20)
    )
    calibration_revisions = [-10, -10, 10, 10, -10, -10, 10, 10]
    calibration_events = [True, True, True, False, True, False, False, False]
    revisions = calibration_revisions + [current_revision] * 4
    events = calibration_events + current_events
    local = pd.to_datetime(local_values, format="mixed")
    utc = [item.tz_convert("UTC").isoformat() for item in local]
    return pd.DataFrame(
        {
            "delivery_start_local": local_values,
            "delivery_start_utc": utc,
            "id1_price_eur_mwh": [51.0 if item else 49.0 for item in events],
            "id3_price_eur_mwh": [50.0] * 12,
            "wind_onshore_a01": [100.0] * 12,
            "wind_offshore_a01": [50.0] * 12,
            "wind_onshore_a40": [100.0 + item for item in revisions],
            "wind_offshore_a40": [50.0] * 12,
            "solar_actual_mw": [999.0] * 12,
            "wind_onshore_actual_mw": [999.0] * 12,
            "wind_offshore_actual_mw": [999.0] * 12,
        }
    )


def test_live_negative_regime_uses_latest_date_at_0800_and_excludes_current_outcomes() -> None:
    positive_ex_post = run_live_forecast(
        REQUEST,
        _live_data(current_revision=-20, current_events=[True] * 4),
    )
    negative_ex_post_source = _live_data(
        current_revision=-20, current_events=[False] * 4
    )
    current = negative_ex_post_source["delivery_start_local"].str.startswith("2026-09-13")
    negative_ex_post_source.loc[current, "solar_actual_mw"] = -999999
    negative_ex_post = run_live_forecast(REQUEST, negative_ex_post_source)

    assert positive_ex_post.current_delivery_date.isoformat() == "2026-09-13"
    assert positive_ex_post.simulated_current_local == pd.Timestamp("2026-09-13 08:00")
    assert positive_ex_post.historical_calibration_observations == 8
    assert positive_ex_post.current_product_count == 4
    assert positive_ex_post.current_mean_total_wind_revision == -20.0
    assert positive_ex_post.signal_regime == "negative"
    assert positive_ex_post.inferred_probability == 0.75
    assert positive_ex_post.comparison_probability == 0.25
    assert positive_ex_post.applicable_difference_percentage_points == 50.0
    assert positive_ex_post.inferred_probability == negative_ex_post.inferred_probability
    assert positive_ex_post.ex_post_demo_event_rate == 1.0
    assert negative_ex_post.ex_post_demo_event_rate == 0.0
    assert "simulated live" in positive_ex_post.interpretation


def test_live_non_negative_regime_reverses_applicable_effect_and_interval() -> None:
    negative = run_live_forecast(
        REQUEST,
        _live_data(current_revision=-1, current_events=[True] * 4),
    )
    result = run_live_forecast(
        REQUEST,
        _live_data(current_revision=10, current_events=[True, False, True, False]),
    )

    assert result.signal_regime == "non-negative"
    assert result.inferred_probability == 0.25
    assert result.comparison_probability == 0.75
    assert result.applicable_difference_percentage_points == -50.0
    assert result.applicable_confidence_interval_95_percentage_points == pytest.approx(
        (
            -negative.applicable_confidence_interval_95_percentage_points[1],
            -negative.applicable_confidence_interval_95_percentage_points[0],
        )
    )
    assert result.ex_post_demo_event_rate == 0.5


def test_live_signal_is_materialized_only_from_current_a01_a40_inputs() -> None:
    source = _live_data(current_revision=-7.5, current_events=[True] * 4)
    assert "total_wind_revision" not in source.columns

    result = run_live_forecast(REQUEST, source)

    assert result.current_mean_total_wind_revision == -7.5


def test_runtime_routes_live_forecast_to_real_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live_result = run_live_forecast(
        REQUEST,
        _live_data(current_revision=-5, current_events=[True] * 4),
    )
    monkeypatch.setattr(
        runtime_module,
        "route_request",
        lambda request: RouteDecision(ExecutionMode.LIVE_FORECAST, "explicit live"),
    )
    monkeypatch.setattr(
        runtime_module,
        "resolve_master_dataset_path",
        lambda: "master.csv",
    )
    monkeypatch.setattr(
        runtime_module,
        "run_live_forecast",
        lambda request, path: live_result,
    )

    result = AgentRuntime().run(REQUEST)

    assert result.status == "live_complete"
    assert result.data_summary["live_result"]["signal_regime"] == "negative"
    assert result.events[-1].label == "LIVE QUANT"
