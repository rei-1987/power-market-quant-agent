import pandas as pd
import pytest

import app.agent_runtime as runtime_module
from app.agent_runtime import AgentRuntime
from app.mode_router import RouteDecision
from execution.historical_replay_pipeline import (
    parse_replay_timestamp,
    run_historical_replay,
)
from schemas.hypothesis_schema import ExecutionMode


REQUEST = "Replay 2025-06-12 from 08:00 and show what the agent would have known."


def _replay_data(*, replay_revision: float, replay_events: list[bool]) -> pd.DataFrame:
    local_values = []
    for day in ("2025-06-10", "2025-06-11"):
        local_values.extend(f"{day}T{hour:02d}:00:00+02:00" for hour in (17, 18, 19, 20))
    local_values.extend(
        f"2025-06-12T{hour:02d}:00:00+02:00" for hour in (17, 18, 19, 20)
    )
    local_values.extend(
        f"2025-06-13T{hour:02d}:00:00+02:00" for hour in (17, 18, 19, 20)
    )

    calibration_revisions = [-10, -10, 10, 10, -10, -10, 10, 10]
    calibration_events = [True, True, True, False, True, False, False, False]
    revisions = calibration_revisions + [replay_revision] * 4 + [-999] * 4
    events = calibration_events + replay_events + [True] * 4
    id3 = [50.0] * len(local_values)
    id1 = [51.0 if event else 49.0 for event in events]

    local = pd.to_datetime(local_values, format="mixed")
    utc = [item.tz_convert("UTC").isoformat() for item in local]
    return pd.DataFrame(
        {
            "delivery_start_local": local_values,
            "delivery_start_utc": utc,
            "id1_price_eur_mwh": id1,
            "id3_price_eur_mwh": id3,
            "wind_onshore_a01": [100.0] * len(local_values),
            "wind_offshore_a01": [50.0] * len(local_values),
            "wind_onshore_a40": [100.0 + value for value in revisions],
            "wind_offshore_a40": [50.0] * len(local_values),
        }
    )


def test_parse_replay_timestamp() -> None:
    assert parse_replay_timestamp(REQUEST) == pd.Timestamp("2025-06-12 08:00")


def test_negative_signal_uses_pre_date_conditional_probability_and_separates_ex_post() -> None:
    positive_outcomes = run_historical_replay(
        REQUEST,
        _replay_data(replay_revision=-20, replay_events=[True] * 4),
    )
    negative_outcomes = run_historical_replay(
        REQUEST,
        _replay_data(replay_revision=-20, replay_events=[False] * 4),
    )

    assert positive_outcomes.historical_calibration_observations == 8
    assert positive_outcomes.replay_product_count == 4
    assert positive_outcomes.replay_mean_total_wind_revision == -20.0
    assert positive_outcomes.signal_regime == "negative"
    assert positive_outcomes.inferred_probability == 0.75
    assert positive_outcomes.comparison_probability == 0.25
    assert positive_outcomes.applicable_difference_percentage_points == 50.0
    assert positive_outcomes.applicable_confidence_interval_95_percentage_points[0] < 50.0
    assert positive_outcomes.inferred_probability == negative_outcomes.inferred_probability
    assert positive_outcomes.ex_post_event_rate == 1.0
    assert negative_outcomes.ex_post_event_rate == 0.0
    assert "not used" in positive_outcomes.inference


def test_non_negative_signal_uses_complement_probability() -> None:
    negative = run_historical_replay(
        REQUEST,
        _replay_data(replay_revision=-1, replay_events=[True] * 4),
    )
    result = run_historical_replay(
        REQUEST,
        _replay_data(replay_revision=5, replay_events=[True, False, True, False]),
    )

    assert result.replay_mean_total_wind_revision == 5.0
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
    assert result.ex_post_event_rate == 0.5


def test_future_delivery_rows_never_enter_calibration() -> None:
    source = _replay_data(replay_revision=-5, replay_events=[True] * 4)
    result = run_historical_replay(REQUEST, source)

    future = source["delivery_start_local"].str.startswith("2025-06-13")
    source.loc[future, "id1_price_eur_mwh"] = -10000
    source.loc[future, "wind_onshore_a40"] = -10000
    changed = run_historical_replay(REQUEST, source)

    assert changed.historical_calibration_observations == 8
    assert changed.inferred_probability == result.inferred_probability
    assert changed.applicable_difference_percentage_points == (
        result.applicable_difference_percentage_points
    )


def test_replay_signal_is_materialized_from_a01_and_a40_inputs() -> None:
    source = _replay_data(replay_revision=-7.5, replay_events=[True] * 4)
    assert "total_wind_revision" not in source.columns

    result = run_historical_replay(REQUEST, source)

    assert result.replay_mean_total_wind_revision == -7.5


def test_replay_requires_explicit_supported_timestamp() -> None:
    with pytest.raises(ValueError, match="Replay YYYY-MM-DD"):
        parse_replay_timestamp("Replay yesterday")
    with pytest.raises(ValueError, match="08:00 only"):
        run_historical_replay(
            "Replay 2025-06-12 from 09:00",
            _replay_data(replay_revision=-1, replay_events=[True] * 4),
        )


def test_runtime_routes_historical_replay_to_real_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_result = run_historical_replay(
        REQUEST,
        _replay_data(replay_revision=-5, replay_events=[True] * 4),
    )
    monkeypatch.setattr(
        runtime_module,
        "route_request",
        lambda request: RouteDecision(ExecutionMode.HISTORICAL_REPLAY, "explicit replay"),
    )
    monkeypatch.setattr(
        runtime_module,
        "resolve_master_dataset_path",
        lambda: "master.csv",
    )
    monkeypatch.setattr(
        runtime_module,
        "run_historical_replay",
        lambda request, path: replay_result,
    )

    result = AgentRuntime().run(REQUEST)

    assert result.status == "replay_complete"
    assert result.data_summary["replay_result"]["signal_regime"] == "negative"
    assert result.events[-1].label == "REPLAY QUANT"
    assert result.events[-1].status == "completed"
