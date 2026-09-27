from dataclasses import FrozenInstanceError, asdict, replace
from datetime import date, datetime

import pandas as pd
import pytest

from compiler.variable_registry import VARIABLE_REGISTRY
from tools.backtest import run_backtest
from tools.data_tools import (
    DiagnosticSeverity,
    diagnose_data_quality,
    filter_absolute_date_range,
    filter_delivery_time_window,
    filter_relative_lookback,
    materialize_variables,
    normalize_datetime_variable,
)


def test_backtest_is_explicitly_a_placeholder() -> None:
    with pytest.raises(NotImplementedError):
        run_backtest()


def test_materialize_raw_variable_returns_copy_without_mutating_caller() -> None:
    source = pd.DataFrame({"id1_price_eur_mwh": [10.0, 20.0]})
    original = source.copy(deep=True)

    result = materialize_variables(source, ["id1_price_eur_mwh"])

    assert result is not source
    pd.testing.assert_frame_equal(source, original)
    pd.testing.assert_series_equal(
        result["id1_price_eur_mwh"],
        source["id1_price_eur_mwh"],
    )


def test_materialize_id1_minus_id3() -> None:
    source = pd.DataFrame(
        {
            "id1_price_eur_mwh": [30.0, 15.0],
            "id3_price_eur_mwh": [20.0, 18.0],
        }
    )

    result = materialize_variables(source, ["id1_minus_id3"])

    assert result["id1_minus_id3"].tolist() == [10.0, -3.0]


def test_materialize_nested_total_wind_revision_and_intermediates() -> None:
    source = pd.DataFrame(
        {
            "wind_onshore_a40": [100.0, 110.0],
            "wind_offshore_a40": [50.0, 55.0],
            "wind_onshore_a01": [90.0, 120.0],
            "wind_offshore_a01": [45.0, 50.0],
        }
    )

    result = materialize_variables(source, ["total_wind_revision"])

    assert result["total_wind_a40"].tolist() == [150.0, 165.0]
    assert result["total_wind_a01"].tolist() == [135.0, 170.0]
    assert result["total_wind_revision"].tolist() == [15.0, -5.0]


def test_materialize_multiple_variables() -> None:
    source = pd.DataFrame(
        {
            "id1_price_eur_mwh": [30.0],
            "id3_price_eur_mwh": [20.0],
            "solar_a40": [80.0],
            "solar_a01": [100.0],
        }
    )

    result = materialize_variables(
        source,
        ["id1_minus_id3", "solar_revision", "id1_minus_id3"],
    )

    assert result["id1_minus_id3"].tolist() == [10.0]
    assert result["solar_revision"].tolist() == [-20.0]


def test_materialize_rejects_unknown_requested_variable() -> None:
    with pytest.raises(ValueError, match="Unknown registry variable: unknown"):
        materialize_variables(pd.DataFrame(), ["unknown"])


def test_materialize_reports_all_missing_raw_columns() -> None:
    with pytest.raises(ValueError) as error:
        materialize_variables(pd.DataFrame(), ["id1_minus_id3"])

    message = str(error.value)
    assert "Missing required source columns" in message
    assert "id1_price_eur_mwh" in message
    assert "id3_price_eur_mwh" in message


def test_materialize_rejects_unknown_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    metadata = replace(
        VARIABLE_REGISTRY["id1_minus_id3"],
        dependencies=("missing_dependency",),
        formula="missing_dependency",
    )
    monkeypatch.setitem(VARIABLE_REGISTRY, "broken_derived", metadata)

    with pytest.raises(ValueError, match="Unknown dependency 'missing_dependency'"):
        materialize_variables(pd.DataFrame(), ["broken_derived"])


def test_materialize_detects_dependency_cycle(monkeypatch: pytest.MonkeyPatch) -> None:
    base = VARIABLE_REGISTRY["id1_minus_id3"]
    monkeypatch.setitem(
        VARIABLE_REGISTRY,
        "cycle_a",
        replace(base, dependencies=("cycle_b",), formula="cycle_b"),
    )
    monkeypatch.setitem(
        VARIABLE_REGISTRY,
        "cycle_b",
        replace(base, dependencies=("cycle_a",), formula="cycle_a"),
    )

    with pytest.raises(ValueError, match="Variable dependency cycle detected"):
        materialize_variables(pd.DataFrame(), ["cycle_a"])


def test_revision_uses_a40_minus_a01_order() -> None:
    source = pd.DataFrame({"solar_a40": [80.0], "solar_a01": [100.0]})

    result = materialize_variables(source, ["solar_revision"])

    assert result["solar_revision"].tolist() == [-20.0]


def test_aggregate_adds_all_dependencies() -> None:
    source = pd.DataFrame(
        {
            "solar_a40": [10.0],
            "wind_onshore_a40": [20.0],
            "wind_offshore_a40": [30.0],
        }
    )

    result = materialize_variables(source, ["wind_solar_a40"])

    assert result["wind_solar_a40"].tolist() == [60.0]


def test_deviation_uses_actual_minus_forecast_order() -> None:
    source = pd.DataFrame({"solar_actual_mw": [75.0], "solar_a40": [80.0]})

    result = materialize_variables(source, ["solar_a40_deviation"])

    assert result["solar_a40_deviation"].tolist() == [-5.0]


def test_materialize_datetime_components_and_season() -> None:
    source = pd.DataFrame(
        {
            "delivery_start_local": [
                "2026-01-05 03:15",
                "2026-04-12 11:30",
                "2026-07-19 18:45",
                "2026-10-25 23:00",
            ]
        }
    )

    result = materialize_variables(
        source,
        ["delivery_hour", "quarter_hour", "weekday", "month", "year", "season"],
    )

    assert result["delivery_hour"].tolist() == [3, 11, 18, 23]
    assert result["quarter_hour"].tolist() == [15, 30, 45, 0]
    assert result["weekday"].tolist() == [0, 6, 6, 6]
    assert result["month"].tolist() == [1, 4, 7, 10]
    assert result["year"].tolist() == [2026, 2026, 2026, 2026]
    assert result["season"].tolist() == ["winter", "spring", "summer", "autumn"]


def test_invalid_datetime_is_rejected() -> None:
    source = pd.DataFrame({"delivery_start_local": ["not-a-date"]})

    with pytest.raises(ValueError, match="Invalid datetime values"):
        materialize_variables(source, ["delivery_hour"])


def test_raw_variable_with_different_source_column_exposes_canonical_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = replace(
        VARIABLE_REGISTRY["id1_price_eur_mwh"],
        source_column="physical_id1",
    )
    monkeypatch.setitem(VARIABLE_REGISTRY, "canonical_id1", metadata)
    source = pd.DataFrame({"physical_id1": [12.0]})

    result = materialize_variables(source, ["canonical_id1"])

    assert result["physical_id1"].tolist() == [12.0]
    assert result["canonical_id1"].tolist() == [12.0]


def test_empty_variable_list_returns_unchanged_copy() -> None:
    source = pd.DataFrame({"unrelated": [1, 2]})

    result = materialize_variables(source, [])

    assert result is not source
    pd.testing.assert_frame_equal(result, source)


def test_unsupported_formula_syntax_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = replace(
        VARIABLE_REGISTRY["id1_minus_id3"],
        formula="id1_price_eur_mwh * id3_price_eur_mwh",
    )
    monkeypatch.setitem(VARIABLE_REGISTRY, "unsafe_formula", metadata)
    source = pd.DataFrame(
        {"id1_price_eur_mwh": [2.0], "id3_price_eur_mwh": [3.0]}
    )

    with pytest.raises(ValueError, match="Unsupported formula for variable"):
        materialize_variables(source, ["unsafe_formula"])


def test_normalize_utc_accepts_z_and_offsets_and_converts_to_utc() -> None:
    source = pd.DataFrame(
        {
            "delivery_start_utc": [
                "2026-01-01T12:00:00Z",
                "2026-01-01T13:00:00+01:00",
            ]
        }
    )

    result = normalize_datetime_variable(source, "delivery_start_utc")

    assert isinstance(result["delivery_start_utc"].dtype, pd.DatetimeTZDtype)
    assert str(result["delivery_start_utc"].dt.tz) == "UTC"
    assert result["delivery_start_utc"].iloc[0] == result["delivery_start_utc"].iloc[1]


def test_normalize_utc_rejects_naive_values() -> None:
    source = pd.DataFrame({"delivery_start_utc": ["2026-01-01 12:00"]})

    with pytest.raises(ValueError, match="requires timezone-aware"):
        normalize_datetime_variable(source, "delivery_start_utc")


def test_normalize_local_is_naive_and_does_not_mutate_caller() -> None:
    source = pd.DataFrame({"delivery_start_local": ["2026-01-01 12:00"]})
    original = source.copy(deep=True)

    result = normalize_datetime_variable(source, "delivery_start_local")

    assert result is not source
    assert pd.api.types.is_datetime64_dtype(result["delivery_start_local"].dtype)
    pd.testing.assert_frame_equal(source, original)


@pytest.mark.parametrize(
    "source_value",
    [
        "2026-01-01T12:00:00+01:00",
        "2026-07-01T18:00:00+02:00",
    ],
)
def test_normalize_local_offset_preserves_wall_clock(source_value: str) -> None:
    source = pd.DataFrame({"delivery_start_local": [source_value]})
    original = source.copy(deep=True)

    result = normalize_datetime_variable(source, "delivery_start_local")

    expected = pd.Timestamp(source_value).tz_localize(None)
    assert result["delivery_start_local"].iloc[0] == expected
    assert result["delivery_start_local"].dt.tz is None
    pd.testing.assert_frame_equal(source, original)


def test_normalize_local_supports_mixed_dst_offsets() -> None:
    source = pd.DataFrame(
        {
            "delivery_start_local": [
                "2026-01-01T18:00:00+01:00",
                "2026-07-01T18:00:00+02:00",
                "2026-08-01 19:00:00",
                None,
            ]
        }
    )

    result = normalize_datetime_variable(source, "delivery_start_local")

    assert result["delivery_start_local"].tolist()[:3] == [
        pd.Timestamp("2026-01-01 18:00:00"),
        pd.Timestamp("2026-07-01 18:00:00"),
        pd.Timestamp("2026-08-01 19:00:00"),
    ]
    assert pd.isna(result["delivery_start_local"].iloc[3])


def test_normalize_rejects_invalid_datetime_missing_column_and_wrong_variable() -> None:
    with pytest.raises(ValueError, match="Invalid datetime values"):
        normalize_datetime_variable(
            pd.DataFrame({"delivery_start_local": ["not-a-date"]}),
            "delivery_start_local",
        )
    with pytest.raises(ValueError, match="Missing required source columns"):
        normalize_datetime_variable(pd.DataFrame(), "delivery_start_local")
    with pytest.raises(ValueError, match="Unknown registry variable"):
        normalize_datetime_variable(pd.DataFrame(), "unknown")
    with pytest.raises(ValueError, match="not a raw datetime variable"):
        normalize_datetime_variable(
            pd.DataFrame({"id1_price_eur_mwh": [1.0]}),
            "id1_price_eur_mwh",
        )


def test_utc_normalization_behavior_is_unchanged_for_offset_values() -> None:
    source = pd.DataFrame(
        {"delivery_start_utc": ["2026-07-01T18:00:00+02:00"]}
    )

    result = normalize_datetime_variable(source, "delivery_start_utc")

    assert result["delivery_start_utc"].iloc[0] == pd.Timestamp(
        "2026-07-01T16:00:00Z"
    )


def test_delivery_hour_preserves_aware_local_source_hour() -> None:
    source = pd.DataFrame(
        {"delivery_start_local": ["2026-07-01T18:00:00+02:00"]}
    )

    result = materialize_variables(source, ["delivery_hour"])

    assert result["delivery_hour"].tolist() == [18]


def test_delivery_window_uses_wall_clock_from_aware_local_source() -> None:
    source = pd.DataFrame(
        {
            "delivery_start_local": [
                "2026-01-01T16:45:00+01:00",
                "2026-01-01T17:00:00+01:00",
                "2026-07-01T20:45:00+02:00",
                "2026-07-01T21:00:00+02:00",
            ]
        }
    )

    result = filter_delivery_time_window(
        source,
        "delivery_start_local",
        start_local="17:00",
        end_local="21:00",
    )

    assert result.index.tolist() == [1, 2]


def test_normalize_datetime_uses_alternate_registered_source_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        VARIABLE_REGISTRY,
        "delivery_start_local",
        replace(
            VARIABLE_REGISTRY["delivery_start_local"],
            source_column="physical_delivery_local",
        ),
    )
    source = pd.DataFrame({"physical_delivery_local": ["2026-01-01 12:00"]})

    result = normalize_datetime_variable(source, "delivery_start_local")

    assert "physical_delivery_local" in result
    assert "delivery_start_local" in result


def test_absolute_date_range_is_calendar_inclusive_and_half_open() -> None:
    source = pd.DataFrame(
        {
            "delivery_start_local": [
                "2025-12-31 23:45",
                "2026-01-01 00:00",
                "2026-01-02 23:45",
                "2026-01-03 00:00",
            ]
        },
        index=[10, 11, 12, 13],
    )

    result = filter_absolute_date_range(
        source,
        "delivery_start_local",
        start_date="2026-01-01",
        end_date="2026-01-02",
    )

    assert result.index.tolist() == [11, 12]


def test_absolute_date_range_supports_open_boundaries() -> None:
    source = pd.DataFrame(
        {"delivery_start_local": ["2026-01-01", "2026-01-02", "2026-01-03"]}
    )

    from_start = filter_absolute_date_range(
        source, "delivery_start_local", start_date="2026-01-02"
    )
    through_end = filter_absolute_date_range(
        source, "delivery_start_local", end_date="2026-01-02"
    )

    assert from_start.index.tolist() == [1, 2]
    assert through_end.index.tolist() == [0, 1]


def test_absolute_date_range_validates_boundaries() -> None:
    source = pd.DataFrame({"delivery_start_local": ["2026-01-01"]})

    with pytest.raises(ValueError, match="At least one"):
        filter_absolute_date_range(source, "delivery_start_local")
    with pytest.raises(ValueError, match="must not be after"):
        filter_absolute_date_range(
            source,
            "delivery_start_local",
            start_date="2026-01-02",
            end_date="2026-01-01",
        )


def test_absolute_date_range_supports_utc_instants() -> None:
    source = pd.DataFrame(
        {
            "delivery_start_utc": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T23:45:00Z",
                "2026-01-02T00:00:00Z",
            ]
        }
    )

    result = filter_absolute_date_range(
        source,
        "delivery_start_utc",
        start_date="2026-01-01",
        end_date="2026-01-01",
    )

    assert result.index.tolist() == [0, 1]


@pytest.mark.parametrize(
    ("unit", "value", "expected_indices"),
    [
        ("days", 1, [3, 4]),
        ("weeks", 1, [2, 3, 4]),
        ("months", 1, [1, 2, 3, 4]),
        ("years", 1, [0, 1, 2, 3, 4]),
    ],
)
def test_relative_lookback_units(
    unit: str,
    value: int,
    expected_indices: list[int],
) -> None:
    source = pd.DataFrame(
        {
            "delivery_start_local": [
                "2025-05-31",
                "2026-04-30",
                "2026-05-24",
                "2026-05-30",
                "2026-05-31 23:45",
            ]
        }
    )

    result = filter_relative_lookback(
        source,
        "delivery_start_local",
        as_of_date=date(2026, 5, 31),
        lookback_value=value,
        lookback_unit=unit,
    )

    assert result.index.tolist() == expected_indices


def test_relative_lookback_month_end_and_leap_year_clamping() -> None:
    month_source = pd.DataFrame(
        {"delivery_start_local": ["2026-04-29", "2026-04-30", "2026-05-31"]}
    )
    leap_source = pd.DataFrame(
        {"delivery_start_local": ["2023-02-27", "2023-02-28", "2024-02-29"]}
    )

    month_result = filter_relative_lookback(
        month_source,
        "delivery_start_local",
        as_of_date=date(2026, 5, 31),
        lookback_value=1,
        lookback_unit="months",
    )
    leap_result = filter_relative_lookback(
        leap_source,
        "delivery_start_local",
        as_of_date=date(2024, 2, 29),
        lookback_value=1,
        lookback_unit="years",
    )

    assert month_result.index.tolist() == [1, 2]
    assert leap_result.index.tolist() == [1, 2]


def test_relative_lookback_validates_contract() -> None:
    source = pd.DataFrame({"delivery_start_local": ["2026-01-01"]})

    for invalid in (0, -1, True):
        with pytest.raises(ValueError, match="positive integer"):
            filter_relative_lookback(
                source,
                "delivery_start_local",
                as_of_date=date(2026, 1, 1),
                lookback_value=invalid,
                lookback_unit="days",
            )
    with pytest.raises(ValueError, match="lookback_unit"):
        filter_relative_lookback(
            source,
            "delivery_start_local",
            as_of_date=date(2026, 1, 1),
            lookback_value=1,
            lookback_unit="hours",
        )
    with pytest.raises(ValueError, match="datetime.date"):
        filter_relative_lookback(
            source,
            "delivery_start_local",
            as_of_date=datetime(2026, 1, 1),
            lookback_value=1,
            lookback_unit="days",
        )


def test_delivery_window_is_start_inclusive_and_end_exclusive() -> None:
    source = pd.DataFrame(
        {
            "delivery_start_local": [
                "2026-01-01 16:45",
                "2026-01-01 17:00",
                "2026-01-01 20:45",
                "2026-01-01 21:00",
            ]
        }
    )

    result = filter_delivery_time_window(
        source,
        "delivery_start_local",
        start_local="17:00",
        end_local="21:00",
    )

    assert result.index.tolist() == [1, 2]


def test_recurring_delivery_window_crosses_midnight() -> None:
    source = pd.DataFrame(
        {
            "delivery_start_local": [
                "2026-01-01 21:45",
                "2026-01-01 22:00",
                "2026-01-01 23:45",
                "2026-01-02 00:00",
                "2026-01-02 01:45",
                "2026-01-02 02:00",
            ]
        }
    )

    result = filter_delivery_time_window(
        source,
        "delivery_start_local",
        start_local="22:00",
        end_local="02:00",
    )

    assert result.index.tolist() == [1, 2, 3, 4]


def test_dated_delivery_window_crosses_midnight_and_excludes_adjacent_dates() -> None:
    source = pd.DataFrame(
        {
            "delivery_start_local": [
                "2025-12-31 23:00",
                "2026-01-01 22:00",
                "2026-01-01 23:45",
                "2026-01-02 01:45",
                "2026-01-02 02:00",
                "2026-01-02 23:00",
            ]
        }
    )

    result = filter_delivery_time_window(
        source,
        "delivery_start_local",
        start_local="22:00",
        end_local="02:00",
        delivery_date="2026-01-01",
    )

    assert result.index.tolist() == [1, 2, 3]


def test_delivery_window_validates_times_and_requires_local_semantics() -> None:
    local = pd.DataFrame({"delivery_start_local": ["2026-01-01 12:00"]})
    utc = pd.DataFrame({"delivery_start_utc": ["2026-01-01T12:00:00Z"]})

    with pytest.raises(ValueError, match="must differ"):
        filter_delivery_time_window(
            local,
            "delivery_start_local",
            start_local="12:00",
            end_local="12:00",
        )
    with pytest.raises(ValueError, match="HH:MM"):
        filter_delivery_time_window(
            local,
            "delivery_start_local",
            start_local="25:00",
            end_local="12:00",
        )
    with pytest.raises(ValueError, match="delivery_start_local semantics"):
        filter_delivery_time_window(
            utc,
            "delivery_start_utc",
            start_local="10:00",
            end_local="12:00",
        )


def test_delivery_filter_rejects_invalid_local_timestamp_and_preserves_caller() -> None:
    invalid = pd.DataFrame({"delivery_start_local": ["invalid"]})
    source = pd.DataFrame({"delivery_start_local": ["2026-01-01 12:00"]})
    original = source.copy(deep=True)

    with pytest.raises(ValueError, match="Invalid datetime values"):
        filter_delivery_time_window(
            invalid,
            "delivery_start_local",
            start_local="10:00",
            end_local="14:00",
        )
    filter_delivery_time_window(
        source,
        "delivery_start_local",
        start_local="10:00",
        end_local="14:00",
    )
    pd.testing.assert_frame_equal(source, original)


def test_data_quality_empty_report_is_immutable_serializable_and_non_mutating() -> None:
    source = pd.DataFrame(columns=["empty"])
    original = source.copy(deep=True)

    report = diagnose_data_quality(source)

    assert report.row_count == 0
    assert report.column_count == 1
    assert report.fully_empty_columns == ("empty",)
    assert report.constant_columns == ()
    assert asdict(report)["columns"][0]["column"] == "empty"
    with pytest.raises(FrozenInstanceError):
        report.row_count = 1  # type: ignore[misc]
    pd.testing.assert_frame_equal(source, original)


def test_column_quality_reports_missing_empty_constant_and_infinities() -> None:
    source = pd.DataFrame(
        {
            "missing": [1.0, None, 3.0, None],
            "empty": [None, None, None, None],
            "constant": [7, 7, 7, 7],
            "numeric": [float("inf"), float("-inf"), 0.0, 1.0],
            "boolean": [True, False, True, False],
            "text": ["x", "y", "z", "w"],
        }
    )

    report = diagnose_data_quality(source)
    columns = {item.column: item for item in report.columns}

    assert columns["missing"].missing_count == 2
    assert columns["missing"].missing_ratio == 0.5
    assert columns["empty"].fully_empty is True
    assert columns["empty"].constant is False
    assert columns["constant"].constant is True
    assert columns["numeric"].positive_infinity_count == 1
    assert columns["numeric"].negative_infinity_count == 1
    assert columns["boolean"].positive_infinity_count == 0
    assert columns["text"].negative_infinity_count == 0


def test_full_row_and_supplied_key_duplicates_are_separate() -> None:
    source = pd.DataFrame(
        {
            "delivery_start_local": ["2026-01-01 00:00"] * 3,
            "product": ["A", "A", "B"],
            "value": [1, 1, 2],
        }
    )

    report = diagnose_data_quality(
        source,
        key_columns=("delivery_start_local", "product"),
        datetime_variable="delivery_start_local",
    )

    assert report.duplicate_full_row_count == 2
    assert report.duplicate_full_row_excess_count == 1
    assert report.duplicate_keys is not None
    assert report.duplicate_keys.duplicate_row_count == 2
    assert report.duplicate_keys.duplicate_excess_count == 1
    assert report.datetime_quality is not None
    assert report.datetime_quality.duplicate_timestamp_count == 3


def test_missing_supplied_key_column_is_reported_as_error() -> None:
    report = diagnose_data_quality(
        pd.DataFrame({"value": [1]}),
        key_columns=("missing_key",),
    )

    issue = next(item for item in report.issues if item.code == "missing_key_columns")
    assert issue.severity is DiagnosticSeverity.ERROR


def test_empty_key_name_is_a_programming_error() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        diagnose_data_quality(pd.DataFrame(), key_columns=("",))


def test_requested_materialization_is_checked_and_unknown_name_rejected() -> None:
    present = diagnose_data_quality(
        pd.DataFrame({"id1_minus_id3": [1.0]}),
        requested_variables=("id1_minus_id3",),
    )
    missing = diagnose_data_quality(
        pd.DataFrame({"id1_price_eur_mwh": [1.0]}),
        requested_variables=("id1_minus_id3",),
    )

    assert present.missing_requested_variables == ()
    assert missing.missing_requested_variables == ("id1_minus_id3",)
    assert any(
        issue.code == "missing_requested_variables"
        and issue.severity is DiagnosticSeverity.ERROR
        for issue in missing.issues
    )
    with pytest.raises(ValueError, match="Unknown registry variable"):
        diagnose_data_quality(pd.DataFrame(), requested_variables=("unknown",))


def test_datetime_coverage_order_and_utc_context() -> None:
    local = diagnose_data_quality(
        pd.DataFrame(
            {
                "delivery_start_local": [
                    "2026-01-01 00:15",
                    "2026-01-01 00:00",
                ]
            }
        ),
        datetime_variable="delivery_start_local",
    )
    utc = diagnose_data_quality(
        pd.DataFrame(
            {
                "delivery_start_utc": [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:15:00+00:00",
                ]
            }
        ),
        datetime_variable="delivery_start_utc",
    )

    assert local.datetime_quality is not None
    assert local.datetime_quality.earliest == "2026-01-01T00:00:00"
    assert local.datetime_quality.latest == "2026-01-01T00:15:00"
    assert local.datetime_quality.is_non_decreasing is False
    assert utc.datetime_quality is not None
    assert utc.datetime_quality.timezone_context == "utc_absolute"
    assert utc.datetime_quality.earliest.endswith("+00:00")


def test_true_missing_timestamps_are_warnings_not_parse_errors() -> None:
    report = diagnose_data_quality(
        pd.DataFrame(
            {"delivery_start_local": [None, "2026-01-01 00:00", pd.NaT]}
        ),
        datetime_variable="delivery_start_local",
    )

    assert report.datetime_quality is not None
    assert report.datetime_quality.missing_count == 2
    assert any(
        issue.code == "missing_timestamps"
        and issue.severity is DiagnosticSeverity.WARNING
        for issue in report.issues
    )
    assert not any(issue.code == "malformed_timestamps" for issue in report.issues)


def test_all_missing_timestamps_have_no_coverage() -> None:
    report = diagnose_data_quality(
        pd.DataFrame({"delivery_start_local": [None, pd.NaT]}),
        datetime_variable="delivery_start_local",
    )

    assert report.datetime_quality is not None
    assert report.datetime_quality.earliest is None
    assert report.datetime_quality.latest is None
    assert report.datetime_quality.is_non_decreasing is None
    assert any(
        issue.code == "datetime_coverage_unavailable"
        and issue.severity is DiagnosticSeverity.ERROR
        for issue in report.issues
    )


@pytest.mark.parametrize(
    ("values", "variable"),
    [
        (["malformed"], "delivery_start_local"),
        (["2026-01-01 00:00"], "delivery_start_utc"),
    ],
)
def test_malformed_non_null_or_timezone_invalid_timestamps_are_errors(
    values: list[str],
    variable: str,
) -> None:
    report = diagnose_data_quality(
        pd.DataFrame({variable: values}),
        datetime_variable=variable,
    )

    assert report.datetime_quality is not None
    assert report.datetime_quality.earliest is None
    assert any(
        issue.code == "malformed_timestamps"
        and issue.severity is DiagnosticSeverity.ERROR
        for issue in report.issues
    )


def test_resolution_check_is_opt_in() -> None:
    source = pd.DataFrame(
        {
            "delivery_start_local": [
                "2026-01-01 00:00",
                "2026-01-01 01:00",
            ],
            "id1_price_eur_mwh": [1.0, 2.0],
        }
    )

    report = diagnose_data_quality(
        source,
        requested_variables=("id1_price_eur_mwh",),
        datetime_variable="delivery_start_local",
        resolution_variables=None,
    )

    assert report.datetime_quality is not None
    assert report.datetime_quality.expected_resolution is None
    assert report.datetime_quality.gap_count == 0
    assert not any(issue.code == "timestamp_gaps" for issue in report.issues)


def test_filtered_delivery_hours_only_warn_when_continuity_is_requested() -> None:
    source = pd.DataFrame(
        {
            "delivery_start_local": [
                "2026-01-01 17:00",
                "2026-01-01 17:15",
                "2026-01-02 17:00",
            ],
            "id1_price_eur_mwh": [1.0, 2.0, 3.0],
        }
    )

    default = diagnose_data_quality(
        source,
        datetime_variable="delivery_start_local",
    )
    explicit = diagnose_data_quality(
        source,
        datetime_variable="delivery_start_local",
        resolution_variables=("id1_price_eur_mwh",),
    )

    assert default.datetime_quality is not None
    assert default.datetime_quality.gap_count == 0
    assert explicit.datetime_quality is not None
    assert explicit.datetime_quality.gap_count == 1


def test_gap_count_and_missing_interval_count_are_distinct() -> None:
    report = diagnose_data_quality(
        pd.DataFrame(
            {
                "delivery_start_local": [
                    "2026-01-01 10:00",
                    "2026-01-01 10:15",
                    "2026-01-01 11:00",
                ]
            }
        ),
        datetime_variable="delivery_start_local",
        resolution_variables=("id1_price_eur_mwh",),
    )

    assert report.datetime_quality is not None
    assert report.datetime_quality.gap_count == 1
    assert report.datetime_quality.missing_interval_count == 2
    assert report.datetime_quality.irregular_interval_count == 0


def test_non_multiple_oversized_interval_is_gap_and_irregular_without_invention() -> None:
    report = diagnose_data_quality(
        pd.DataFrame(
            {
                "delivery_start_local": [
                    "2026-01-01 10:00",
                    "2026-01-01 10:40",
                ]
            }
        ),
        datetime_variable="delivery_start_local",
        resolution_variables=("id1_price_eur_mwh",),
    )

    assert report.datetime_quality is not None
    assert report.datetime_quality.gap_count == 1
    assert report.datetime_quality.missing_interval_count == 0
    assert report.datetime_quality.irregular_interval_count == 1


def test_short_irregular_interval_duplicates_and_unsorted_data_are_separate() -> None:
    report = diagnose_data_quality(
        pd.DataFrame(
            {
                "delivery_start_local": [
                    "2026-01-01 10:10",
                    "2026-01-01 10:00",
                    "2026-01-01 10:00",
                    "2026-01-01 10:15",
                ],
                "value": [1, 2, 3, 4],
            }
        ),
        datetime_variable="delivery_start_local",
        resolution_variables=("id1_price_eur_mwh",),
    )

    assert report.duplicate_full_row_count == 0
    assert report.datetime_quality is not None
    assert report.datetime_quality.duplicate_timestamp_count == 2
    assert report.datetime_quality.is_non_decreasing is False
    assert report.datetime_quality.gap_count == 0
    assert report.datetime_quality.irregular_interval_count == 2


def test_resolution_configuration_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    source = pd.DataFrame({"delivery_start_local": ["2026-01-01 00:00"]})
    with pytest.raises(ValueError, match="non-empty"):
        diagnose_data_quality(
            source,
            datetime_variable="delivery_start_local",
            resolution_variables=(),
        )
    with pytest.raises(ValueError, match="Unknown registry variable"):
        diagnose_data_quality(
            source,
            datetime_variable="delivery_start_local",
            resolution_variables=("unknown",),
        )
    with pytest.raises(ValueError, match="datetime_variable is required"):
        diagnose_data_quality(
            source,
            resolution_variables=("id1_price_eur_mwh",),
        )

    monkeypatch.setitem(
        VARIABLE_REGISTRY,
        "different_resolution",
        replace(VARIABLE_REGISTRY["id1_price_eur_mwh"], resolution="1h"),
    )
    conflicting = diagnose_data_quality(
        source,
        datetime_variable="delivery_start_local",
        resolution_variables=("id1_price_eur_mwh", "different_resolution"),
    )
    unavailable = diagnose_data_quality(
        source,
        datetime_variable="delivery_start_local",
        resolution_variables=("delivery_start_local",),
    )
    assert any(issue.code == "conflicting_resolutions" for issue in conflicting.issues)
    assert any(issue.code == "resolution_unavailable" for issue in unavailable.issues)


def test_local_irregularities_report_dst_classification_limit() -> None:
    report = diagnose_data_quality(
        pd.DataFrame(
            {
                "delivery_start_local": [
                    "2026-03-29 01:45",
                    "2026-03-29 03:00",
                ]
            }
        ),
        datetime_variable="delivery_start_local",
        resolution_variables=("id1_price_eur_mwh",),
    )

    assert any(
        issue.code == "local_time_dst_classification_unavailable"
        for issue in report.issues
    )
