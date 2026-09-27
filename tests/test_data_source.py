from pathlib import Path

import pandas as pd
import pytest

from tools.data_source import (
    latest_available_timestamp,
    load_master_dataset,
    resolve_master_dataset_path,
)


def _write_csv(path: Path) -> Path:
    pd.DataFrame(
        {
            "delivery_start_utc": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:15:00Z",
            ],
            "delivery_start_local": [
                "2026-01-01 01:00",
                "2026-01-01 01:15",
            ],
            "id1_price_eur_mwh": [30.0, 40.0],
            "id3_price_eur_mwh": [20.0, 25.0],
            "solar_a40": [80.0, 90.0],
            "solar_a01": [100.0, 95.0],
            "wind_onshore_a40": [100.0, 110.0],
            "wind_offshore_a40": [50.0, 55.0],
            "wind_onshore_a01": [90.0, 120.0],
            "wind_offshore_a01": [45.0, 50.0],
        }
    ).to_csv(path, index=False)
    return path


def test_resolve_path_from_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = _write_csv(tmp_path / "master.csv")
    monkeypatch.setenv("MARKET_DATA_CSV", str(csv_path))

    assert resolve_master_dataset_path() == csv_path.resolve()


@pytest.mark.parametrize("configured", [None, "", "   "])
def test_missing_or_blank_environment_has_no_hidden_fallback(
    configured: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if configured is None:
        monkeypatch.delenv("MARKET_DATA_CSV", raising=False)
    else:
        monkeypatch.setenv("MARKET_DATA_CSV", configured)

    with pytest.raises(RuntimeError, match="MARKET_DATA_CSV is not set or is blank"):
        resolve_master_dataset_path()


def test_nonexistent_file_and_directory_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_master_dataset_path(path_override=tmp_path / "missing.csv")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_master_dataset_path(path_override=tmp_path)


def test_override_has_priority_and_environment_is_not_mutated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured = _write_csv(tmp_path / "configured.csv")
    override = _write_csv(tmp_path / "override.csv")
    monkeypatch.setenv("MARKET_DATA_CSV", str(configured))

    assert resolve_master_dataset_path(path_override=override) == override.resolve()
    assert monkeypatch.context is not None
    assert Path(str(configured)) == configured
    assert resolve_master_dataset_path() == configured.resolve()


def test_blank_override_is_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARKET_DATA_CSV", "ignored.csv")

    with pytest.raises(ValueError, match="path_override must not be blank"):
        resolve_master_dataset_path(path_override="   ")


def test_relative_path_uses_current_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = _write_csv(tmp_path / "relative.csv")
    monkeypatch.chdir(tmp_path)

    assert resolve_master_dataset_path(path_override="relative.csv") == csv_path.resolve()


def test_expand_user_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    csv_path = _write_csv(home / "master.csv")
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HOME", str(home))

    assert resolve_master_dataset_path(path_override="~/master.csv") == csv_path.resolve()


def test_variables_none_loads_complete_raw_csv_without_derived_materialization(
    tmp_path: Path,
) -> None:
    csv_path = _write_csv(tmp_path / "master.csv")

    result = load_master_dataset(path_override=csv_path, materialize=True)

    assert "id1_price_eur_mwh" in result.columns
    assert "id1_minus_id3" not in result.columns
    assert "solar_revision" not in result.columns


def test_requested_raw_variable_loads_only_its_source_column(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "master.csv")

    result = load_master_dataset(
        ["id1_price_eur_mwh"],
        path_override=csv_path,
    )

    assert result.columns.tolist() == ["id1_price_eur_mwh"]


def test_requested_derived_variable_loads_dependencies_and_materializes_only_request(
    tmp_path: Path,
) -> None:
    csv_path = _write_csv(tmp_path / "master.csv")

    result = load_master_dataset(
        ["id1_minus_id3"],
        path_override=csv_path,
        materialize=True,
    )

    assert result.columns.tolist() == [
        "id1_price_eur_mwh",
        "id3_price_eur_mwh",
        "id1_minus_id3",
    ]
    assert result["id1_minus_id3"].tolist() == [10.0, 15.0]
    assert "solar_revision" not in result.columns


def test_nested_requested_variable_materializes_intermediate_dependencies(
    tmp_path: Path,
) -> None:
    csv_path = _write_csv(tmp_path / "master.csv")

    result = load_master_dataset(
        ["total_wind_revision"],
        path_override=csv_path,
    )

    assert "total_wind_a40" in result.columns
    assert "total_wind_a01" in result.columns
    assert result["total_wind_revision"].tolist() == [15.0, -5.0]


def test_materialize_false_returns_only_required_raw_columns(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "master.csv")

    result = load_master_dataset(
        ["id1_minus_id3"],
        path_override=csv_path,
        materialize=False,
    )

    assert result.columns.tolist() == ["id1_price_eur_mwh", "id3_price_eur_mwh"]


def test_empty_and_unknown_variable_requests_are_rejected(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "master.csv")

    with pytest.raises(ValueError, match="non-empty"):
        load_master_dataset([], path_override=csv_path)
    with pytest.raises(ValueError, match="Unknown registry variable"):
        load_master_dataset(["unknown"], path_override=csv_path)


def test_missing_required_csv_column_is_reported(tmp_path: Path) -> None:
    csv_path = tmp_path / "incomplete.csv"
    pd.DataFrame({"id1_price_eur_mwh": [1.0]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="id3_price_eur_mwh"):
        load_master_dataset(["id1_minus_id3"], path_override=csv_path)


def test_latest_utc_timestamp_is_maximum_aware_and_ignores_missing() -> None:
    source = pd.DataFrame(
        {
            "delivery_start_utc": [
                "2026-01-02T00:00:00Z",
                None,
                "2026-01-03T01:00:00+01:00",
                "2026-01-01T00:00:00Z",
            ]
        }
    )
    original = source.copy(deep=True)

    latest = latest_available_timestamp(source)

    assert latest == pd.Timestamp("2026-01-03T00:00:00Z")
    assert str(latest.tz) == "UTC"
    pd.testing.assert_frame_equal(source, original)


def test_latest_local_timestamp_preserves_naive_semantics() -> None:
    source = pd.DataFrame(
        {"delivery_start_local": ["2026-01-01 12:00", "2026-01-01 13:00"]}
    )

    latest = latest_available_timestamp(source, "delivery_start_local")

    assert latest == pd.Timestamp("2026-01-01 13:00")
    assert latest.tzinfo is None


@pytest.mark.parametrize("variable", ["delivery_hour", "weekday", "month", "year"])
def test_latest_timestamp_rejects_derived_time_dimensions(variable: str) -> None:
    with pytest.raises(ValueError, match="delivery_start_utc or delivery_start_local"):
        latest_available_timestamp(pd.DataFrame({variable: [1]}), variable)


def test_latest_timestamp_fails_for_empty_missing_and_all_missing_frames() -> None:
    with pytest.raises(ValueError, match="empty DataFrame"):
        latest_available_timestamp(pd.DataFrame())
    with pytest.raises(ValueError, match="Missing required source columns"):
        latest_available_timestamp(pd.DataFrame({"other": [1]}))
    with pytest.raises(ValueError, match="only missing values"):
        latest_available_timestamp(
            pd.DataFrame({"delivery_start_utc": [None, pd.NaT]})
        )


@pytest.mark.parametrize(
    ("variable", "values", "message"),
    [
        ("delivery_start_utc", ["bad"], "Invalid datetime values"),
        ("delivery_start_utc", ["2026-01-01 00:00"], "requires timezone-aware"),
        (
            "delivery_start_local",
            ["2026-01-01T00:00:00+01:00"],
            None,
        ),
    ],
)
def test_latest_timestamp_fails_for_malformed_or_timezone_invalid_values(
    variable: str,
    values: list[str],
    message: str,
) -> None:
    if message is None:
        result = latest_available_timestamp(
            pd.DataFrame({variable: values}), variable
        )
        assert result == pd.Timestamp("2026-01-01 00:00:00")
        assert result.tzinfo is None
    else:
        with pytest.raises(ValueError, match=message):
            latest_available_timestamp(pd.DataFrame({variable: values}), variable)
