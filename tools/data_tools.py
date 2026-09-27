"""Resolve and load existing raw market-data columns for a specification."""

import ast
import calendar
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum
from pathlib import Path

import pandas as pd

from compiler.variable_registry import DataType, VARIABLE_REGISTRY
from schemas.hypothesis_schema import ResearchSpecification


_SEASONS = {
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
_ISO_DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
_LOCAL_TIME_PATTERN = re.compile(r"(?:[01]\d|2[0-3]):[0-5]\d\Z")
_LOOKBACK_UNITS = {"days", "weeks", "months", "years"}
_MAX_GAP_EXAMPLES = 10


class DiagnosticSeverity(str, Enum):
    """Severity of one observed data-quality issue."""

    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class DiagnosticIssue:
    """One machine-readable data-quality finding."""

    code: str
    severity: DiagnosticSeverity
    message: str
    columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class ColumnQuality:
    """Observed quality metrics for one DataFrame column."""

    column: str
    dtype: str
    missing_count: int
    missing_ratio: float
    fully_empty: bool
    constant: bool
    positive_infinity_count: int
    negative_infinity_count: int


@dataclass(frozen=True)
class DuplicateKeyQuality:
    """Duplicate metrics for a caller-supplied record key."""

    key_columns: tuple[str, ...]
    duplicate_row_count: int
    duplicate_excess_count: int


@dataclass(frozen=True)
class DatetimeQuality:
    """Coverage, ordering, duplication, and optional cadence diagnostics."""

    variable: str
    missing_count: int
    earliest: str | None
    latest: str | None
    is_non_decreasing: bool | None
    duplicate_timestamp_count: int
    expected_resolution: str | None
    gap_count: int
    missing_interval_count: int
    irregular_interval_count: int
    gap_examples: tuple[str, ...]
    timezone_context: str


@dataclass(frozen=True)
class DataQualityReport:
    """Immutable, mode-neutral summary of a prepared DataFrame's quality."""

    row_count: int
    column_count: int
    columns: tuple[ColumnQuality, ...]
    fully_empty_columns: tuple[str, ...]
    constant_columns: tuple[str, ...]
    duplicate_full_row_count: int
    duplicate_full_row_excess_count: int
    duplicate_keys: DuplicateKeyQuality | None
    datetime_quality: DatetimeQuality | None
    requested_variables: tuple[str, ...]
    missing_requested_variables: tuple[str, ...]
    issues: tuple[DiagnosticIssue, ...]


def _registered_datetime_metadata(variable: str):
    if variable not in VARIABLE_REGISTRY:
        raise ValueError(f"Unknown registry variable: {variable}")
    metadata = VARIABLE_REGISTRY[variable]
    if metadata.data_type != DataType.RAW or metadata.unit != "datetime":
        raise ValueError(
            f"Registry variable '{variable}' is not a raw datetime variable."
        )
    if metadata.source_column is None:
        raise ValueError(
            f"Raw registry variable '{variable}' has no source column."
        )
    return metadata


def _parse_calendar_date(value: date | str, field_name: str) -> date:
    if isinstance(value, datetime) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be a datetime.date or YYYY-MM-DD string.")
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not _ISO_DATE_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must use YYYY-MM-DD format.")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field_name} must be a valid calendar date.") from error


def _parse_local_time(value: time | str, field_name: str) -> time:
    if isinstance(value, time):
        if value.tzinfo is not None and value.utcoffset() is not None:
            raise ValueError(f"{field_name} must be timezone-naive market-local time.")
        if value.second or value.microsecond:
            raise ValueError(f"{field_name} must have HH:MM precision.")
        return value.replace(tzinfo=None)
    if not isinstance(value, str) or not _LOCAL_TIME_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must use HH:MM format.")
    return time.fromisoformat(value)


def _strict_datetime_series(value: pd.Series, variable: str) -> pd.Series:
    """Normalize one registered datetime series under its explicit timezone contract.

    ``delivery_start_utc`` requires timezone-bearing absolute instants and is
    converted to UTC. ``delivery_start_local`` accepts naive values or explicit
    offsets and preserves the written market-local wall-clock components as
    timezone-naive values.
    """
    _registered_datetime_metadata(variable)
    timezone_mode = {
        "delivery_start_utc": "utc",
        "delivery_start_local": "local",
    }.get(variable)
    if timezone_mode is None:
        raise ValueError(
            f"Datetime timezone semantics are not defined for '{variable}'."
        )

    local_values: list[pd.Timestamp | object] = []
    for item in value:
        if pd.isna(item):
            if timezone_mode == "local":
                local_values.append(pd.NaT)
            continue
        try:
            parsed_item = pd.Timestamp(item)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid datetime values in registered variable '{variable}'."
            ) from error
        is_aware = (
            parsed_item.tzinfo is not None
            and parsed_item.utcoffset() is not None
        )
        if timezone_mode == "utc" and not is_aware:
            raise ValueError(
                "delivery_start_utc requires timezone-aware timestamps with "
                "an explicit Z or UTC offset."
            )
        if timezone_mode == "local":
            local_values.append(
                parsed_item.tz_localize(None) if is_aware else parsed_item
            )

    if timezone_mode == "local":
        return pd.Series(
            pd.to_datetime(local_values, errors="raise"),
            index=value.index,
            name=value.name,
        )

    try:
        return pd.to_datetime(
            value,
            errors="raise",
            utc=timezone_mode == "utc",
            format="mixed",
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid datetime values in registered variable '{variable}'."
        ) from error


def normalize_datetime_variable(
    dataframe: pd.DataFrame,
    variable: str,
) -> pd.DataFrame:
    """Return a copy with one registered datetime variable strictly normalized."""
    metadata = _registered_datetime_metadata(variable)
    source_column = metadata.source_column
    if source_column not in dataframe.columns:
        raise ValueError(f"Missing required source columns: {source_column}")
    result = dataframe.copy()
    result[variable] = _strict_datetime_series(result[source_column], variable)
    return result


def _timestamp_boundary(value: date, timezone_aware: bool) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timezone_aware else timestamp


def filter_absolute_date_range(
    dataframe: pd.DataFrame,
    datetime_variable: str,
    *,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
) -> pd.DataFrame:
    """Filter by inclusive calendar dates using half-open timestamp boundaries."""
    if start_date is None and end_date is None:
        raise ValueError("At least one absolute date-range boundary is required.")
    parsed_start = (
        _parse_calendar_date(start_date, "start_date")
        if start_date is not None
        else None
    )
    parsed_end = (
        _parse_calendar_date(end_date, "end_date")
        if end_date is not None
        else None
    )
    if parsed_start is not None and parsed_end is not None and parsed_start > parsed_end:
        raise ValueError("start_date must not be after end_date.")

    normalized = normalize_datetime_variable(dataframe, datetime_variable)
    values = normalized[datetime_variable]
    timezone_aware = datetime_variable == "delivery_start_utc"
    mask = pd.Series(True, index=normalized.index)
    if parsed_start is not None:
        mask &= values >= _timestamp_boundary(parsed_start, timezone_aware)
    if parsed_end is not None:
        exclusive_end = _timestamp_boundary(
            parsed_end + timedelta(days=1),
            timezone_aware,
        )
        mask &= values < exclusive_end
    return normalized.loc[mask].copy()


def _subtract_calendar_period(anchor: date, value: int, unit: str) -> date:
    if unit == "days":
        return anchor - timedelta(days=value)
    if unit == "weeks":
        return anchor - timedelta(weeks=value)
    if unit == "months":
        total_months = anchor.year * 12 + anchor.month - 1 - value
        year, zero_based_month = divmod(total_months, 12)
        month = zero_based_month + 1
        day = min(anchor.day, calendar.monthrange(year, month)[1])
        return date(year, month, day)
    year = anchor.year - value
    day = min(anchor.day, calendar.monthrange(year, anchor.month)[1])
    return date(year, anchor.month, day)


def filter_relative_lookback(
    dataframe: pd.DataFrame,
    datetime_variable: str,
    *,
    as_of_date: date,
    lookback_value: int,
    lookback_unit: str,
) -> pd.DataFrame:
    """Filter an inclusive calendar lookback anchored to an explicit date."""
    if isinstance(as_of_date, datetime) or not isinstance(as_of_date, date):
        raise ValueError("as_of_date must be a datetime.date value.")
    if (
        isinstance(lookback_value, bool)
        or not isinstance(lookback_value, int)
        or lookback_value <= 0
    ):
        raise ValueError("lookback_value must be a positive integer.")
    if lookback_unit not in _LOOKBACK_UNITS:
        raise ValueError(
            "lookback_unit must be one of: days, weeks, months, years."
        )
    start_date = _subtract_calendar_period(
        as_of_date,
        lookback_value,
        lookback_unit,
    )
    return filter_absolute_date_range(
        dataframe,
        datetime_variable,
        start_date=start_date,
        end_date=as_of_date,
    )


def filter_delivery_time_window(
    dataframe: pd.DataFrame,
    datetime_variable: str,
    *,
    start_local: time | str,
    end_local: time | str,
    delivery_date: date | str | None = None,
) -> pd.DataFrame:
    """Filter a market-local delivery window using ``[start, end)`` semantics."""
    if datetime_variable != "delivery_start_local":
        raise ValueError(
            "Delivery-time filtering requires delivery_start_local semantics."
        )
    parsed_start = _parse_local_time(start_local, "start_local")
    parsed_end = _parse_local_time(end_local, "end_local")
    if parsed_start == parsed_end:
        raise ValueError("Delivery window start_local and end_local must differ.")

    normalized = normalize_datetime_variable(dataframe, datetime_variable)
    values = normalized[datetime_variable]
    if delivery_date is not None:
        parsed_date = _parse_calendar_date(delivery_date, "delivery_date")
        start_boundary = pd.Timestamp.combine(parsed_date, parsed_start)
        end_date = (
            parsed_date + timedelta(days=1)
            if parsed_start > parsed_end
            else parsed_date
        )
        end_boundary = pd.Timestamp.combine(end_date, parsed_end)
        mask = (values >= start_boundary) & (values < end_boundary)
    else:
        seconds = (
            values.dt.hour * 3600
            + values.dt.minute * 60
            + values.dt.second
        )
        start_seconds = parsed_start.hour * 3600 + parsed_start.minute * 60
        end_seconds = parsed_end.hour * 3600 + parsed_end.minute * 60
        if parsed_start < parsed_end:
            mask = (seconds >= start_seconds) & (seconds < end_seconds)
        else:
            mask = (seconds >= start_seconds) | (seconds < end_seconds)
    return normalized.loc[mask].copy()


def _column_quality(dataframe: pd.DataFrame) -> tuple[ColumnQuality, ...]:
    row_count = len(dataframe)
    reports: list[ColumnQuality] = []
    for column in dataframe.columns:
        series = dataframe[column]
        missing_count = int(series.isna().sum())
        non_missing_count = row_count - missing_count
        fully_empty = non_missing_count == 0
        constant = not fully_empty and series.nunique(dropna=True) == 1
        if pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(
            series.dtype
        ):
            positive_infinity_count = int(series.eq(float("inf")).fillna(False).sum())
            negative_infinity_count = int(series.eq(float("-inf")).fillna(False).sum())
        else:
            positive_infinity_count = 0
            negative_infinity_count = 0
        reports.append(
            ColumnQuality(
                column=str(column),
                dtype=str(series.dtype),
                missing_count=missing_count,
                missing_ratio=(missing_count / row_count if row_count else 0.0),
                fully_empty=fully_empty,
                constant=constant,
                positive_infinity_count=positive_infinity_count,
                negative_infinity_count=negative_infinity_count,
            )
        )
    return tuple(reports)


def _expected_resolution(
    resolution_variables: tuple[str, ...],
) -> tuple[str | None, pd.Timedelta | None, DiagnosticIssue | None]:
    resolutions = tuple(
        dict.fromkeys(
            VARIABLE_REGISTRY[variable].resolution
            for variable in resolution_variables
            if VARIABLE_REGISTRY[variable].resolution != "timestamp"
        )
    )
    if not resolutions:
        return (
            None,
            None,
            DiagnosticIssue(
                code="resolution_unavailable",
                severity=DiagnosticSeverity.ERROR,
                message="No time-series resolution is available for the supplied variables.",
                columns=resolution_variables,
            ),
        )
    if len(resolutions) != 1:
        return (
            None,
            None,
            DiagnosticIssue(
                code="conflicting_resolutions",
                severity=DiagnosticSeverity.ERROR,
                message=(
                    "Resolution variables have conflicting registry resolutions: "
                    + ", ".join(resolutions)
                ),
                columns=resolution_variables,
            ),
        )
    resolution = resolutions[0]
    try:
        duration = pd.to_timedelta(resolution)
    except (TypeError, ValueError) as error:
        return (
            resolution,
            None,
            DiagnosticIssue(
                code="invalid_resolution",
                severity=DiagnosticSeverity.ERROR,
                message=f"Registry resolution cannot be interpreted: {resolution}",
                columns=resolution_variables,
            ),
        )
    if duration <= pd.Timedelta(0):
        return (
            resolution,
            None,
            DiagnosticIssue(
                code="invalid_resolution",
                severity=DiagnosticSeverity.ERROR,
                message=f"Registry resolution must be positive: {resolution}",
                columns=resolution_variables,
            ),
        )
    return resolution, duration, None


def _datetime_quality(
    dataframe: pd.DataFrame,
    datetime_variable: str,
    resolution_variables: tuple[str, ...] | None,
) -> tuple[DatetimeQuality, tuple[DiagnosticIssue, ...]]:
    metadata = _registered_datetime_metadata(datetime_variable)
    source_column = metadata.source_column
    timezone_context = (
        "utc_absolute"
        if datetime_variable == "delivery_start_utc"
        else "market_local_naive"
    )
    if source_column not in dataframe.columns:
        issue = DiagnosticIssue(
            code="missing_datetime_column",
            severity=DiagnosticSeverity.ERROR,
            message=f"Datetime source column is missing: {source_column}",
            columns=(source_column,),
        )
        return (
            DatetimeQuality(
                variable=datetime_variable,
                missing_count=0,
                earliest=None,
                latest=None,
                is_non_decreasing=None,
                duplicate_timestamp_count=0,
                expected_resolution=None,
                gap_count=0,
                missing_interval_count=0,
                irregular_interval_count=0,
                gap_examples=(),
                timezone_context=timezone_context,
            ),
            (issue,),
        )

    source = dataframe[source_column]
    missing_count = int(source.isna().sum())
    issues: list[DiagnosticIssue] = []
    if missing_count:
        issues.append(
            DiagnosticIssue(
                code="missing_timestamps",
                severity=DiagnosticSeverity.WARNING,
                message=f"Datetime column contains {missing_count} missing values.",
                columns=(source_column,),
            )
        )
    try:
        parsed = _strict_datetime_series(source, datetime_variable)
    except ValueError as error:
        issues.append(
            DiagnosticIssue(
                code="malformed_timestamps",
                severity=DiagnosticSeverity.ERROR,
                message=str(error),
                columns=(source_column,),
            )
        )
        return (
            DatetimeQuality(
                variable=datetime_variable,
                missing_count=missing_count,
                earliest=None,
                latest=None,
                is_non_decreasing=None,
                duplicate_timestamp_count=0,
                expected_resolution=None,
                gap_count=0,
                missing_interval_count=0,
                irregular_interval_count=0,
                gap_examples=(),
                timezone_context=timezone_context,
            ),
            tuple(issues),
        )

    valid = parsed.dropna()
    if valid.empty:
        issues.append(
            DiagnosticIssue(
                code="datetime_coverage_unavailable",
                severity=DiagnosticSeverity.ERROR,
                message="Datetime coverage cannot be established because all values are missing.",
                columns=(source_column,),
            )
        )
        earliest = latest = None
        is_non_decreasing = None
        duplicate_timestamp_count = 0
    else:
        earliest = valid.min().isoformat()
        latest = valid.max().isoformat()
        is_non_decreasing = bool(valid.is_monotonic_increasing)
        duplicate_timestamp_count = int(valid.duplicated(keep=False).sum())
        if not is_non_decreasing:
            issues.append(
                DiagnosticIssue(
                    code="timestamps_out_of_order",
                    severity=DiagnosticSeverity.WARNING,
                    message="Non-missing timestamps are not in non-decreasing order.",
                    columns=(source_column,),
                )
            )
        if duplicate_timestamp_count:
            issues.append(
                DiagnosticIssue(
                    code="duplicate_timestamps",
                    severity=DiagnosticSeverity.WARNING,
                    message=(
                        f"{duplicate_timestamp_count} rows participate in duplicate timestamps."
                    ),
                    columns=(source_column,),
                )
            )

    expected_resolution: str | None = None
    expected_duration: pd.Timedelta | None = None
    if resolution_variables is not None:
        expected_resolution, expected_duration, resolution_issue = _expected_resolution(
            resolution_variables
        )
        if resolution_issue is not None:
            issues.append(resolution_issue)

    gap_count = 0
    missing_interval_count = 0
    irregular_interval_count = 0
    gap_examples: list[str] = []
    if expected_duration is not None and len(valid) >= 2:
        unique_sorted = valid.drop_duplicates().sort_values()
        previous = unique_sorted.iloc[0]
        for current in unique_sorted.iloc[1:]:
            delta = current - previous
            if delta > expected_duration:
                gap_count += 1
                exact_multiple = delta % expected_duration == pd.Timedelta(0)
                if exact_multiple:
                    missing_interval_count += int(delta // expected_duration) - 1
                else:
                    irregular_interval_count += 1
                if len(gap_examples) < _MAX_GAP_EXAMPLES:
                    gap_examples.append(
                        f"{previous.isoformat()} -> {current.isoformat()} ({delta})"
                    )
            elif delta < expected_duration:
                irregular_interval_count += 1
            previous = current
        if gap_count:
            issues.append(
                DiagnosticIssue(
                    code="timestamp_gaps",
                    severity=DiagnosticSeverity.WARNING,
                    message=f"Detected {gap_count} timestamp gap spans.",
                    columns=(source_column,),
                )
            )
        if irregular_interval_count:
            issues.append(
                DiagnosticIssue(
                    code="irregular_timestamp_intervals",
                    severity=DiagnosticSeverity.WARNING,
                    message=(
                        f"Detected {irregular_interval_count} intervals that do not "
                        "align with the expected cadence."
                    ),
                    columns=(source_column,),
                )
            )

    if datetime_variable == "delivery_start_local" and (
        duplicate_timestamp_count or gap_count or irregular_interval_count
    ):
        issues.append(
            DiagnosticIssue(
                code="local_time_dst_classification_unavailable",
                severity=DiagnosticSeverity.WARNING,
                message=(
                    "Market-local timestamps are timezone-naive; possible DST gaps "
                    "or duplicates cannot be classified without a named timezone "
                    "or paired UTC analysis."
                ),
                columns=(source_column,),
            )
        )

    return (
        DatetimeQuality(
            variable=datetime_variable,
            missing_count=missing_count,
            earliest=earliest,
            latest=latest,
            is_non_decreasing=is_non_decreasing,
            duplicate_timestamp_count=duplicate_timestamp_count,
            expected_resolution=expected_resolution,
            gap_count=gap_count,
            missing_interval_count=missing_interval_count,
            irregular_interval_count=irregular_interval_count,
            gap_examples=tuple(gap_examples),
            timezone_context=timezone_context,
        ),
        tuple(issues),
    )


def diagnose_data_quality(
    dataframe: pd.DataFrame,
    *,
    requested_variables: tuple[str, ...] | list[str] = (),
    key_columns: tuple[str, ...] | list[str] = (),
    datetime_variable: str | None = None,
    resolution_variables: tuple[str, ...] | list[str] | None = None,
) -> DataQualityReport:
    """Observe prepared data quality without cleaning or mutating the DataFrame."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Data-quality diagnostics require a pandas DataFrame.")
    requested = tuple(dict.fromkeys(requested_variables))
    unknown_requested = tuple(
        variable for variable in requested if variable not in VARIABLE_REGISTRY
    )
    if unknown_requested:
        raise ValueError(
            f"Unknown registry variable: {', '.join(unknown_requested)}"
        )
    keys = tuple(dict.fromkeys(key_columns))
    if any(not isinstance(column, str) or not column.strip() for column in keys):
        raise ValueError("key_columns must contain non-empty column names.")

    explicit_resolution_variables: tuple[str, ...] | None = None
    if resolution_variables is not None:
        explicit_resolution_variables = tuple(dict.fromkeys(resolution_variables))
        if not explicit_resolution_variables:
            raise ValueError(
                "resolution_variables must be non-empty when explicitly supplied."
            )
        unknown_resolution = tuple(
            variable
            for variable in explicit_resolution_variables
            if variable not in VARIABLE_REGISTRY
        )
        if unknown_resolution:
            raise ValueError(
                f"Unknown registry variable: {', '.join(unknown_resolution)}"
            )
        if datetime_variable is None:
            raise ValueError(
                "datetime_variable is required when resolution_variables are supplied."
            )

    column_reports = _column_quality(dataframe)
    issues: list[DiagnosticIssue] = []
    for report in column_reports:
        if report.missing_count:
            issues.append(
                DiagnosticIssue(
                    code="missing_values",
                    severity=DiagnosticSeverity.WARNING,
                    message=(
                        f"Column '{report.column}' contains {report.missing_count} "
                        "missing values."
                    ),
                    columns=(report.column,),
                )
            )
        if report.fully_empty:
            issues.append(
                DiagnosticIssue(
                    code="fully_empty_column",
                    severity=DiagnosticSeverity.WARNING,
                    message=f"Column '{report.column}' is fully empty.",
                    columns=(report.column,),
                )
            )
        elif report.constant:
            issues.append(
                DiagnosticIssue(
                    code="constant_column",
                    severity=DiagnosticSeverity.WARNING,
                    message=f"Column '{report.column}' is constant.",
                    columns=(report.column,),
                )
            )
        if report.positive_infinity_count or report.negative_infinity_count:
            issues.append(
                DiagnosticIssue(
                    code="non_finite_numeric_values",
                    severity=DiagnosticSeverity.WARNING,
                    message=(
                        f"Column '{report.column}' contains "
                        f"{report.positive_infinity_count} positive and "
                        f"{report.negative_infinity_count} negative infinity values."
                    ),
                    columns=(report.column,),
                )
            )

    duplicate_full_row_count = int(dataframe.duplicated(keep=False).sum())
    duplicate_full_row_excess_count = int(dataframe.duplicated(keep="first").sum())
    if duplicate_full_row_count:
        issues.append(
            DiagnosticIssue(
                code="duplicate_full_rows",
                severity=DiagnosticSeverity.WARNING,
                message=(
                    f"{duplicate_full_row_count} rows participate in full-row duplicates."
                ),
            )
        )

    duplicate_keys: DuplicateKeyQuality | None = None
    if keys:
        missing_keys = tuple(column for column in keys if column not in dataframe.columns)
        if missing_keys:
            issues.append(
                DiagnosticIssue(
                    code="missing_key_columns",
                    severity=DiagnosticSeverity.ERROR,
                    message=f"Supplied key columns are missing: {', '.join(missing_keys)}",
                    columns=missing_keys,
                )
            )
            duplicate_keys = DuplicateKeyQuality(keys, 0, 0)
        else:
            duplicate_key_rows = int(
                dataframe.duplicated(subset=list(keys), keep=False).sum()
            )
            duplicate_key_excess = int(
                dataframe.duplicated(subset=list(keys), keep="first").sum()
            )
            duplicate_keys = DuplicateKeyQuality(
                keys,
                duplicate_key_rows,
                duplicate_key_excess,
            )
            if duplicate_key_rows:
                issues.append(
                    DiagnosticIssue(
                        code="duplicate_keys",
                        severity=DiagnosticSeverity.WARNING,
                        message=(
                            f"{duplicate_key_rows} rows participate in duplicate "
                            "supplied keys."
                        ),
                        columns=keys,
                    )
                )

    missing_requested = tuple(
        variable for variable in requested if variable not in dataframe.columns
    )
    if missing_requested:
        issues.append(
            DiagnosticIssue(
                code="missing_requested_variables",
                severity=DiagnosticSeverity.ERROR,
                message=(
                    "Requested canonical variables are not materialized: "
                    + ", ".join(missing_requested)
                ),
                columns=missing_requested,
            )
        )

    datetime_quality: DatetimeQuality | None = None
    if datetime_variable is not None:
        datetime_quality, datetime_issues = _datetime_quality(
            dataframe,
            datetime_variable,
            explicit_resolution_variables,
        )
        issues.extend(datetime_issues)

    return DataQualityReport(
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        columns=column_reports,
        fully_empty_columns=tuple(
            report.column for report in column_reports if report.fully_empty
        ),
        constant_columns=tuple(
            report.column for report in column_reports if report.constant
        ),
        duplicate_full_row_count=duplicate_full_row_count,
        duplicate_full_row_excess_count=duplicate_full_row_excess_count,
        duplicate_keys=duplicate_keys,
        datetime_quality=datetime_quality,
        requested_variables=requested,
        missing_requested_variables=missing_requested,
        issues=tuple(issues),
    )


def _formula_error(variable: str, formula: str, detail: str) -> ValueError:
    return ValueError(
        f"Unsupported formula for variable '{variable}': {formula!r}. {detail}"
    )


def _datetime_component(
    value: pd.Series,
    function_name: str,
    variable: str,
    datetime_variable: str,
) -> pd.Series:
    """Apply one approved datetime transform; weekdays use Monday=0, Sunday=6."""
    try:
        parsed = _strict_datetime_series(value, datetime_variable)
    except ValueError as error:
        raise ValueError(
            f"Invalid datetime values while materializing '{variable}' "
            f"with {function_name}(). {error}"
        ) from error

    if function_name == "hour":
        return parsed.dt.hour
    if function_name == "minute":
        return parsed.dt.minute
    if function_name == "weekday":
        return parsed.dt.weekday
    if function_name == "month":
        return parsed.dt.month
    if function_name == "year":
        return parsed.dt.year
    if function_name == "season":
        return parsed.dt.month.map(_SEASONS)
    raise AssertionError(f"Unexpected datetime function: {function_name}")


def _evaluate_formula(
    dataframe: pd.DataFrame,
    variable: str,
) -> pd.Series:
    """Evaluate one registry formula using a deliberately restricted AST."""
    metadata = VARIABLE_REGISTRY[variable]
    formula = metadata.formula
    if not formula:
        raise _formula_error(variable, formula or "", "A formula is required.")
    try:
        expression = ast.parse(formula, mode="eval")
    except SyntaxError as error:
        raise _formula_error(variable, formula, "The expression is invalid.") from error

    dependencies = set(metadata.dependencies)

    def evaluate(node: ast.AST) -> pd.Series:
        if isinstance(node, ast.Name):
            if node.id not in dependencies:
                raise _formula_error(
                    variable,
                    formula,
                    f"Name '{node.id}' is not a registered dependency.",
                )
            if node.id not in dataframe.columns:
                raise ValueError(
                    f"Materialized dependency '{node.id}' is missing for '{variable}'."
                )
            return dataframe[node.id]

        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            left = evaluate(node.left)
            right = evaluate(node.right)
            return left + right if isinstance(node.op, ast.Add) else left - right

        if isinstance(node, ast.Call):
            if (
                not isinstance(node.func, ast.Name)
                or node.func.id
                not in {"hour", "minute", "weekday", "month", "year", "season"}
                or len(node.args) != 1
                or node.keywords
                or not isinstance(node.args[0], ast.Name)
            ):
                raise _formula_error(
                    variable,
                    formula,
                    "Only one-argument approved datetime functions are allowed.",
                )
            argument = evaluate(node.args[0])
            return _datetime_component(
                argument,
                node.func.id,
                variable,
                node.args[0].id,
            )

        raise _formula_error(
            variable,
            formula,
            f"Syntax '{type(node).__name__}' is not allowed.",
        )

    return evaluate(expression.body)


def research_variables(
    specification: ResearchSpecification,
) -> tuple[str, ...]:
    """Return all variables used by a complete historical research task."""
    variables: list[str] = []

    def add(variable: str | None) -> None:
        if variable is not None and variable not in variables:
            variables.append(variable)

    add(specification.target)
    add(specification.reference)
    if specification.event is not None:
        add(specification.event.variable)
    for condition in specification.conditions:
        add(condition.variable)
    for condition in specification.filters:
        add(condition.variable)
    for condition in specification.baseline:
        add(condition.variable)
    return tuple(variables)


def specification_variables(
    specification: ResearchSpecification,
) -> tuple[str, ...]:
    """Return all variables required by a Research specification."""
    return research_variables(specification)


def raw_dependencies(variable: str) -> tuple[str, ...]:
    """Recursively resolve one canonical variable to its raw dependencies."""
    raw: list[str] = []

    def visit(current: str, visiting: tuple[str, ...]) -> None:
        if current not in VARIABLE_REGISTRY:
            if visiting:
                raise ValueError(
                    f"Unknown dependency '{current}' required by '{visiting[-1]}'."
                )
            raise ValueError(f"Unknown registry variable: {current}")
        if current in visiting:
            cycle = " -> ".join((*visiting, current))
            raise ValueError(f"Variable dependency cycle detected: {cycle}")

        metadata = VARIABLE_REGISTRY[current]
        if metadata.data_type == DataType.RAW:
            if current not in raw:
                raw.append(current)
            return
        for dependency in metadata.dependencies:
            visit(dependency, (*visiting, current))

    visit(variable, ())
    return tuple(raw)


def required_source_columns_for_variables(
    variables: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    """Resolve canonical variables to their existing raw source columns."""
    columns: list[str] = []
    for variable in variables:
        for raw_variable in raw_dependencies(variable):
            source_column = VARIABLE_REGISTRY[raw_variable].source_column
            if source_column is None:
                raise ValueError(
                    f"Raw registry variable '{raw_variable}' has no source column."
                )
            if source_column not in columns:
                columns.append(source_column)
    return tuple(columns)


def materialize_variables(
    dataframe: pd.DataFrame,
    variables: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    """Return a copy with requested variables and derived dependencies materialized."""
    requested = tuple(dict.fromkeys(variables))
    required_columns = required_source_columns_for_variables(requested)
    missing = tuple(
        column for column in required_columns if column not in dataframe.columns
    )
    if missing:
        raise ValueError(f"Missing required source columns: {', '.join(missing)}")

    result = dataframe.copy()
    materialized: set[str] = set()
    visiting: tuple[str, ...] = ()

    def materialize(variable: str) -> None:
        nonlocal visiting
        if variable in materialized:
            return
        if variable not in VARIABLE_REGISTRY:
            if visiting:
                raise ValueError(
                    f"Unknown dependency '{variable}' required by '{visiting[-1]}'."
                )
            raise ValueError(f"Unknown registry variable: {variable}")
        if variable in visiting:
            cycle = " -> ".join((*visiting, variable))
            raise ValueError(f"Variable dependency cycle detected: {cycle}")

        metadata = VARIABLE_REGISTRY[variable]
        if metadata.data_type == DataType.RAW:
            source_column = metadata.source_column
            if source_column is None:
                raise ValueError(
                    f"Raw registry variable '{variable}' has no source column."
                )
            if variable != source_column:
                result[variable] = result[source_column]
            materialized.add(variable)
            return

        previous_visiting = visiting
        visiting = (*visiting, variable)
        try:
            for dependency in metadata.dependencies:
                materialize(dependency)
            result[variable] = _evaluate_formula(result, variable)
            materialized.add(variable)
        finally:
            visiting = previous_visiting

    for variable in requested:
        materialize(variable)
    return result


def required_source_columns(
    specification: ResearchSpecification,
) -> tuple[str, ...]:
    """Return raw source columns needed by a research specification."""
    variables = list(specification_variables(specification))
    if (
        (
            specification.study_period is not None
            or specification.delivery_window is not None
        )
        and "delivery_start_local" not in variables
    ):
        variables.append("delivery_start_local")
    return required_source_columns_for_variables(variables)


def load_market_data(
    csv_path: str | Path,
    required_columns: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    """Load a CSV, optionally selecting and verifying required source columns."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Market data file does not exist: {path}")

    if required_columns is None:
        return pd.read_csv(path)

    requested = tuple(dict.fromkeys(required_columns))
    available = tuple(pd.read_csv(path, nrows=0).columns)
    missing = tuple(column for column in requested if column not in available)
    if missing:
        raise ValueError(f"Missing required source columns: {', '.join(missing)}")
    return pd.read_csv(path, usecols=list(requested))


def extract_variable_data(
    dataframe: pd.DataFrame,
    variables: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    """Copy existing raw columns required by an explicit variable collection."""
    columns = required_source_columns_for_variables(variables)
    missing = tuple(column for column in columns if column not in dataframe.columns)
    if missing:
        raise ValueError(f"Missing required source columns: {', '.join(missing)}")
    return dataframe.loc[:, list(columns)].copy()


def extract_required_data(
    dataframe: pd.DataFrame,
    specification: ResearchSpecification,
) -> pd.DataFrame:
    """Return a copy containing only required existing raw source columns."""
    columns = required_source_columns(specification)
    missing = tuple(column for column in columns if column not in dataframe.columns)
    if missing:
        raise ValueError(f"Missing required source columns: {', '.join(missing)}")
    return dataframe.loc[:, list(columns)].copy()


def prepare_variable_data(
    csv_path: str | Path,
    variables: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    """Load existing raw columns required by an explicit variable collection."""
    columns = required_source_columns_for_variables(variables)
    return load_market_data(csv_path, columns)


def prepare_market_data(
    csv_path: str | Path,
    specification: ResearchSpecification,
) -> pd.DataFrame:
    """Load only existing raw columns required by a specification."""
    columns = required_source_columns(specification)
    return load_market_data(csv_path, columns)
