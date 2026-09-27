"""Resolve semantic research study periods into requested calendar bounds."""

import calendar
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from schemas.hypothesis_schema import (
    RelativePeriodUnit,
    StudyPeriod,
    StudyPeriodType,
)


@dataclass(frozen=True)
class ResolvedStudyPeriod:
    """Concrete requested calendar scope resolved against an explicit anchor."""

    source_type: StudyPeriodType
    as_of_date: date
    start_date: date | None
    end_date: date


def _subtract_calendar_months(value: date, months: int) -> date:
    """Subtract calendar months, clamping to the destination month's last day."""
    month_index = value.year * 12 + value.month - 1 - months
    target_year, zero_based_month = divmod(month_index, 12)
    target_month = zero_based_month + 1
    target_day = min(
        value.day,
        calendar.monthrange(target_year, target_month)[1],
    )
    return date(target_year, target_month, target_day)


def _subtract_calendar_years(value: date, years: int) -> date:
    """Subtract calendar years, clamping leap day when necessary."""
    target_year = value.year - years
    target_day = min(
        value.day,
        calendar.monthrange(target_year, value.month)[1],
    )
    return date(target_year, value.month, target_day)


def _parse_iso_date(value: str | None, field_name: str) -> date:
    """Parse one required ISO calendar date with a clear execution error."""
    if not isinstance(value, str):
        raise ValueError(
            f"ABSOLUTE_DATE_RANGE requires {field_name} in YYYY-MM-DD format."
        )
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(
            f"ABSOLUTE_DATE_RANGE requires a valid {field_name} in YYYY-MM-DD "
            "format."
        ) from error


def resolve_study_period(
    study_period: StudyPeriod,
    as_of_date: date,
) -> ResolvedStudyPeriod:
    """Resolve a typed study period against an explicit calendar date."""
    if not isinstance(study_period, StudyPeriod):
        raise ValueError("study_period must be a StudyPeriod.")
    if not isinstance(as_of_date, date) or isinstance(as_of_date, datetime):
        raise ValueError("as_of_date must be a datetime.date value.")

    if study_period.type == StudyPeriodType.RELATIVE_LOOKBACK:
        lookback_value = study_period.lookback_value
        if (
            not isinstance(lookback_value, int)
            or isinstance(lookback_value, bool)
            or lookback_value <= 0
        ):
            raise ValueError(
                "RELATIVE_LOOKBACK requires a positive integer lookback_value."
            )
        if not isinstance(study_period.lookback_unit, RelativePeriodUnit):
            raise ValueError("RELATIVE_LOOKBACK requires a lookback_unit.")

        if study_period.lookback_unit == RelativePeriodUnit.DAYS:
            start_date = as_of_date - timedelta(days=lookback_value)
        elif study_period.lookback_unit == RelativePeriodUnit.WEEKS:
            start_date = as_of_date - timedelta(weeks=lookback_value)
        elif study_period.lookback_unit == RelativePeriodUnit.MONTHS:
            start_date = _subtract_calendar_months(as_of_date, lookback_value)
        else:
            start_date = _subtract_calendar_years(as_of_date, lookback_value)

        return ResolvedStudyPeriod(
            source_type=study_period.type,
            as_of_date=as_of_date,
            start_date=start_date,
            end_date=as_of_date,
        )

    if study_period.type == StudyPeriodType.ABSOLUTE_DATE_RANGE:
        start_date = _parse_iso_date(study_period.start_date, "start_date")
        end_date = _parse_iso_date(study_period.end_date, "end_date")
        if start_date > end_date:
            raise ValueError(
                "ABSOLUTE_DATE_RANGE start_date must not be after end_date."
            )
        return ResolvedStudyPeriod(
            source_type=study_period.type,
            as_of_date=as_of_date,
            start_date=start_date,
            end_date=end_date,
        )

    if study_period.type == StudyPeriodType.ALL_AVAILABLE:
        return ResolvedStudyPeriod(
            source_type=study_period.type,
            as_of_date=as_of_date,
            start_date=None,
            end_date=as_of_date,
        )

    raise ValueError(f"Unsupported study-period type: {study_period.type}")
