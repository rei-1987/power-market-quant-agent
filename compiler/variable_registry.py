"""Closed-world variable registry for available power-market data."""

from dataclasses import dataclass
from enum import Enum


class DomainTermType(str, Enum):
    """Kinds of named market-domain terms."""

    INTRADAY_PRICE_INDEX = "intraday_price_index"
    FORECAST_VINTAGE = "forecast_vintage"


class DataType(str, Enum):
    """Whether a variable is stored directly or deterministically derived."""

    RAW = "raw"
    DERIVED = "derived"


class AvailabilityType(str, Enum):
    """Supported point-in-time availability categories."""

    DAY_AHEAD = "day_ahead"
    DELIVERY_DAY_FIXED_TIME = "delivery_day_fixed_time"
    FINAL_AFTER_TRADING_WINDOW = "final_after_trading_window"
    EX_POST = "ex_post"
    DERIVED_FROM_DEPENDENCIES = "derived_from_dependencies"
    RECORD_METADATA = "record_metadata"


class VariableRole(str, Enum):
    """Ways a variable may be used by later research components."""

    SIGNAL = "signal"
    TARGET = "target"
    HISTORICAL_ANALYSIS = "historical_analysis"
    EX_POST_VALIDATION = "ex_post_validation"
    FILTER = "filter"


@dataclass(frozen=True)
class Availability:
    """Point-in-time availability metadata."""

    type: AvailabilityType
    fixed_time: str | None = None
    timezone_context: str | None = None


@dataclass(frozen=True)
class DomainTerm:
    """Definition of an official market term or process."""

    name: str
    term_type: DomainTermType
    meaning: str
    availability: Availability
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class VariableMetadata:
    """Metadata for one canonical raw or derived variable."""

    category: str
    data_type: DataType
    unit: str
    description: str
    aliases: tuple[str, ...]
    resolution: str
    source_column: str | None
    availability: Availability
    allowed_roles: tuple[VariableRole, ...]
    dependencies: tuple[str, ...] = ()
    formula: str | None = None


DAY_AHEAD = Availability(AvailabilityType.DAY_AHEAD)
DELIVERY_DAY_0800 = Availability(
    AvailabilityType.DELIVERY_DAY_FIXED_TIME,
    fixed_time="08:00",
    timezone_context="market local time",
)
FINAL_AFTER_TRADING = Availability(AvailabilityType.FINAL_AFTER_TRADING_WINDOW)
EX_POST = Availability(AvailabilityType.EX_POST)
DERIVED = Availability(AvailabilityType.DERIVED_FROM_DEPENDENCIES)
RECORD_METADATA = Availability(AvailabilityType.RECORD_METADATA)

DOMAIN_TERMS: dict[str, DomainTerm] = {
    "id1": DomainTerm(
        name="ID1",
        term_type=DomainTermType.INTRADAY_PRICE_INDEX,
        meaning="Volume-weighted average price over the last 1 hour before delivery.",
        availability=FINAL_AFTER_TRADING,
        aliases=("id1",),
    ),
    "id3": DomainTerm(
        name="ID3",
        term_type=DomainTermType.INTRADAY_PRICE_INDEX,
        meaning="Volume-weighted average price over the last 3 hours before delivery.",
        availability=FINAL_AFTER_TRADING,
        aliases=("id3",),
    ),
    "a01": DomainTerm(
        name="A01",
        term_type=DomainTermType.FORECAST_VINTAGE,
        meaning="Day-ahead generation forecast.",
        availability=DAY_AHEAD,
        aliases=("a01",),
    ),
    "a40": DomainTerm(
        name="A40",
        term_type=DomainTermType.FORECAST_VINTAGE,
        meaning=(
            "Delivery-day generation forecast available at 08:00 market-local time."
        ),
        availability=DELIVERY_DAY_0800,
        aliases=("a40",),
    ),
}

SIGNAL_ROLES = (VariableRole.SIGNAL, VariableRole.HISTORICAL_ANALYSIS)
TARGET_ROLES = (VariableRole.TARGET, VariableRole.HISTORICAL_ANALYSIS)
EX_POST_ROLES = (
    VariableRole.TARGET,
    VariableRole.HISTORICAL_ANALYSIS,
    VariableRole.EX_POST_VALIDATION,
)
FILTER_ROLES = (VariableRole.FILTER, VariableRole.HISTORICAL_ANALYSIS)


def _raw(
    category: str,
    unit: str,
    description: str,
    aliases: tuple[str, ...],
    resolution: str,
    source_column: str,
    availability: Availability,
    allowed_roles: tuple[VariableRole, ...],
) -> VariableMetadata:
    return VariableMetadata(
        category=category,
        data_type=DataType.RAW,
        unit=unit,
        description=description,
        aliases=aliases,
        resolution=resolution,
        source_column=source_column,
        availability=availability,
        allowed_roles=allowed_roles,
    )


def _derived(
    category: str,
    unit: str,
    description: str,
    aliases: tuple[str, ...],
    resolution: str,
    allowed_roles: tuple[VariableRole, ...],
    dependencies: tuple[str, ...],
    formula: str,
) -> VariableMetadata:
    return VariableMetadata(
        category=category,
        data_type=DataType.DERIVED,
        unit=unit,
        description=description,
        aliases=aliases,
        resolution=resolution,
        source_column=None,
        availability=DERIVED,
        allowed_roles=allowed_roles,
        dependencies=dependencies,
        formula=formula,
    )


VARIABLE_REGISTRY: dict[str, VariableMetadata] = {
    "id1_price_eur_mwh": _raw(
        "price", "EUR/MWh", "Final ID1 intraday price index.",
        ("id1", "id1 price", "id1 index", "intraday id1"),
        "15min", "id1_price_eur_mwh", FINAL_AFTER_TRADING, TARGET_ROLES,
    ),
    "id3_price_eur_mwh": _raw(
        "price", "EUR/MWh", "Final ID3 intraday price index.",
        ("id3", "id3 price", "id3 index", "intraday id3"),
        "15min", "id3_price_eur_mwh", FINAL_AFTER_TRADING, TARGET_ROLES,
    ),
    "solar_a01": _raw(
        "forecast", "MW", "Day-ahead solar generation forecast.",
        ("solar a01", "day-ahead solar forecast", "solar day-ahead forecast"),
        "15min", "solar_a01", DAY_AHEAD, SIGNAL_ROLES,
    ),
    "solar_a40": _raw(
        "forecast", "MW", "Solar forecast available at 08:00 on delivery day.",
        ("solar a40", "08:00 solar forecast", "delivery-day 08:00 solar forecast"),
        "15min", "solar_a40", DELIVERY_DAY_0800, SIGNAL_ROLES,
    ),
    "wind_onshore_a01": _raw(
        "forecast", "MW", "Day-ahead onshore wind generation forecast.",
        ("onshore wind a01", "day-ahead onshore wind forecast"),
        "15min", "wind_onshore_a01", DAY_AHEAD, SIGNAL_ROLES,
    ),
    "wind_onshore_a40": _raw(
        "forecast", "MW", "Onshore wind forecast available at 08:00 on delivery day.",
        (
            "onshore wind a40",
            "08:00 onshore wind forecast",
            "delivery-day 08:00 onshore wind forecast",
        ),
        "15min", "wind_onshore_a40", DELIVERY_DAY_0800, SIGNAL_ROLES,
    ),
    "wind_offshore_a01": _raw(
        "forecast", "MW", "Day-ahead offshore wind generation forecast.",
        ("offshore wind a01", "day-ahead offshore wind forecast"),
        "15min", "wind_offshore_a01", DAY_AHEAD, SIGNAL_ROLES,
    ),
    "wind_offshore_a40": _raw(
        "forecast", "MW", "Offshore wind forecast available at 08:00 on delivery day.",
        (
            "offshore wind a40",
            "08:00 offshore wind forecast",
            "delivery-day 08:00 offshore wind forecast",
        ),
        "15min", "wind_offshore_a40", DELIVERY_DAY_0800, SIGNAL_ROLES,
    ),
    "solar_actual_mw": _raw(
        "actual_generation", "MW", "Observed solar generation.",
        ("actual solar", "solar actual", "solar generation"),
        "15min", "solar_actual_mw", EX_POST, EX_POST_ROLES,
    ),
    "wind_onshore_actual_mw": _raw(
        "actual_generation", "MW", "Observed onshore wind generation.",
        ("actual onshore wind", "onshore wind actual", "onshore wind generation"),
        "15min", "wind_onshore_actual_mw", EX_POST, EX_POST_ROLES,
    ),
    "wind_offshore_actual_mw": _raw(
        "actual_generation", "MW", "Observed offshore wind generation.",
        ("actual offshore wind", "offshore wind actual", "offshore wind generation"),
        "15min", "wind_offshore_actual_mw", EX_POST, EX_POST_ROLES,
    ),
    "delivery_start_utc": _raw(
        "record_metadata", "datetime", "Delivery interval start in UTC.",
        ("delivery start utc", "utc delivery time"),
        "timestamp", "delivery_start_utc", RECORD_METADATA, FILTER_ROLES,
    ),
    "delivery_start_local": _raw(
        "record_metadata", "datetime", "Delivery interval start in market-local time.",
        ("delivery start local", "local delivery time"),
        "timestamp", "delivery_start_local", RECORD_METADATA, FILTER_ROLES,
    ),
    "id1_minus_id3": _derived(
        "price_difference", "EUR/MWh", "ID1 price minus ID3 price.",
        ("id1 minus id3", "id1 id3 difference", "intraday price difference"),
        "15min", TARGET_ROLES, ("id1_price_eur_mwh", "id3_price_eur_mwh"),
        "id1_price_eur_mwh - id3_price_eur_mwh",
    ),
    "solar_revision": _derived(
        "forecast_revision", "MW", "Solar forecast revision from A01 to A40.",
        ("solar revision", "solar forecast revision"), "15min", SIGNAL_ROLES,
        ("solar_a40", "solar_a01"), "solar_a40 - solar_a01",
    ),
    "wind_onshore_revision": _derived(
        "forecast_revision", "MW", "Onshore wind forecast revision from A01 to A40.",
        ("onshore wind revision", "onshore wind forecast revision"),
        "15min", SIGNAL_ROLES, ("wind_onshore_a40", "wind_onshore_a01"),
        "wind_onshore_a40 - wind_onshore_a01",
    ),
    "wind_offshore_revision": _derived(
        "forecast_revision", "MW", "Offshore wind forecast revision from A01 to A40.",
        ("offshore wind revision", "offshore wind forecast revision"),
        "15min", SIGNAL_ROLES, ("wind_offshore_a40", "wind_offshore_a01"),
        "wind_offshore_a40 - wind_offshore_a01",
    ),
    "total_wind_a01": _derived(
        "wind_aggregate", "MW", "Total onshore and offshore wind A01 forecast.",
        ("total wind a01", "day-ahead total wind forecast"), "15min", SIGNAL_ROLES,
        ("wind_onshore_a01", "wind_offshore_a01"),
        "wind_onshore_a01 + wind_offshore_a01",
    ),
    "total_wind_a40": _derived(
        "wind_aggregate", "MW", "Total onshore and offshore wind A40 forecast.",
        ("total wind a40", "08:00 total wind forecast"), "15min", SIGNAL_ROLES,
        ("wind_onshore_a40", "wind_offshore_a40"),
        "wind_onshore_a40 + wind_offshore_a40",
    ),
    "total_wind_actual": _derived(
        "wind_aggregate", "MW", "Observed total onshore and offshore wind generation.",
        ("total wind actual", "actual total wind", "total wind generation"),
        "15min", EX_POST_ROLES,
        ("wind_onshore_actual_mw", "wind_offshore_actual_mw"),
        "wind_onshore_actual_mw + wind_offshore_actual_mw",
    ),
    "total_wind_revision": _derived(
        "forecast_revision", "MW", "Total wind forecast revision from A01 to A40.",
        ("total wind revision", "total wind forecast revision"),
        "15min", SIGNAL_ROLES, ("total_wind_a40", "total_wind_a01"),
        "total_wind_a40 - total_wind_a01",
    ),
    "wind_solar_a01": _derived(
        "wind_solar_aggregate", "MW", "Combined wind and solar A01 forecast.",
        ("wind solar a01", "day-ahead wind and solar forecast"), "15min", SIGNAL_ROLES,
        ("solar_a01", "wind_onshore_a01", "wind_offshore_a01"),
        "solar_a01 + wind_onshore_a01 + wind_offshore_a01",
    ),
    "wind_solar_a40": _derived(
        "wind_solar_aggregate", "MW", "Combined wind and solar A40 forecast.",
        ("wind solar a40", "08:00 wind and solar forecast"), "15min", SIGNAL_ROLES,
        ("solar_a40", "wind_onshore_a40", "wind_offshore_a40"),
        "solar_a40 + wind_onshore_a40 + wind_offshore_a40",
    ),
    "wind_solar_actual": _derived(
        "wind_solar_aggregate", "MW", "Observed combined wind and solar generation.",
        ("wind solar actual", "actual wind and solar", "wind and solar generation"),
        "15min", EX_POST_ROLES,
        ("solar_actual_mw", "wind_onshore_actual_mw", "wind_offshore_actual_mw"),
        "solar_actual_mw + wind_onshore_actual_mw + wind_offshore_actual_mw",
    ),
    "wind_solar_revision": _derived(
        "forecast_revision", "MW", "Combined wind and solar forecast revision.",
        ("wind and solar revision", "wind solar revision", "combined wind and solar revision"),
        "15min", SIGNAL_ROLES, ("wind_solar_a40", "wind_solar_a01"),
        "wind_solar_a40 - wind_solar_a01",
    ),
    "solar_a01_deviation": _derived(
        "forecast_deviation", "MW", "Solar actual minus A01 forecast.",
        ("solar a01 deviation", "day-ahead solar deviation"), "15min", EX_POST_ROLES,
        ("solar_actual_mw", "solar_a01"), "solar_actual_mw - solar_a01",
    ),
    "solar_a40_deviation": _derived(
        "forecast_deviation", "MW", "Solar actual minus A40 forecast.",
        ("solar a40 deviation", "08:00 solar deviation"), "15min", EX_POST_ROLES,
        ("solar_actual_mw", "solar_a40"), "solar_actual_mw - solar_a40",
    ),
    "wind_onshore_a01_deviation": _derived(
        "forecast_deviation", "MW", "Onshore wind actual minus A01 forecast.",
        ("onshore wind a01 deviation",), "15min", EX_POST_ROLES,
        ("wind_onshore_actual_mw", "wind_onshore_a01"),
        "wind_onshore_actual_mw - wind_onshore_a01",
    ),
    "wind_onshore_a40_deviation": _derived(
        "forecast_deviation", "MW", "Onshore wind actual minus A40 forecast.",
        ("onshore wind a40 deviation",), "15min", EX_POST_ROLES,
        ("wind_onshore_actual_mw", "wind_onshore_a40"),
        "wind_onshore_actual_mw - wind_onshore_a40",
    ),
    "wind_offshore_a01_deviation": _derived(
        "forecast_deviation", "MW", "Offshore wind actual minus A01 forecast.",
        ("offshore wind a01 deviation",), "15min", EX_POST_ROLES,
        ("wind_offshore_actual_mw", "wind_offshore_a01"),
        "wind_offshore_actual_mw - wind_offshore_a01",
    ),
    "wind_offshore_a40_deviation": _derived(
        "forecast_deviation", "MW", "Offshore wind actual minus A40 forecast.",
        ("offshore wind a40 deviation",), "15min", EX_POST_ROLES,
        ("wind_offshore_actual_mw", "wind_offshore_a40"),
        "wind_offshore_actual_mw - wind_offshore_a40",
    ),
    "total_wind_a01_deviation": _derived(
        "forecast_deviation", "MW", "Total wind actual minus A01 forecast.",
        ("total wind a01 deviation",), "15min", EX_POST_ROLES,
        ("total_wind_actual", "total_wind_a01"),
        "total_wind_actual - total_wind_a01",
    ),
    "total_wind_a40_deviation": _derived(
        "forecast_deviation", "MW", "Total wind actual minus A40 forecast.",
        ("total wind a40 deviation",), "15min", EX_POST_ROLES,
        ("total_wind_actual", "total_wind_a40"),
        "total_wind_actual - total_wind_a40",
    ),
    "wind_solar_a01_deviation": _derived(
        "forecast_deviation", "MW", "Wind and solar actual minus A01 forecast.",
        ("wind solar a01 deviation", "wind and solar a01 deviation"),
        "15min", EX_POST_ROLES, ("wind_solar_actual", "wind_solar_a01"),
        "wind_solar_actual - wind_solar_a01",
    ),
    "wind_solar_a40_deviation": _derived(
        "forecast_deviation", "MW", "Wind and solar actual minus A40 forecast.",
        ("wind solar a40 deviation", "wind and solar a40 deviation"),
        "15min", EX_POST_ROLES, ("wind_solar_actual", "wind_solar_a40"),
        "wind_solar_actual - wind_solar_a40",
    ),
    "delivery_hour": _derived(
        "time_dimension", "integer", "Market-local delivery hour.",
        ("delivery hour", "hour"), "15min", FILTER_ROLES,
        ("delivery_start_local",), "hour(delivery_start_local)",
    ),
    "quarter_hour": _derived(
        "time_dimension", "integer",
        "Minute component of the market-local delivery start, one of 0, 15, 30, or 45.",
        ("quarter hour", "quarter-hour"), "15min", FILTER_ROLES,
        ("delivery_start_local",), "minute(delivery_start_local)",
    ),
    "weekday": _derived(
        "time_dimension", "integer", "Market-local day of the week.",
        ("weekday", "day of week"), "15min", FILTER_ROLES,
        ("delivery_start_local",), "weekday(delivery_start_local)",
    ),
    "month": _derived(
        "time_dimension", "integer", "Market-local delivery month.",
        ("month", "delivery month"), "15min", FILTER_ROLES,
        ("delivery_start_local",), "month(delivery_start_local)",
    ),
    "year": _derived(
        "time_dimension", "integer", "Market-local delivery year.",
        ("year", "delivery year"), "15min", FILTER_ROLES,
        ("delivery_start_local",), "year(delivery_start_local)",
    ),
    "season": _derived(
        "time_dimension", "category", "Season derived from market-local month.",
        ("season", "delivery season"), "15min", FILTER_ROLES,
        ("delivery_start_local",), "season(delivery_start_local)",
    ),
}
