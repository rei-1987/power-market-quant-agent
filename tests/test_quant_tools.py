from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from tools.quant_tools import (
    ConditionalProbabilityResult,
    DescriptiveStatsResult,
    apply_condition,
    conditional_probability,
    descriptive_stats,
)


@pytest.mark.parametrize(
    ("operator", "value", "expected"),
    [
        (">", 2, [False, False, True, False]),
        ("<", 2, [True, False, False, False]),
        (">=", 2, [False, True, True, False]),
        ("<=", 2, [True, True, False, False]),
        ("==", 2, [False, True, False, False]),
        ("=", 2, [False, True, False, False]),
        ("between", (1, 2), [True, True, False, False]),
    ],
)
def test_apply_condition_operators(operator, value, expected) -> None:
    source = pd.DataFrame({"value": [1.0, 2.0, 3.0, None]})

    result = apply_condition(source, "value", operator, value)

    assert result.tolist() == expected
    assert result.dtype == bool


def test_apply_condition_rejects_missing_column_and_invalid_operator() -> None:
    source = pd.DataFrame({"value": [1]})

    with pytest.raises(ValueError, match="column is missing"):
        apply_condition(source, "missing", ">", 0)
    with pytest.raises(ValueError, match="Unsupported condition operator"):
        apply_condition(source, "value", "!=", 0)
    with pytest.raises(ValueError, match="two-item"):
        apply_condition(source, "value", "between", (1,))


def test_strong_positive_conditional_probability_effect() -> None:
    source = pd.DataFrame(
        {
            "id1_minus_id3": [1] * 90 + [-1] * 10 + [1] * 10 + [-1] * 90,
            "total_wind_revision": [-1] * 100 + [1] * 100,
        }
    )

    result = conditional_probability(
        source,
        event_variable="id1_minus_id3",
        event_operator=">",
        event_value=0,
        condition_variable="total_wind_revision",
        condition_operator="<",
        condition_value=0,
    )

    assert isinstance(result, ConditionalProbabilityResult)
    assert result.valid_sample_size == 200
    assert result.condition_sample_size == 100
    assert result.baseline_sample_size == 100
    assert result.condition_probability == pytest.approx(0.9)
    assert result.baseline_probability == pytest.approx(0.1)
    assert result.probability_difference_percentage_points == pytest.approx(80.0)
    assert result.confidence_interval_95_percentage_points[0] > 70.0
    assert result.p_value < 0.001


def test_no_effect_and_complement_baseline() -> None:
    source = pd.DataFrame(
        {
            "event": [1, 0, 1, 0, 1, 0, 1, 0],
            "signal": [-1, -1, -1, -1, 1, 1, 1, 1],
        }
    )

    result = conditional_probability(
        source,
        event_variable="event",
        event_operator="=",
        event_value=1,
        condition_variable="signal",
        condition_operator="<",
        condition_value=0,
    )

    assert result.condition_event_count == 2
    assert result.baseline_event_count == 2
    assert result.condition_sample_size + result.baseline_sample_size == 8
    assert result.condition_probability == result.baseline_probability == 0.5
    assert result.probability_difference_percentage_points == 0.0
    assert result.p_value == 1.0


def test_conditional_probability_drops_missing_required_values_and_preserves_caller() -> None:
    source = pd.DataFrame(
        {
            "event": [1.0, 0.0, None, 1.0, 0.0],
            "signal": [-1.0, -1.0, -1.0, None, 1.0],
            "unrelated": [None, 1, 2, 3, 4],
        }
    )
    original = source.copy(deep=True)

    result = conditional_probability(
        source,
        event_variable="event",
        event_operator=">",
        event_value=0,
        condition_variable="signal",
        condition_operator="<",
        condition_value=0,
    )

    assert result.valid_sample_size == 3
    assert result.dropped_missing_count == 2
    assert result.condition_sample_size == 2
    assert result.baseline_sample_size == 1
    pd.testing.assert_frame_equal(source, original)


def test_conditional_probability_rejects_empty_effective_and_groups() -> None:
    kwargs = {
        "event_variable": "event",
        "event_operator": ">",
        "event_value": 0,
        "condition_variable": "signal",
        "condition_operator": "<",
        "condition_value": 0,
    }

    with pytest.raises(ValueError, match="no valid rows"):
        conditional_probability(
            pd.DataFrame({"event": [None], "signal": [None]}), **kwargs
        )
    with pytest.raises(ValueError, match="condition group is empty"):
        conditional_probability(
            pd.DataFrame({"event": [1, 0], "signal": [1, 2]}), **kwargs
        )
    with pytest.raises(ValueError, match="baseline group is empty"):
        conditional_probability(
            pd.DataFrame({"event": [1, 0], "signal": [-1, -2]}), **kwargs
        )


def test_conditional_probability_rejects_missing_required_column() -> None:
    with pytest.raises(ValueError, match="Required calculation column is missing"):
        conditional_probability(
            pd.DataFrame({"event": [1]}),
            event_variable="event",
            event_operator=">",
            event_value=0,
            condition_variable="signal",
            condition_operator="<",
            condition_value=0,
        )


def test_descriptive_statistics_and_immutability() -> None:
    source = pd.DataFrame({"value": [1.0, 2.0, 3.0, None]})
    original = source.copy(deep=True)

    result = descriptive_stats(source, "value")

    assert isinstance(result, DescriptiveStatsResult)
    assert result.sample_size == 3
    assert result.missing_count == 1
    assert result.mean == 2.0
    assert result.median == 2.0
    assert result.std == 1.0
    assert result.min == 1.0
    assert result.max == 3.0
    with pytest.raises(FrozenInstanceError):
        result.mean = 3.0  # type: ignore[misc]
    pd.testing.assert_frame_equal(source, original)


def test_descriptive_statistics_reject_invalid_samples() -> None:
    with pytest.raises(ValueError, match="column is missing"):
        descriptive_stats(pd.DataFrame({"value": [1]}), "missing")
    with pytest.raises(ValueError, match="numeric column"):
        descriptive_stats(pd.DataFrame({"value": ["one"]}), "value")
    with pytest.raises(ValueError, match="no valid numeric observations"):
        descriptive_stats(pd.DataFrame({"value": [None]}, dtype="float64"), "value")
