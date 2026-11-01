"""The tolerance every value check applies.

Contract (checks.py: EPS = 0.1, used as both an absolute and a relative
tolerance, the rule pytest.approx(expected, abs=EPS, rel=EPS) applied before
0.6.0): a predicted value passes when every entry is within
max(0.1, 0.1 * |expected|) of the expected value; otherwise the check raises
AssertionError carrying its error message.
"""

import numpy as np
import pytest

from probing_environments.checks import assert_predicted_value_isclose_expected_value


@pytest.mark.parametrize(
    "expected, predicted",
    [
        (1.0, 1.09),  # tolerance max(0.1, 0.1) = 0.1
        (1.0, 0.91),
        (10.0, 10.9),  # the relative part dominates: tolerance 1.0
        (0.0, 0.09),  # the absolute part dominates: tolerance 0.1
        (1.0, np.array([1.05])),  # array readouts are judged entrywise
    ],
)
def test_accepts_a_value_within_tolerance(expected, predicted):
    assert_predicted_value_isclose_expected_value(expected, predicted, "unused")


@pytest.mark.parametrize(
    "expected, predicted",
    [
        (1.0, 1.11),
        (1.0, 0.89),
        (10.0, 11.1),
        (0.0, -0.11),
        (1.0, np.array([1.05, 1.2])),  # one entry out is enough to fail
        (1.0, float("nan")),
    ],
)
def test_rejects_a_value_outside_tolerance_with_the_given_message(expected, predicted):
    with pytest.raises(AssertionError, match="value net is broken"):
        assert_predicted_value_isclose_expected_value(
            expected, predicted, "value net is broken"
        )
