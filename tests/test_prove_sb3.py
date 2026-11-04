"""``prove`` through a real library: SB3's PPO must pass discounting when
healthy and fail it at gamma 1, with no edit to SB3 (contract: test_prove.py).

The settings were fixed before the first run: PPO on the package's existing
2,000-step budget, with the existing adaptor's ``n_steps`` and learning rate.
"""

import pytest
from stable_baselines3 import PPO

from probing_environments.adaptors.sb3 import SB3
from probing_environments.faults import no_discount
from probing_environments.probes import discounting
from probing_environments.runner import prove, provisional


@pytest.mark.medium
def test_ppo_proves_discounting_catches_no_discount() -> None:
    agent = SB3(PPO, n_steps=32, learning_rate=1e-3)

    got = prove(discounting(0.8), agent, no_discount, provisional(at=2000))

    assert got.healthy.status == "passed", got.healthy.report
    assert got.faulted.status == "flagged", got.faulted.report
    assert "no discount" in got.faulted.report
    assert got.proven


def test_a_setting_the_library_refuses_is_a_crash_not_a_catch() -> None:
    probe = discounting(0.8)

    got = provisional(at=1, seeds=(0,))(
        probe.queries, SB3(PPO).reader(probe, not_a_setting=1.0)
    )

    assert got.status == "crashed"
    assert "not_a_setting" in got.report
