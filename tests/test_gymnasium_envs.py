"""The Gymnasium environments follow Gymnasium's seeding contract.

Contract (Gymnasium's ``Env.reset`` and ``Env.np_random`` docs): ``reset(seed=s)``
seeds the environment's own generator, ``np_random``, and a later ``reset()``
without a seed continues that generator instead of reseeding it. So the
observations after a seeded reset are reproducible, the seed chooses them,
and resetting leaves numpy's global generator to the agent that seeded it.
"""

import numpy as np
import pytest

from probing_environments import envs

ALL_ENVS = [
    envs.ValueLossOrOptimizerEnv,
    envs.ValueBackpropEnv,
    envs.RewardDiscountingEnv,
    envs.AdvantagePolicyLossPolicyUpdateEnv,
    envs.AdvantagePolicyLossPolicyUpdateEnvContinuous,
    envs.PolicyAndValueEnv,
    envs.PolicyAndValueEnvContinuous,
    envs.TwoChoiceMDP,
]
# The envs that start each episode from a random observation, 0 or 1.
RANDOM_START_ENVS = [
    envs.ValueBackpropEnv,
    envs.PolicyAndValueEnv,
    envs.PolicyAndValueEnvContinuous,
]
# 30 fair draws: two independent streams agree with probability 2**-30.
N_RESETS = 30


def _start_observations(env_cls, seed):
    env = env_cls()
    first, _ = env.reset(seed=seed)
    return [first.item()] + [env.reset()[0].item() for _ in range(N_RESETS - 1)]


@pytest.mark.parametrize("env_cls", RANDOM_START_ENVS)
def test_unseeded_resets_continue_the_seeded_stream(env_cls):
    assert _start_observations(env_cls, seed=3) == _start_observations(env_cls, 3)


@pytest.mark.parametrize("env_cls", RANDOM_START_ENVS)
def test_unseeded_resets_do_not_restart_the_stream(env_cls):
    # A stream restarted on every reset would repeat one observation.
    assert set(_start_observations(env_cls, seed=3)) == {0, 1}


@pytest.mark.parametrize("env_cls", RANDOM_START_ENVS)
def test_the_seed_chooses_the_start_observations(env_cls):
    assert _start_observations(env_cls, seed=3) != _start_observations(env_cls, 4)


@pytest.mark.parametrize("env_cls", ALL_ENVS)
def test_reset_leaves_numpys_global_generator_alone(env_cls):
    state = np.random.get_state()
    expected = np.random.random()
    np.random.set_state(state)

    env = env_cls()
    env.reset(seed=5)
    env.reset()

    assert np.random.random() == expected
