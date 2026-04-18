"""Check that the continuous gymnax environments work as expected"""

import jax

from probing_environments.gymnax_envs.continuous_actions import (
    AdvantagePolicyLossPolicyUpdateEnv,
    PolicyAndValueEnv,
    RewardDiscountingEnv,
    ValueBackpropEnv,
    ValueLossOrOptimizerEnv,
)


def test_AdvantagePolicyLossPolicyUpdateEnv_works():
    """Check that this environment yields the expected results (see env docstring)"""
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_act, key_step = jax.random.split(rng, 4)

    env = AdvantagePolicyLossPolicyUpdateEnv()
    params = env.default_params
    obs, state = env.reset(key_reset, params)
    assert obs.shape == (1,)
    assert obs == 1
    for _ in range(10):
        key_act, _rng = jax.random.split(key_act)
        action = env.action_space(params).sample(_rng)
        assert env.action_space(params).contains(action)
    for _ in range(10):
        key_act, _rng = jax.random.split(key_act)
        action = env.action_space(params).sample(_rng)
        obs, state = env.reset(key_reset, params)
        n_obs, new_state, reward, done, info = env.step(key_step, state, action, params)
        assert done
        assert action == reward


def test_PolicyAndValueEnv_works():
    """Check that this environment yields the expected results (see env docstring)"""
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_act, key_step = jax.random.split(rng, 4)

    env = PolicyAndValueEnv()
    params = env.default_params
    for _ in range(10):
        key_reset, _rng = jax.random.split(key_reset)
        obs, state = env.reset(_rng, params)
        assert obs in (-1, 1)
    for _ in range(10):
        key_act, _rng = jax.random.split(key_act)
        action = env.action_space(params).sample(_rng)
        assert -1 < action[0].item() < 1

    for action in (-1, 1):
        rewards = 0
        for _ in range(10):
            key_reset, _rng = jax.random.split(key_reset)
            obs, state = env.reset(_rng, params)
            n_obs, new_state, reward, done, info = env.step(
                key_step, state, action, params
            )
            rewards += reward
            assert done
            if (obs > 0.0 and action > 0.0) or (obs <= 0.0 and action <= 0.0):
                assert reward == 1
            else:
                assert reward == -1
        assert -10 <= rewards <= 10


def test_ValueLossOrOptimizerEnv_continuous_works():
    """Check that this environment yields the expected results (see env docstring)"""
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_act, key_step = jax.random.split(rng, 4)

    env = ValueLossOrOptimizerEnv()
    params = env.default_params
    obs, state = env.reset(key_reset, params)
    assert obs == 0
    action = env.action_space(params).sample(key_act)
    n_obs, new_state, reward, done, info = env.step(key_step, state, action, params)
    assert done
    assert n_obs == 0
    assert reward == 1


def test_ValueBackpropEnv_continuous_works():
    """Check that this environment yields the expected results (see env docstring)"""
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_act, key_step = jax.random.split(rng, 4)

    env = ValueBackpropEnv()
    params = env.default_params
    for _ in range(10):
        key_reset, _rng = jax.random.split(key_reset)
        obs, state = env.reset(_rng, params)
        assert obs in (0, 1)
        action = env.action_space(params).sample(key_act)
        n_obs, new_state, reward, done, info = env.step(key_step, state, action, params)
        assert done
        assert reward == obs


def test_RewardDiscountingEnv_continuous_works():
    """Check that this environment yields the expected results (see env docstring)"""
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_act, key_step = jax.random.split(rng, 4)

    env = RewardDiscountingEnv()
    params = env.default_params
    obs, state = env.reset(key_reset, params)
    for t in range(1, 3):
        action = env.action_space(params).sample(key_act)
        n_obs, state, reward, done, info = env.step(key_step, state, action, params)
        if t == 2:
            assert done
            assert reward == 1
        else:
            assert reward == 0
            assert not done
