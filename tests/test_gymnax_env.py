"""Check that the gymnax environments work as expected"""

import jax

from probing_environments.gymnax_envs import (
    AdvantagePolicyLossPolicyUpdateEnv,
    PolicyAndValueEnv,
    RewardDiscountingEnv,
    ValueBackpropEnv,
    ValueLossOrOptimizerEnv,
)


def test_ValueLossOrOptimizerEnv_works():
    """Check that this environment yields the expected results (see env docstring)"""
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_act, key_step = jax.random.split(rng, 4)

    env = ValueLossOrOptimizerEnv()
    params = env.default_params
    obs, state = env.reset(key_reset, params)
    assert obs == 0
    action = env.action_space(params).sample(key_act)
    n_obs, new_state, reward, terminated, truncated, info = env.step(
        key_step, state, action, params
    )
    done = terminated | truncated
    assert done
    assert n_obs == 0
    assert reward == 1


def test_ValueBackpropEnv_works():
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
        n_obs, new_state, reward, terminated, truncated, info = env.step(
            key_step, state, action, params
        )
        done = terminated | truncated
        assert done
        assert n_obs in (0, 1)
        assert reward == obs


def test_RewardDiscountingEnv_works():
    """Check that this environment yields the expected results (see env docstring)"""
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_act, key_step = jax.random.split(rng, 4)

    env = RewardDiscountingEnv()
    params = env.default_params
    obs, state = env.reset(key_reset, params)
    for t in range(1, 3):
        assert obs in (0, 1, 2)
        action = env.action_space(params).sample(key_act)
        n_obs, state, reward, terminated, truncated, info = env.step(
            key_step, state, action, params
        )
        done = terminated | truncated
        assert n_obs in (0, 1, 2)
        if t == 2:
            assert done
            assert reward == 1
        else:
            assert reward == 0
            assert not done


def test_AdvantagePolicyLossPolicyUpdateEnv_works():
    """Check that this environment yields the expected results (see env docstring)"""
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_act, key_step = jax.random.split(rng, 4)

    env = AdvantagePolicyLossPolicyUpdateEnv()
    params = env.default_params
    obs, state = env.reset(key_reset, params)
    assert obs.shape == (1,)
    assert obs == 0
    for _ in range(10):
        key_act, _rng = jax.random.split(key_act)
        action = env.action_space(params).sample(_rng)
        assert action in (0, 1)
    for action in (0, 1):
        obs, state = env.reset(key_reset, params)
        n_obs, new_state, reward, terminated, truncated, info = env.step(
            key_step, state, action, params
        )
        done = terminated | truncated
        assert done
        assert action == 1 - reward


def test_PolicyAndValueEnv_works():
    """Check that this environment yields the expected results (see env docstring)"""
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_act, key_step = jax.random.split(rng, 4)

    env = PolicyAndValueEnv()
    params = env.default_params
    for _ in range(10):
        key_reset, _rng = jax.random.split(key_reset)
        obs, state = env.reset(_rng, params)
        assert obs in (0, 1)
    for _ in range(10):
        key_act, _rng = jax.random.split(key_act)
        action = env.action_space(params).sample(_rng)
        assert action in (0, 1)

    for action in (0, 1):
        rewards = 0
        for _ in range(10):
            key_reset, _rng = jax.random.split(key_reset)
            obs, state = env.reset(_rng, params)
            n_obs, new_state, reward, terminated, truncated, info = env.step(
                key_step, state, action, params
            )
            done = terminated | truncated
            rewards += reward
            assert done
            obs = obs.astype(int)
            if action == obs:
                assert reward == 1
            else:
                assert reward == 0
        assert 0 <= rewards <= 10
