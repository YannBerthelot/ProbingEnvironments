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

    # Instantiate the environment & its settings.
    env = ValueLossOrOptimizerEnv()
    _ = None
    # Reset the environment.
    obs, state = env.reset(key_reset, _)
    assert obs == 0
    # Sample a random action.
    action = env.action_space(_).sample(key_act)

    # Perform the step transition.
    n_obs, _, reward, done, _ = env.step(key_step, state, action, _)
    assert done
    assert n_obs == 0
    assert reward == 1


def test_ValueBackpropEnv_works():
    """Check that this environment yields the expected resutls (see env docstring)"""
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_act, key_step = jax.random.split(rng, 4)

    # Instantiate the environment & its settings.
    env = ValueBackpropEnv()
    _ = None
    # Reset the environment.
    for _ in range(10):
        key_reset, _rng = jax.random.split(key_reset)
        obs, state = env.reset(_rng, _)
        assert obs in (0, 1)
        # Sample a random action.
        action = env.action_space(_).sample(key_act)
        # Perform the step transition.
        n_obs, _, reward, done, _ = env.step(key_step, state, action, _)
        assert done
        assert n_obs in (0, 1)
        assert reward == obs


def test_RewardDiscountingEnv_works():
    """Check that this environment yields the expected resutls (see env docstring)"""
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_act, key_step = jax.random.split(rng, 4)

    # Instantiate the environment & its settings.
    env = RewardDiscountingEnv()
    _ = None
    # Reset the environment.

    obs, state = env.reset(key_reset, _)
    for t in range(1, 3):
        assert obs in (0, 1, 2)
        # Sample a random action.
        action = env.action_space(_).sample(key_act)
        # Perform the step transition.
        n_obs, state, reward, done, _ = env.step(key_step, state, action, _)

        assert n_obs in (0, 1, 2)
        if t == 2:
            assert done
            assert reward == 1
        else:
            assert reward == 0
            assert not done


def test_AdvantagePolicyLossPolicyUpdateEnv_works():
    """Check that this environment yields the expected resutls (see env docstring)"""
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_act, key_step = jax.random.split(rng, 4)

    # Instantiate the environment & its settings.
    env = AdvantagePolicyLossPolicyUpdateEnv()
    _ = None
    # Reset the environment.

    obs, state = env.reset(key_reset, _)
    assert obs.shape == (1,)
    assert obs == 0
    # Sample a random action.
    for _ in range(10):
        key_act, _rng = jax.random.split(key_act)
        action = env.action_space(_).sample(_rng)
        assert action in (0, 1)
    # Perform the step transition.
    for action in (0, 1):
        _, state, reward, done, _ = env.step(key_step, state, action, _)
        assert done
        assert action == 1 - reward


def test_PolicyAndValueEnv_works():
    """Check that this environment yields the expected resutls (see env docstring)"""
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_act, key_step = jax.random.split(rng, 4)

    # Instantiate the environment & its settings.
    env = PolicyAndValueEnv()
    _ = None
    # Reset the environment.
    for _ in range(10):
        key_act, _rng = jax.random.split(key_reset)
        obs, state = env.reset(_rng, _)
        assert obs in (0, 1)
    # Sample a random action.
    for _ in range(10):
        key_act, _rng = jax.random.split(key_act)
        action = env.action_space(_).sample(_rng)
        assert action in (0, 1)
    # Perform the step transition.

    for action in (0, 1):
        rewards = 0
        for _ in range(10):
            key_reset, _rng = jax.random.split(key_reset)
            obs, state = env.reset(_rng, _)
            _, state, reward, done, _ = env.step(key_step, state, action, _)
            rewards += reward
            assert done
            obs = obs.astype(int)
            if action == obs:
                assert reward == 1
            else:
                assert reward == 0
        assert 0 <= rewards <= 10
