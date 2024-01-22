"""Check that the gymnax environments work as expected"""
import jax

from probing_environments.gymnax_envs.continuous_actions import (
    AdvantagePolicyLossPolicyUpdateEnv,
    PolicyAndValueEnv,
)


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
        assert env.action_space(_).contains(action)
    # Perform the step transition.
    for _ in range(10):
        key_act, _rng = jax.random.split(key_act)
        action = env.action_space(_).sample(_rng)
        _, state, reward, done, _ = env.step(key_step, state, action, _)
        assert done
        assert action == reward


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
        assert 0 < action[0].item() < 1
    # Perform the step transition.

    for action in (0, 1):
        rewards = 0
        for _ in range(10):
            key_reset, _rng = jax.random.split(key_reset)
            obs, state = env.reset(_rng, _)
            _, state, reward, done, _ = env.step(key_step, state, action, _)
            rewards += reward
            assert done
            obs = int(obs)
            if (obs > 0.5 and action > 0.5) or (obs <= 0.5 and action <= 0.5):
                assert reward == 1
            else:
                assert reward == 0
        assert 0 < rewards < 10
