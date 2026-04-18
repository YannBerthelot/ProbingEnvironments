"""Test that the recurrent env works properly."""
import jax
import jax.numpy as jnp

from probing_environments.gymnax_envs.RecurrentValueEnv import (
    EnvState,
    RecurrentValueEnv,
)


def test_recurrent_env():
    """Test that the recurrent env works properly.
    Two timesteps: initial obs is 0 or 1.
    Step 1: reward=0, obs=2.0
    Step 2: reward=1.0 if original_state was 1, else 0. Terminal."""
    env = RecurrentValueEnv()
    key = jax.random.PRNGKey(42)
    params = env.default_params

    for init_obs in (0.0, 1.0):
        _, env_state = env.reset(key, params)
        env_state = EnvState(
            x=init_obs, time=0, original_state=init_obs
        )
        # Step 1: time 0->1, reward=0, not terminal
        obs, env_state, reward, done, info = env.step(key, env_state, 0, params)
        assert not done
        assert reward == 0.0
        assert info["obs_st"] == jnp.array([2.0])

        # Step 2: time 1->2, terminal, reward depends on original_state
        obs, env_state, reward, done, info = env.step(key, env_state, 0, params)
        assert done
        if init_obs == 0.0:
            assert reward == 0.0
        else:
            assert reward == 1.0
