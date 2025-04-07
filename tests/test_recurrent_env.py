"""Test that the recurrent env works properly. """
import jax
import jax.numpy as jnp

from probing_environments.gymnax_envs.RecurrentValueEnv import (
    EnvState,
    RecurrentValueEnv,
)


def test_recurrent_env():
    """Test that the recurrent env works proerly. Should return state 0 or 1 on frame 1\
    , 2 on frame 2, and 3 on frame 3. Should always return a reward of 0,\
    unless on frame 3 if the initial state was 1"""
    env = RecurrentValueEnv()
    key = jax.random.PRNGKey(42)
    _ = None

    
    for init_obs in (0.0,1.0):
        _, env_state = env.reset(key, _)
        env_state = EnvState(  # type: ignore
            x=jnp.float32(0) + 1.0, t=0, original_state=init_obs
        )
        env.original_state = init_obs
        obs, new_env_state, reward, done, _ = env.step(key, env_state, _)
        assert obs == 2.0
        assert reward == 0.0
        obs, _, reward, done, _ = env.step(key, new_env_state, _)
        assert done
        assert reward == 0.0 if init_obs == 0 else reward == 1.0

    