"""Test that the recurrent env works properly. """
import jax

from probing_environments.gymnax_envs.RecurrentValueEnv import ReccurentValueEnv


def test_recurrent_env():
    """Test that the recurrent env works proerly. Should return state 0 or 1 on frame 1\
    , 2 on frame 2, and 3 on frame 3. Should always return a reward of 0,\
    unless on frame 3 if the initial state was 1"""
    env = ReccurentValueEnv()
    key_0 = jax.random.PRNGKey(0)
    key_1 = jax.random.PRNGKey(4)
    _ = None

    obs, env_state = env.reset(key_0, _)
    assert obs == 0.0
    obs, new_env_state, reward, done, _ = env.step(key_0, env_state, _)
    assert obs == 2.0
    assert reward == 0.0
    obs, _, reward, done, _ = env.step(key_0, new_env_state, _)
    assert done
    assert reward == 0.0

    obs, env_state = env.reset(key_1, _)
    assert obs == 1.0
    obs, new_env_state, reward, done, _ = env.step(key_1, env_state, _)
    assert obs == 2.0
    assert reward == 0.0
    obs, _, reward, done, _ = env.step(key_1, new_env_state, _)
    assert done
    assert reward == 1.0
