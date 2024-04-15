import jax

from probing_environments.gymnax_envs.average_reward.two_choice_MDP import (
    EnvParams,
    TwoChoiceMDP,
)


def test_TwoChoiceMDPWorks():
    """Check that this environment yields the expected results (see env docstring)"""
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_act, key_step = jax.random.split(rng, 4)

    # Instantiate the environment & its settings.
    env = TwoChoiceMDP()
    params = None
    # Reset the environment.
    obs, state = env.reset(key_reset, params)
    assert obs == 0
    # Sample a random action.
    action = env.action_space(params).sample(key_act)
    assert action in (0, 1)

    # Perform the step transition.
    expected_obs = {0: [1, 2, 3, 4], 1: [5, 6, 7, 8]}
    for action in (0, 1):
        for i in range(4):

            n_obs, state, reward, done, _ = env.step(key_step, state, action, params)
            assert n_obs == expected_obs[action][i]
            if i == 0 and action == 0:
                assert reward == 1
            elif action == 1 and i == 3:
                assert reward == 2
            else:
                assert reward == 0
        n_obs, state, reward, done, _ = env.step(key_step, state, action, params)
        assert reward == 0
        assert n_obs == 0

    max_steps = 3
    env_params = EnvParams(max_steps_in_episode=max_steps)
    obs, state = env.reset(key_reset, env_params)
    for _ in range(max_steps):
        n_obs, state, reward, done, _ = env.step(key_step, state, action, env_params)
    assert done
