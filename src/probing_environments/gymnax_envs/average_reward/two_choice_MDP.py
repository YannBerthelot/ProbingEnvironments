"""AdvantagePolicyLossPolicyUpdateEnv"""
from typing import Optional, Tuple

import chex
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces
from jax import lax


# pylint: disable=W0613
@struct.dataclass
class EnvState:
    """Represents the state of the env in gymnax format"""

    x: int
    time: int


@struct.dataclass
class EnvParams:
    """Environment parameters (unused here)"""

    max_steps_in_episode: int = 1000


class TwoChoiceMDP(environment.Environment):
    """
    Two actions, zero observation, one timestep long, action-dependent +1/-1\
    reward: The first env to exercise the policy! If my agent can't learn \
    to pick the better action, there's something wrong with either my \
    advantage calculations, my policy loss or my policy update. That's three \
    things, but it's easy to work out by hand the expected values for each one \
    and check that the values produced by your actual code line up with them.
    """

    def __init__(self):
        """Define the spaces shape"""
        super().__init__()
        self.obs_shape = (1,)
        self.action_shape = (1,)

    @property
    def default_params(self) -> EnvParams:
        """Get default params for the env"""
        return EnvParams()

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Performs step transitions in the environment."""

        next_x = lax.cond(
            jnp.array_equal(state.x, 0),
            lambda: lax.cond(
                jnp.array_equal(action, 0), lambda: 1, lambda: 5
            ),  # If in state 0 and action 0 then move to 1, if action is 1 then move to 5
            lambda: lax.cond(
                jnp.logical_or(
                    jnp.array_equal(state.x, 4), jnp.array_equal(state.x, 8)
                ),
                lambda: 0,
                lambda: state.x + 1,
            ),  # If not in state 0, if state is 4 or 8, move to 0, else move to state + 1
        )
        time = state.time + 1
        state = EnvState(x=next_x, time=time)
        reward = lax.cond(
            jnp.array_equal(state.x, 1),
            lambda: 1,
            lambda: lax.cond(jnp.array_equal(state.x, 8), lambda: 2, lambda: 0),
        )
        done = self.is_terminal(state, params)
        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        state = EnvState(x=0, time=0)  # type: ignore
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        """Applies observation function to state."""
        return jnp.array([state.x])

    @property
    def name(self) -> str:
        """Environment name."""
        return "TwoChoiceMDP"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 1

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        return state.time >= params.max_steps_in_episode

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(2)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Discrete(9, dtype=jnp.int32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "x": spaces.Discrete(9, dtype=jnp.int32),
                "time": spaces.Discrete(params.max_steps_in_episode, dtype=jnp.int32),
            }
        )
