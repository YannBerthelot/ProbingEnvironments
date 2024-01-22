"""AdvantagePolicyLossPolicyUpdateEnv"""
from typing import Any, Optional, Tuple

import chex
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces
from jax import lax


# pylint: disable=W0613
@struct.dataclass
class EnvState:
    """Represents the state of the env in gymnax format"""

    x: float


@struct.dataclass
class EnvParams:
    """Environment parameters (unused here)"""

    unused: Optional[Any] = None


class AdvantagePolicyLossPolicyUpdateEnv(environment.Environment):
    """
    Two actions, zero observation, one timestep long, action-dependent\
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
        reward = action  # the higher the action, the better
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
        obs = 0
        state = EnvState(x=obs)  # type: ignore
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        """Applies observation function to state."""
        return jnp.array([state.x])

    @property
    def name(self) -> str:
        """Environment name."""
        return "ValueLossOrOptimizerEnv"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 1

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        return True

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Box(0, 1, (1,), dtype=jnp.float32)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 1, (1,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "x": spaces.Box(0, 0, (), jnp.float32),
            }
        )
