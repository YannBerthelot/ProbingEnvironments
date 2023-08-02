"""PolicyAndValueEnv"""
from typing import Any, Optional, Tuple

import chex
import jax
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


class PolicyAndValueEnv(environment.Environment):
    """
    Two actions, random +1/-1 observation, one timestep long, action-and-obs \
    dependent +1/-1 reward: Now we've got a dependence on both obs and action.\
    The policy and value networks interact here, so there's a couple of things \
    to verify: that the policy network learns to pick the right action in each \
    of the two states, and that the value network learns that the value of \
    each state is +1. If everything's worked up until now, then if - for \
    example - the value network fails to learn here, it likely means your \
    batching process is feeding the value network stale experience.
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
        done = self.is_terminal(state, params)
        reward = jax.lax.cond(
            (jnp.array_equal(jnp.array([action]), state.x)), lambda: 1.0, lambda: 0.0
        )

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
        obs = jax.random.randint(key, (1,), jnp.array(0), jnp.array(2))
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
        return spaces.Discrete(2)

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
