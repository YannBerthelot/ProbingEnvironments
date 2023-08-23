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
    t: int
    original_state: float


@struct.dataclass
class EnvParams:
    """Environment parameters (unused here)"""

    unused: Optional[Any] = None


class ReccurentValueEnv(environment.Environment):
    """Single action. 0 or 1 initial observation. two timesteps long. +1 reward at\
          the end if the initial observation was 1. If the agent can learn values\
         in non-recurrent env but not in this one, it should be that it lacks memory."""

    def __init__(self):
        """Define the spaces shape"""
        super().__init__()
        self.obs_shape = (1,)
        self.action_shape = (1,)
        self.original_state = None

    @property
    def default_params(self) -> EnvParams:
        """Get default params for the env"""
        return EnvParams()

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Performs step transitions in the environment."""
        done = self.is_terminal(state, params)
        t = state.t
        t += 1
        reward = jax.lax.cond(
            t == 1,
            lambda: 0.0,
            lambda: jax.lax.cond(state.original_state == 0.0, lambda: 0.0, lambda: 1.0),
        )
        state = EnvState(  # type: ignore
            x=jnp.float32(t) + 1.0, t=t, original_state=state.original_state
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
        obs = jax.random.choice(key, jnp.array([0.0, 1.0]))
        state = EnvState(x=obs, t=0, original_state=obs)  # type: ignore
        self.original_state = obs
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
        return state.t == 1

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
