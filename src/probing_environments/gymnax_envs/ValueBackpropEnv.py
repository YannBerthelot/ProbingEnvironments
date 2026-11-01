"""ValueBackpropEnv"""

from typing import Optional, Tuple

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
    time: int = 0


@struct.dataclass
class EnvParams:
    """Environment parameters"""

    max_steps_in_episode: int = 1


class ValueBackpropEnv(environment.Environment):
    """
    One action, random +1/-1 observation, one timestep long, obs-dependent \
    +1/-1 reward every time: If my agent can learn the value in ProbeEnv1 but not \
    this one - meaning it can learn a constant reward but not a \
    predictable one! - it must be that backpropagation through my network is broken.
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
        reward = state.x
        state = EnvState(x=state.x, time=state.time + 1)  # type: ignore
        terminated = self.is_terminated(state, params)

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            terminated,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        obs = jax.random.choice(key, jnp.array([0.0, 1.0]))
        state = EnvState(x=obs, time=0)  # type: ignore
        return self.get_obs(state), state

    def get_obs(
        self,
        state: EnvState,
        params: Optional[EnvParams] = None,
        key: Optional[chex.PRNGKey] = None,
    ) -> chex.Array:
        """Applies observation function to state."""
        return jnp.array([state.x])

    @property
    def name(self) -> str:
        """Environment name."""
        return "ValueLossOrOptimizerEnv"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        # Derive from action_space so the two can never disagree.
        return int(self.action_space().n)

    def is_terminated(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        return True

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(1)

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
