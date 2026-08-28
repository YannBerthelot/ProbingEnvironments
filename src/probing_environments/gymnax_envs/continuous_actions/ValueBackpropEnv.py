"""ValueBackpropEnv — continuous action version.

One continuous action, random 0/1 observation, one timestep long,
obs-dependent reward. Tests that backpropagation through the value network
works.
"""

from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces
from jax import lax


@struct.dataclass
class EnvState:
    x: chex.Array
    time: int = 0


@struct.dataclass
class EnvParams:
    max_steps_in_episode: int = 1000


class ValueBackpropEnv(environment.Environment):
    """Continuous-action version: reward = obs value."""

    def __init__(self):
        super().__init__()
        self.obs_shape = (1,)
        self.action_shape = (1,)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
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
        obs = jax.random.choice(key, jnp.array([0.0, 1.0]))
        state = EnvState(x=obs, time=0)  # type: ignore
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        return jnp.array([state.x])

    @property
    def num_actions(self) -> int:
        return 1

    def is_terminated(self, state: EnvState, params: EnvParams) -> bool:
        return True

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        return spaces.Box(-1.0, 1.0, (1,), dtype=jnp.float32)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(0, 1, (1,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        return spaces.Dict({"x": spaces.Box(0, 0, (), jnp.float32)})
