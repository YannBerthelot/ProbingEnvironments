"""RewardDiscountingEnv — continuous action version.

One continuous action, zero-then-one observation, two timesteps long,
+1 reward at the end. Tests reward discounting.
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
    x: float
    time: int = 0


@struct.dataclass
class EnvParams:
    max_steps_in_episode: int = 1000


class RewardDiscountingEnv(environment.Environment):
    """Continuous-action version: +1 reward only at terminal step."""

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
        t = state.time + 1
        state = EnvState(x=jnp.float32(t), time=t)  # type: ignore
        done = self.is_terminal(state, params)
        reward = jax.lax.cond(done, lambda: 1.0, lambda: 0.0)
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
        state = EnvState(x=0.0, time=0)  # type: ignore
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        return jnp.array([state.x])

    @property
    def num_actions(self) -> int:
        return 1

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        return state.time == 2

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        return spaces.Box(-1.0, 1.0, (1,), dtype=jnp.float32)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(0, 2, (1,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        return spaces.Dict({"x": spaces.Box(0, 0, (), jnp.float32)})
