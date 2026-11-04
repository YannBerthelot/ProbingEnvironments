"""Smoke checks on the installed package, one per install (RELEASING.md §2, step 4).

The release's package gate (``.github/workflows/package.yml``) installs the
built wheel in a fresh environment outside the checkout and runs this with
that environment's Python, once per install:

    python tests/release_smoke.py core   # probing-environments alone
    python tests/release_smoke.py sb3    # probing-environments[sb3]

It is a script, not a pytest module, so the suite never collects it.

Contract (the public modules and the extras in ``pyproject.toml``, and the
CHANGELOG's promises for 0.6.0):

core
    Importing the package and its checks needs neither Stable-Baselines3 nor
    torch. Every gymnax environment the package lists resets, and steps once
    with the gymnax 1.0 API: six values, ``terminated`` and ``truncated``
    reported apart. Every gymnasium environment resets and steps once.
sb3
    With the extra, the premade Stable-Baselines3 adaptor drives a check end
    to end: PPO learns that the value of the only observation of
    ValueLossOrOptimizerEnv is 1. ``prove`` runs end to end too: on one seed,
    PPO passes discounting and is caught at gamma 1.
"""

from __future__ import annotations

import os
import sys

# The late imports are the point: JAX must see JAX_PLATFORMS before it loads,
# and the sb3 check imports Stable-Baselines3, which the core install lacks.
# pylint: disable=wrong-import-position,import-outside-toplevel

# Before JAX is imported: the gate's runners have no GPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

import probing_environments.checks  # noqa: E402,F401
from probing_environments import envs as gymnasium_envs  # noqa: E402
from probing_environments.gymnax_envs import env_list  # noqa: E402
from probing_environments.gymnax_envs.average_reward.two_choice_MDP import (  # noqa: E402
    TwoChoiceMDP,
)
from probing_environments.gymnax_envs.continuous_actions import (  # noqa: E402
    env_list as continuous_env_list,
)

OPTIONAL_MODULES = ("stable_baselines3", "torch")


def core() -> None:
    loaded = [name for name in OPTIONAL_MODULES if name in sys.modules]
    assert not loaded, f"importing the package loaded the optional {loaded}"

    for env_cls in [*env_list, *continuous_env_list, TwoChoiceMDP]:
        env = env_cls()
        params = env.default_params
        key_reset, key_action, key_step = jax.random.split(jax.random.PRNGKey(0), 3)
        _, state = env.reset(key_reset, params)
        action = env.action_space(params).sample(key_action)
        out = env.step(key_step, state, action, params)
        assert len(out) == 6, f"{env_cls.__name__}.step returned {len(out)} values"
        print(f"ok {env_cls.__name__}")

    for name in (
        "ValueLossOrOptimizerEnv",
        "ValueBackpropEnv",
        "RewardDiscountingEnv",
        "AdvantagePolicyLossPolicyUpdateEnv",
        "AdvantagePolicyLossPolicyUpdateEnvContinuous",
        "PolicyAndValueEnv",
        "PolicyAndValueEnvContinuous",
        "TwoChoiceMDP",
    ):
        env = getattr(gymnasium_envs, name)()
        env.reset(seed=0)
        env.step(env.action_space.sample())
        print(f"ok gymnasium {name}")


def sb3() -> None:
    from stable_baselines3 import PPO

    from probing_environments.adaptors.sb3 import (
        SB3,
        get_value,
        init_agent,
        train_agent,
    )
    from probing_environments.checks import check_loss_or_optimizer_value_net
    from probing_environments.faults import no_discount
    from probing_environments.probes import discounting
    from probing_environments.runner import prove, provisional

    check_loss_or_optimizer_value_net(
        PPO, init_agent, train_agent, get_value, budget=2_000
    )
    print("ok sb3 adaptor")

    agent = SB3(PPO, n_steps=32, learning_rate=1e-3)
    rule = provisional(at=2_000, seeds=(0,))
    proof = prove(discounting(0.8), agent, no_discount, rule)
    assert proof.proven, f"{proof.healthy.report}\n{proof.faulted.report}"
    print("ok prove")


if __name__ == "__main__":
    {"core": core, "sb3": sb3}[sys.argv[1]]()
