"""
Adaptor for the Ajax JAX-based RL library.

Ajax agents (SAC, REDQ, PPO, AVG, ASAC, APO) use gymnax environments and JAX
for JIT-compiled training. This adaptor bridges Ajax's interface to the
ProbingEnvironments check functions.

The "agent" flowing through the adaptor is a dict with:
    - "agent_cls": the Ajax agent class (e.g. SAC)
    - "gamma": discount factor
    - "state": the trained agent state (populated after train_agent)
    - "env": the probing environment class
    - "env_params": the environment parameters
    - "key": JAX PRNG key
    - "kind": "q" for Q-function agents, "v" for V-function agents
"""

from typing import Any, List, Optional

import jax
import jax.numpy as jnp
import numpy as np

AgentDict = dict[str, Any]

# Agent classes that use V-function critics (PPO, APO)
_V_FUNCTION_AGENTS = set()
# Agent classes that use Q(s, a)-function critics (SAC, REDQ, ASAC, AVG)
_Q_FUNCTION_AGENTS = set()
# Agent classes that use a Q(s)->R^{n_actions} network (DQN, PQN). These
# are discrete-action and value-based: V(s) = max_a Q(s, a), policy =
# argmax. They share the get_value / get_policy logic; only their
# constructor kwargs differ (handled per-name in init_agent).
_DQN_AGENTS = set()
# On-policy agents (no replay buffer)
_ON_POLICY_AGENTS = set()
# Average-reward agents (no gamma)
_AVG_REWARD_AGENTS = set()


def _classify_agent(agent_cls):
    """Classify an agent by its architecture."""
    name = agent_cls.__name__
    if name in ("PPO", "APO"):
        _V_FUNCTION_AGENTS.add(name)
        _ON_POLICY_AGENTS.add(name)
    elif name in ("DQN", "PQN"):
        _DQN_AGENTS.add(name)
    else:
        _Q_FUNCTION_AGENTS.add(name)
    if name in ("ASAC", "APO", "AVG"):
        _AVG_REWARD_AGENTS.add(name)


def init_agent(
    agent: type,
    env,
    run_name: str = "",
    gamma: float = 0.5,
    learning_rate: float = 1e-3,
    num_envs: Optional[int] = 1,
    seed: int = 42,
    budget: Optional[int] = None,
) -> AgentDict:
    """Initialize an Ajax agent on a probing environment."""
    _classify_agent(agent)
    name = agent.__name__
    env_instance = env()
    env_params = env_instance.default_params
    # Override max_steps_in_episode so gymnax's heuristic (truncated = time >=
    # max_steps) doesn't misclassify natural terminations (is_terminal=True) as
    # truncations. The envs' is_terminal logic still controls episode length;
    # this just prevents the time-based clause from firing first.
    env_params = env_params.replace(max_steps_in_episode=10_000)

    if name == "PQN":
        # PQN: single Q-network, discrete actions, on-policy (no replay
        # buffer, no target network). Takes rollout kwargs instead.
        agent_instance = agent(
            env_id=env_instance,
            n_envs=num_envs or 1,
            learning_rate=learning_rate,
            architecture=("64", "relu", "64", "relu"),
            env_params=env_params,
            gamma=gamma,
            n_steps=16,
            n_epochs=4,
            num_minibatches=1,
        )
        return {
            "agent_cls": agent,
            "agent_instance": agent_instance,
            "gamma": gamma,
            "state": None,
            "env": env_instance,
            "env_params": env_params,
            "key": jax.random.PRNGKey(seed),
            "seed": seed,
            "kind": "dqn",
        }

    if name in _DQN_AGENTS:
        # DQN: single Q-network, discrete actions, replay buffer. Its
        # __init__ takes `learning_rate` / `architecture` (one network),
        # not the actor/critic-split kwargs the other agents use.
        agent_instance = agent(
            env_id=env_instance,
            n_envs=num_envs or 1,
            learning_rate=learning_rate,
            architecture=("64", "relu", "64", "relu"),
            env_params=env_params,
            gamma=gamma,
            learning_starts=100,
            buffer_size=10_000,
            batch_size=64,
            target_update_interval=100,
        )
        return {
            "agent_cls": agent,
            "agent_instance": agent_instance,
            "gamma": gamma,
            "state": None,
            "env": env_instance,
            "env_params": env_params,
            "key": jax.random.PRNGKey(seed),
            "seed": seed,
            "kind": "dqn",
        }

    common_kwargs = dict(
        env_id=env_instance,
        n_envs=num_envs or 1,
        actor_learning_rate=learning_rate,
        critic_learning_rate=learning_rate,
        env_params=env_params,
        actor_architecture=("64", "relu", "64", "relu"),
        critic_architecture=("64", "relu", "64", "relu"),
    )

    if name in _ON_POLICY_AGENTS:
        # PPO / APO: on-policy, no replay buffer
        common_kwargs.update(
            normalize_observations=False,
            normalize_rewards=False,
            n_steps=32,
            batch_size=32,
            n_epochs=4,
        )
        if name not in _AVG_REWARD_AGENTS:
            common_kwargs["gamma"] = gamma
    elif name == "ASAC":
        # ASAC: average-reward SAC, no gamma
        common_kwargs.update(
            buffer_size=10_000,
            batch_size=64,
            learning_starts=100,
            normalize_observations=False,
            normalize_rewards=False,
        )
    elif name == "AVG":
        # AVG: on-policy average-reward, no buffer
        common_kwargs.update(
            gamma=gamma,
            learning_starts=100,
        )
    else:
        # SAC / REDQ: off-policy with replay buffer
        common_kwargs.update(
            gamma=gamma,
            normalize_observations=False,
            normalize_rewards=False,
            learning_starts=100,
            buffer_size=10_000,
            batch_size=64,
        )

    agent_instance = agent(**common_kwargs)

    kind = "v" if name in _V_FUNCTION_AGENTS else "q"

    return {
        "agent_cls": agent,
        "agent_instance": agent_instance,
        "gamma": gamma,
        "state": None,
        "env": env_instance,
        "env_params": env_params,
        "key": jax.random.PRNGKey(seed),
        "seed": seed,
        "kind": kind,
    }


def train_agent(agent: AgentDict, budget: Optional[int] = int(1e3)) -> AgentDict:
    """Train the Ajax agent for the given number of timesteps."""
    result = agent["agent_instance"].train(
        seed=agent["seed"],
        n_timesteps=budget,
        logging_config=None,
    )
    # train() returns (state, metrics) tuple, vmapped over seeds.
    # We used a single seed, so extract the first (only) entry.
    state = jax.tree.map(lambda x: x[0], result[0])
    agent["state"] = state
    return agent


def get_value(agent: AgentDict, obs: np.ndarray) -> float:
    """Get the critic's value estimate for an observation.

    For Q-function agents (SAC, REDQ, AVG, ASAC): V(s) = Q(s, pi(s))
    For V-function agents (PPO, APO): V(s) directly from critic
    """
    state = agent["state"]
    obs_jax = jnp.array(obs, dtype=jnp.float32)
    if obs_jax.ndim == 0:
        obs_jax = obs_jax[None]
    # Ensure a batch dimension for agents whose networks require it
    # (e.g. AVG's L2-normalized encoder reduces along axis=1).
    obs_batched = obs_jax[None] if obs_jax.ndim == 1 else obs_jax

    if agent["kind"] == "v":
        # PPO/APO: critic directly outputs V(s)
        value = state.critic_state.apply_fn(state.critic_state.params, obs_batched)
        if value.ndim > 1:
            value = jnp.mean(value, axis=0)
        return float(value.squeeze())
    elif agent["kind"] == "dqn":
        # DQN: Q(s) -> R^{n_actions}; greedy state value is V(s) = max_a Q(s, a).
        pi = state.actor_state.apply_fn(state.actor_state.params, obs_batched)
        value = jnp.max(pi.q_values, axis=-1)
        return float(value.squeeze())
    else:
        # SAC/REDQ/AVG/ASAC: Q-function critic, compute V(s) = Q(s, pi(s))
        pi = state.actor_state.apply_fn(state.actor_state.params, obs_batched)
        action = pi.mean() if hasattr(pi, "mean") else pi.mode()
        x = jnp.concatenate([obs_batched, action], axis=-1)
        q_values = state.critic_state.apply_fn(state.critic_state.params, x)
        if q_values.ndim > 1:
            value = jnp.mean(q_values, axis=0)
        else:
            value = q_values
        return float(value.squeeze())


def get_policy(agent: AgentDict, obs: np.ndarray) -> List[float]:
    """Get action probabilities for a discrete-action observation."""
    state = agent["state"]
    obs_jax = jnp.array(obs, dtype=jnp.float32)
    if obs_jax.ndim == 0:
        obs_jax = obs_jax[None]

    pi = state.actor_state.apply_fn(state.actor_state.params, obs_jax)

    n_actions = int(agent["env"].num_actions)
    if n_actions <= 1:
        return [1.0]

    if agent["kind"] == "dqn":
        # DQN's policy is greedy over Q; report it as a one-hot over the
        # argmax action (exploration is a collection-time concern only).
        best = int(jnp.argmax(pi.q_values, axis=-1).squeeze())
        return [1.0 if i == best else 0.0 for i in range(n_actions)]

    probs = []
    for i in range(n_actions):
        a = jnp.array([float(i)])
        log_p = pi.log_prob(a)
        probs.append(float(jnp.exp(log_p)))

    total = sum(probs)
    if total > 0:
        probs = [p / total for p in probs]
    return probs


def get_action(agent: AgentDict, obs: np.ndarray, key=None) -> float:
    """Get the deterministic action for an observation (continuous)."""
    state = agent["state"]
    obs_jax = jnp.array(obs, dtype=jnp.float32)
    if obs_jax.ndim == 0:
        obs_jax = obs_jax[None]
    obs_batched = obs_jax[None] if obs_jax.ndim == 1 else obs_jax

    pi = state.actor_state.apply_fn(state.actor_state.params, obs_batched)
    action = pi.mean() if hasattr(pi, "mean") else pi.mode()
    return float(action.squeeze())


def get_gamma(agent: AgentDict) -> float:
    """Get the discount factor from the agent."""
    return agent["gamma"]
