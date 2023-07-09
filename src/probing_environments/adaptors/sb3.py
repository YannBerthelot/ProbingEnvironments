"""
Connectors template for your agent.
"""
from typing import List, Optional

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv


def init_agent(
    agent: OnPolicyAlgorithm,
    env: gym.Env,
    run_name: str,  # pylint: disable=W0613
    gamma: Optional[float] = 0.5,
    learning_rate: Optional[float] = 1e-3,
    num_envs: Optional[int] = None,
    seed: Optional[int] = 42,
) -> OnPolicyAlgorithm:
    """
    Initialize your agent on a given env while also setting the discount factor.

    Args:
        agent (OnPolicyAlgorithm) : The agent to be used
        env (gym.Env): The env to use with your agent.
        gamma (float, optional): The discount factor to use. Defaults to 0.5.

    Raises:
        NotImplementedError: While you haven't implemented your own functions or picked\
              from the existing ones

    Returns:
        AgentType: Your agent with the right settings.
    """

    def make_env():
        return env(num_envs)

    set_random_seed(seed=seed)
    if num_envs is not None and num_envs > 1:
        vec_env = make_vec_env(make_env, n_envs=num_envs, vec_env_cls=DummyVecEnv)
        return agent(
            "MlpPolicy", vec_env, gamma=gamma, learning_rate=learning_rate, seed=seed
        )
    else:
        return agent(
            "MlpPolicy", env(), gamma=gamma, learning_rate=learning_rate, seed=seed
        )


def train_agent(
    agent: OnPolicyAlgorithm, budget: Optional[int] = int(1e3)
) -> OnPolicyAlgorithm:
    """
    Train your agent for a given budget/number of timesteps.

    Args:
        agent (AgentType): Your agent (created by init_agent)
        budget (int, optional): The number of timesteps to train the agent on. Defaults\
              to int(1e3).

    Raises:
        NotImplementedError: While you haven't implemented your own functions or picked\
              from the existing ones

    Returns:
        AgentType: Your trained agents.
    """
    return agent.learn(budget)


def get_value(agent: OnPolicyAlgorithm, obs: np.ndarray) -> np.ndarray:
    """
    Predict the value of a given obs (in numpy array format) using your current value \
        net.

    Args:
        agent (AgentType): Your agent to make the prediction.
        obs (np.ndarray): The observation to make the prediction on.

    Raises:
        NotImplementedError: While you haven't implemented your own functions or picked\
              from the existing ones

    Returns:
        np.ndarray: The predicted value of the given observation.
    """
    return (
        agent.policy.predict_values(torch.tensor(np.array([obs])))
        .detach()
        .numpy()[0][0]
    )


def get_policy(agent: OnPolicyAlgorithm, obs: np.ndarray) -> List[float]:
    """
    Predict the probability of actions of a given obs (in numpy array format)\
          using your current policy net.

    Args:
        agent (AgentType): Your agent to make the prediction.
        obs (np.ndarray): The observation to make the prediction on.

    Raises:
        NotImplementedError: While you haven't implemented your own functions or picked\
              from the existing ones

    Returns:
        List[float]: The probability of taking every action.
    """
    dis = agent.policy.get_distribution(torch.tensor(np.array([obs])))
    probs = dis.distribution.probs
    probs_np = probs.detach().numpy()
    return probs_np[0]


def get_gamma(agent: OnPolicyAlgorithm) -> float:
    """
    Fetch the gamma/discount factor value from your agent (to use it in tests)

    Args:
        agent (AgentType): Your agent.

    Raises:
        NotImplementedError: While you haven't implemented your own functions or picked\
              from the existing ones

    Returns:
        float: The gamma/discount factor value of your agent
    """
    return agent.gamma
