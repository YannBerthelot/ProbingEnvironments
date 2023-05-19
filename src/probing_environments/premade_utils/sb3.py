"""
Connectors template for your agent.
"""
from typing import Optional

import gym
import numpy as np
import torch
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm


def init_agent(
    agent: OnPolicyAlgorithm, env: gym.Env, gamma: Optional[float] = 0.5
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
    return agent("MlpPolicy", env, gamma=gamma)


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


def get_action(agent: OnPolicyAlgorithm, obs: np.ndarray) -> int:
    """
    Predict the action of a given obs (in numpy array format) using your current policy\
         net.

    Args:
        agent (AgentType): Your agent to make the prediction.
        obs (np.ndarray): The observation to make the prediction on.

    Raises:
        NotImplementedError: While you haven't implemented your own functions or picked\
              from the existing ones

    Returns:
        int: The predicted action for the given observation.
    """
    return agent.predict(obs)[0]


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
