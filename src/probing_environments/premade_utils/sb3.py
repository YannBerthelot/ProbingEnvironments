"""
Connectors template for your agent.
"""
from typing import Any

import gym
import numpy as np
import torch
from stable_baselines3.a2c import A2C

AgentType = Any


def init_agent(env: gym.Env, gamma: float = 0.5) -> AgentType:
    """
    Initialize your agent on a given env while also setting the discount factor.

    Args:
        env (gym.Env): The env to use with your agent.
        gamma (float, optional): The discount factor to use. Defaults to 0.5.

    Raises:
        NotImplementedError: While you haven't implemented your own functions or picked\
              from the existing ones

    Returns:
        AgentType: Your agent with the right settings.
    """
    return A2C("MlpPolicy", env, gamma=gamma)


def train_agent(agent: AgentType, budget: int = int(1e3)) -> AgentType:
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
    raise agent.learn(budget)


def get_value(agent: AgentType, obs: np.ndarray) -> np.ndarray:
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


def get_gamma(agent: AgentType) -> float:
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