"""
Premade connectors for rlberry
"""
from typing import Any, Optional

import gym
import numpy as np
import torch
from rlberry.agents.torch import A2CAgent

AgentType = Any


def init_agent(env: gym.Env, gamma: Optional[float] = 0.5) -> AgentType:
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
    agent = A2CAgent(env, gamma=gamma, learning_rate=0.01)
    return agent


def train_agent(agent: A2CAgent, budget: Optional[int] = int(1e3)) -> AgentType:
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
    agent.fit(budget * 10)
    return agent


def get_value(agent: A2CAgent, obs: np.ndarray) -> np.ndarray:
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
    return agent.value_net(torch.tensor(np.array([obs])))[0][0].detach().numpy()


def get_gamma(agent: A2CAgent) -> float:
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
