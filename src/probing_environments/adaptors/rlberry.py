"""
Premade connectors for rlberry
"""
from typing import List, Optional

import gymnasium as gym
import numpy as np
import torch

from probing_environments.utils.type_hints import AgentType


def init_agent(
    agent: AgentType,
    env: gym.Env,
    gamma: Optional[float] = 0.5,
    learning_rate: Optional[float] = 1e-3,
) -> AgentType:
    """
    Initialize your agent on a given env while also setting the discount factor.

    Args:
        agent (AgentType) : The agent to be used
        env (gym.Env): The env to use with your agent.
        gamma (float, optional): The discount factor to use. Defaults to 0.5.

    Raises:
        NotImplementedError: While you haven't implemented your own functions or picked\
              from the existing ones

    Returns:
        AgentType: Your agent with the right settings.
    """
    agent = agent(env, gamma=gamma, learning_rate=learning_rate)
    return agent


def train_agent(agent: AgentType, budget: Optional[int] = int(1e3)) -> AgentType:
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
    return agent.value_net(torch.tensor(np.array([obs])))[0][0].detach().numpy()


def get_policy(agent: AgentType, obs: np.ndarray) -> List[float]:
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
    return agent._policy_old(torch.tensor(np.array([obs]))).probs.detach().numpy()[0]


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
