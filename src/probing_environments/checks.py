"""
Premade tests including the initialization of the agent, the training and the
parameter tests.
"""

from typing import Any, Callable

import gym
import numpy as np
import pytest
from mypy_extensions import DefaultNamedArg

from probing_environments.envs import ProbeEnv1, ProbeEnv2, ProbeEnv3

EPS = 1e-1
GAMMA = 0.5

AgentType = Any


def check_loss_or_optimizer_value_net(
    init_agent: Callable[[gym.Env, DefaultNamedArg(float, "gamma")], AgentType],
    train_agent: Callable[[AgentType, float], AgentType],
    get_value: Callable[[AgentType, np.ndarray], np.ndarray],
):
    """
    Train and test your agent on ProbeEnv1 : Check for problems in the loss calculation\
          or optimizer of the value network.

    Args:
        init_agent (Callable[[gym.Env, Optional[float]], None ]): Init your agent on\
              a given Env and gamma/discount factor. See template.
        train_agent (Callable[[float], AgentType]): Train your agent for a given budget\
              See template.
        get_value (Callable[[AgentType, np.ndarray], np.ndarray]): Get value for a \
            given obs using your critic. See template.
    """
    agent = init_agent(ProbeEnv1())
    agent = train_agent(agent, int(1e3))

    try:
        assert pytest.approx(1, abs=EPS) == get_value(agent, agent.get_env().reset())
    except AssertionError:
        print(
            "There's most likely a problem with the value         loss"
            " calculation or the optimizer"
        )


def check_backprop_value_net(
    init_agent: Callable[[gym.Env, DefaultNamedArg(float, "gamma")], None],
    train_agent: Callable[[AgentType, float], AgentType],
    get_value: Callable[[AgentType, np.ndarray], np.ndarray],
):
    """
    Train and test your agent on ProbeEnv2 : Check for problems in the backprop of your\
          value net.

    Args:
        init_agent (Callable[[gym.Env, float]]): Init your agent on a given Env and \
            gamma/discount factor. See template.
        train_agent (Callable[[float], AgentType]): Train your agent for a given \
            budget. See template.
        get_value (Callable[[AgentType, np.ndarray], np.ndarray]): Get value for a \
            given obs using your critic. See template.
    """
    agent = init_agent(ProbeEnv2())
    agent = train_agent(agent, int(1e3))
    try:
        assert pytest.approx(0, abs=EPS) == get_value(agent, np.array(0))
        assert pytest.approx(1, abs=EPS) == get_value(agent, np.array(1))
    except AssertionError:
        print("There is most lilely a problem with the backprop in your value network")


def check_reward_discounting(
    init_agent: Callable[[gym.Env, DefaultNamedArg(float, "gamma")], None],
    train_agent: Callable[[AgentType, float], AgentType],
    get_value: Callable[[AgentType, np.ndarray], np.ndarray],
    get_gamma: Callable[[AgentType], float],
):
    """
    Train and test yout agent on ProbeEnv3: Check problems in the reward discounting\
          computation

    Args:
        init_agent (Callable[[gym.Env, float]]): Init your agent on a given Env and \
            gamma/discount factor. See template.
        train_agent (Callable[[float], AgentType]): Train your agent for a given budget\
              See template.
        get_value (Callable[[AgentType, np.ndarray], np.ndarray]): Get value for a \
            given obs using your critic. See template.
        get_gamma (Callable[[AgentType], float]): Get the current value of \
            gamma/discount factor or your agent. See template.
    """
    agent = init_agent(ProbeEnv3(), gamma=0.5)
    agent = train_agent(agent, int(1e3))
    try:
        assert pytest.approx(get_gamma(agent), rel=EPS) == get_value(
            agent, np.array([0])
        )
        assert pytest.approx(1, rel=EPS) == get_value(agent, np.array([1]))
    except AssertionError:
        print("There is most likely a problem with your reward discounting")
