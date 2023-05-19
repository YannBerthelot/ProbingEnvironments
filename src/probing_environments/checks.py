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
    discrete: bool = False,
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
    env = ProbeEnv1(discrete)
    agent = init_agent(env)
    agent = train_agent(agent, int(1e3))
    expected_value = 1
    predicted_value = get_value(agent, env.reset())
    err_msg = (
        "There's most likely a problem with the value loss calculation or the"
        f" optimizer. Expected {expected_value} and got {predicted_value}"
    )
    assert pytest.approx(expected_value, abs=EPS) == predicted_value, err_msg


def check_backprop_value_net(
    init_agent: Callable[[gym.Env, DefaultNamedArg(float, "gamma")], None],
    train_agent: Callable[[AgentType, float], AgentType],
    get_value: Callable[[AgentType, np.ndarray], np.ndarray],
    discrete: bool = False,
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
    agent = init_agent(ProbeEnv2(discrete))
    agent = train_agent(agent, int(1e3))

    if discrete:
        expected_value = 0
        predicted_value = get_value(agent, np.array(0))
        err_msg = (
            "There is most lilely a problem with the backprop in your value network."
            f" Expected a value of {expected_value}, got {predicted_value}"
        )
        assert pytest.approx(0, abs=EPS) == get_value(agent, predicted_value), err_msg

        expected_value = get_value(agent, np.array(1))
        predicted_value = np.array(0)
        err_msg = (
            "There is most lilely a problem with the backprop in your value network."
            f" Expected a value of {expected_value}, got {predicted_value}"
        )
        assert pytest.approx(expected_value, abs=EPS) == get_value(
            agent, predicted_value
        ), err_msg
    else:
        expected_value = 0
        predicted_value = get_value(agent, np.array([0, 0, 0]))
        err_msg = (
            "There is most lilely a problem with the backprop in your value network."
            f" Expected a value of {expected_value}, got {predicted_value}"
        )
        assert pytest.approx(0, abs=EPS) == get_value(
            agent, np.array([0, 0, 0])
        ), err_msg

        expected_value = 1
        predicted_value = get_value(agent, np.array([1, 1, 1]))
        err_msg = (
            "There is most lilely a problem with the backprop in your value network."
            f" Expected a value of {expected_value}, got {predicted_value}"
        )
        assert pytest.approx(1, abs=EPS) == get_value(
            agent, np.array([1, 1, 1])
        ), err_msg


def check_reward_discounting(
    init_agent: Callable[[gym.Env, DefaultNamedArg(float, "gamma")], None],
    train_agent: Callable[[AgentType, float], AgentType],
    get_value: Callable[[AgentType, np.ndarray], np.ndarray],
    get_gamma: Callable[[AgentType], float],
    discrete: bool = True,
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
    agent = init_agent(ProbeEnv3(discrete), gamma=0.5)
    agent = train_agent(agent, int(1e3))
    expected_value = get_gamma(agent)
    if discrete:
        predicted_value = get_value(agent, np.array([0]))
        err_msg = (
            "There is most likely a problem with your reward discounting. Expected a"
            f" value of {expected_value} but got {predicted_value}"
        )
        assert pytest.approx(expected_value, rel=EPS) == predicted_value, err_msg
    else:
        predicted_value = get_value(agent, np.array([0, 0, 0]))
        err_msg = (
            "There is most likely a problem with your reward discounting. Expected a"
            f" value of {expected_value} but got {predicted_value}"
        )
        assert pytest.approx(expected_value, rel=EPS) == predicted_value, err_msg
    expected_value = 1
    if discrete:
        predicted_value = get_value(agent, np.array([1]))
        err_msg = (
            "There is most likely a problem with your reward discounting. Expected a"
            f" value of {expected_value} but got {predicted_value}"
        )
        assert pytest.approx(expected_value, rel=EPS) == predicted_value, err_msg
    else:
        predicted_value = get_value(agent, np.array([1, 1, 1]))
        err_msg = (
            "There is most likely a problem with your reward discounting. Expected a"
            f" value of {expected_value} but got {predicted_value}"
        )
        assert pytest.approx(expected_value, rel=EPS) == predicted_value, err_msg
