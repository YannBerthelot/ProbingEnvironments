"""
Premade tests including the initialization of the agent, the training and the
parameter tests.
"""
from typing import Callable, List

import gym
import numpy as np
import pytest
from mypy_extensions import DefaultNamedArg

from probing_environments.envs import (
    AdvantagePolicyLossPolicyUpdateEnv,
    PolicyAndValueEnv,
    RewardDiscountingEnv,
    ValueBackpropEnv,
    ValueLossOrOptimizerEnv,
)
from probing_environments.types import AgentType

EPS = 1e-1
GAMMA = 0.5


def check_loss_or_optimizer_value_net(
    agent: AgentType,
    init_agent: Callable[
        [AgentType, gym.Env, DefaultNamedArg(float, "gamma")], AgentType
    ],
    train_agent: Callable[[AgentType, float], AgentType],
    get_value: Callable[[AgentType, np.ndarray], np.ndarray],
    discrete: bool = True,
):
    """
    Train and test your agent on ProbeEnv1 : Check for problems in the loss calculation\
          or optimizer of the value network.

    Args:
        agent (AgentType) : The agent to be used
        init_agent (Callable[[gym.Env, Optional[float]], None ]): Init your agent on\
              a given Env and gamma/discount factor. See template.
        train_agent (Callable[[float], AgentType]): Train your agent for a given budget\
              See template.
        get_value (Callable[[AgentType, np.ndarray], np.ndarray]): Get value for a \
            given obs using your critic. See template.
        discrete (bool, optional): Wether or not to handle state as discrete. \
            Defaults to True.
    """
    env = ValueLossOrOptimizerEnv(discrete)
    agent = init_agent(agent, env)
    agent = train_agent(agent, int(1e3))
    expected_value = 1
    predicted_value = get_value(agent, env.reset())
    err_msg = (
        "There's most likely a problem with the value loss calculation or the"
        f" optimizer. Expected {expected_value} and got {predicted_value}"
    )
    assert pytest.approx(expected_value, abs=EPS) == predicted_value, err_msg


def check_backprop_value_net(
    agent: AgentType,
    init_agent: Callable[
        [AgentType, gym.Env, DefaultNamedArg(float, "gamma")], AgentType
    ],
    train_agent: Callable[[AgentType, float], AgentType],
    get_value: Callable[[AgentType, np.ndarray], np.ndarray],
    discrete: bool = True,
):
    """
    Train and test your agent on ProbeEnv2 : Check for problems in the backprop of your\
          value net.

    Args:
        agent (AgentType) : The agent to be used
        init_agent (Callable[[gym.Env, float]]): Init your agent on a given Env and \
            gamma/discount factor. See template.
        train_agent (Callable[[float], AgentType]): Train your agent for a given \
            budget. See template.
        get_value (Callable[[AgentType, np.ndarray], np.ndarray]): Get value for a \
            given obs using your critic. See template.
        discrete (bool, optional): Wether or not to handle state as discrete. \
            Defaults to True.
    """
    agent = init_agent(agent, ValueBackpropEnv(discrete))
    agent = train_agent(agent, int(1e3))

    if discrete:
        expected_value = 0
        predicted_value = get_value(agent, np.array(0))
        err_msg = (
            "There is most lilely a problem with the backprop in your value network."
            f" Expected a value of {expected_value}, got {predicted_value}"
        )
        assert (
            pytest.approx(expected_value, abs=EPS, rel=EPS) == predicted_value
        ), err_msg

        predicted_value = get_value(agent, np.array(1))
        expected_value = 1
        err_msg = (
            "There is most lilely a problem with the backprop in your value network."
            f" Expected a value of {expected_value}, got {predicted_value}"
        )
        assert (
            pytest.approx(expected_value, abs=EPS, rel=EPS) == predicted_value
        ), err_msg
    else:
        expected_value = 0
        predicted_value = get_value(agent, np.array([0, 0, 0]))
        err_msg = (
            "There is most lilely a problem with the backprop in your value network."
            f" Expected a value of {expected_value}, got {predicted_value}"
        )
        assert (
            pytest.approx(expected_value, abs=EPS, rel=EPS) == predicted_value
        ), err_msg

        expected_value = 1
        predicted_value = get_value(agent, np.array([1, 1, 1]))
        err_msg = (
            "There is most lilely a problem with the backprop in your value network."
            f" Expected a value of {expected_value}, got {predicted_value}"
        )
        assert (
            pytest.approx(expected_value, abs=EPS, rel=EPS) == predicted_value
        ), err_msg


def check_reward_discounting(
    agent: AgentType,
    init_agent: Callable[
        [AgentType, gym.Env, DefaultNamedArg(float, "gamma")], AgentType
    ],
    train_agent: Callable[[AgentType, float], AgentType],
    get_value: Callable[[AgentType, np.ndarray], np.ndarray],
    get_gamma: Callable[[AgentType], float],
    discrete: bool = True,
):
    """
    Train and test yout agent on ProbeEnv3: Check problems in the reward discounting\
          computation

    Args:
        agent (AgentType) : The agent to be used
        init_agent (Callable[[gym.Env, float]]): Init your agent on a given Env and \
            gamma/discount factor. See template.
        train_agent (Callable[[float], AgentType]): Train your agent for a given budget\
              See template.
        get_value (Callable[[AgentType, np.ndarray], np.ndarray]): Get value for a \
            given obs using your critic. See template.
        get_gamma (Callable[[AgentType], float]): Get the current value of \
            gamma/discount factor or your agent. See template.
        discrete (bool, optional): Wether or not to handle state as discrete. \
            Defaults to True.
    """
    agent = init_agent(agent, RewardDiscountingEnv(discrete), gamma=0.5)
    agent = train_agent(agent, int(1e3))
    expected_value = get_gamma(agent)
    if discrete:
        predicted_value = get_value(agent, np.array([0]))
        err_msg = (
            "There is most likely a problem with your reward discounting. Expected a"
            f" value of {expected_value} but got {predicted_value}"
        )
        assert (
            pytest.approx(expected_value, abs=EPS, rel=EPS) == predicted_value
        ), err_msg
    else:
        predicted_value = get_value(agent, np.array([0, 0, 0]))
        err_msg = (
            "There is most likely a problem with your reward discounting. Expected a"
            f" value of {expected_value} but got {predicted_value}"
        )
        assert (
            pytest.approx(expected_value, abs=EPS, rel=EPS) == predicted_value
        ), err_msg
    expected_value = 1
    if discrete:
        predicted_value = get_value(agent, np.array([1]))
        err_msg = (
            "There is most likely a problem with your reward discounting. Expected a"
            f" value of {expected_value} but got {predicted_value}"
        )
        assert (
            pytest.approx(expected_value, abs=EPS, rel=EPS) == predicted_value
        ), err_msg
    else:
        predicted_value = get_value(agent, np.array([1, 1, 1]))
        err_msg = (
            "There is most likely a problem with your reward discounting. Expected a"
            f" value of {expected_value} but got {predicted_value}"
        )
        assert (
            pytest.approx(expected_value, abs=EPS, rel=EPS) == predicted_value
        ), err_msg


def check_advantage_policy(
    agent: AgentType,
    init_agent: Callable[
        [AgentType, gym.Env, DefaultNamedArg(float, "gamma")], AgentType
    ],
    train_agent: Callable[[AgentType, float], AgentType],
    get_policy: Callable[[AgentType, np.ndarray], np.ndarray],
    discrete: bool = True,
):
    """
    Train and test your agent on ProbeEnv4: Check problems in the advantage computation\
        , the policy update or the policy loss.

    Args:
        agent (AgentType): The agent to be used
        init_agent (Callable[ [AgentType, gym.Env, DefaultNamedArg): Init your agent on\
              a given Env and gamma/discount factor. See template.
        train_agent (Callable[[AgentType, float], AgentType]): Train your agent for a \
            given budget. See template.
        get_action (Callable[[AgentType, np.ndarray], np.ndarray]): Get action for a \
            given obs using your actor. See template.
        discrete (bool, optional): Wether or not to handle state as discrete. \
            Defaults to True.
    """
    env = AdvantagePolicyLossPolicyUpdateEnv(discrete)
    agent = init_agent(agent, env, gamma=0.5)
    agent = train_agent(agent, int(1e3))
    excepted_action = 0
    action_probabilities = get_policy(agent, env.reset())
    err_msg = (
        "There is most likely a problem with your reward advantage computing or your"
        " policy loss or your policy update. Expected the actor to select"
        f" {excepted_action} with at least 90% chance but got"
        f" {action_probabilities[0]*100}%"
    )
    assert action_probabilities[0] > 0.90, err_msg


def check_batching_process(
    agent: AgentType,
    init_agent: Callable[
        [AgentType, gym.Env, DefaultNamedArg(float, "gamma")], AgentType
    ],
    train_agent: Callable[[AgentType, float], AgentType],
    get_policy: Callable[[AgentType, np.ndarray], List[float]],
    get_value: Callable[[AgentType, np.ndarray], np.ndarray],
    discrete: bool = True,
):
    """
    Train and test your agent on ProbeEnv4: Check problems in the advantage computation\
        , the policy update or the policy loss.

    Args:
        agent (AgentType): The agent to be used
        init_agent (Callable[ [AgentType, gym.Env, DefaultNamedArg): Init your agent on\
              a given Env and gamma/discount factor. See template.
        train_agent (Callable[[AgentType, float], AgentType]): Train your agent for a \
            given budget. See template.
        get_action (Callable[[AgentType, np.ndarray], np.ndarray]): Get action for a \
            given obs using your actor. See template.
        get_value (Callable[[AgentType, np.ndarray], np.ndarray]): Get value for a \
            given obs using your critic. See template.
        discrete (bool, optional): Wether or not to handle state as discrete. \
            Defaults to True.
    """
    env = PolicyAndValueEnv(discrete)
    agent = init_agent(agent, env)
    agent = train_agent(agent, int(1e3))

    if discrete:
        action_probabilities = get_policy(agent, np.array([0]))
        assert action_probabilities[0] > 0.9, (
            "Expected action 0 to have at least 90% probability, got"
            f" {action_probabilities[0]*100}%"
        )
        action_probabilities = get_policy(agent, np.array([1]))
        assert action_probabilities[1] > 0.9, (
            "Expected action 1 to have at least 90% probability, got"
            f" {action_probabilities[1]*100}%"
        )

        predicted_value = get_value(agent, np.array([0]))
        assert predicted_value == pytest.approx(
            1, abs=EPS
        ), f"Expected value of 1, got {predicted_value}"
        predicted_value = get_value(agent, np.array([1]))
        assert predicted_value == pytest.approx(
            1, abs=EPS
        ), f"Expected value of 1, got {predicted_value}"

    else:
        action_probabilities = get_policy(agent, np.array([0, 0, 0]))
        assert action_probabilities[0] > 0.9, (
            "Expected action 0 to have at least 90% probability, got"
            f" {action_probabilities[0]*100}%"
        )
        action_probabilities = get_policy(agent, np.array([1, 1, 1]))
        assert action_probabilities[1] > 0.9, (
            "Expected action 1 to have at least 90% probability, got"
            f" {action_probabilities[1]*100}%"
        )
        predicted_value = get_value(agent, np.array([0, 0, 0]))
        assert predicted_value == pytest.approx(
            1, abs=EPS
        ), f"Expected value of 1, got {predicted_value}"
        predicted_value = get_value(agent, np.array([1, 1, 1]))
        assert predicted_value == pytest.approx(
            1, abs=EPS
        ), f"Expected value of 1, got {predicted_value}"
