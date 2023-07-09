"""
Premade tests including the initialization of the agent, the training and the
parameter tests.
"""
from typing import Callable, List, Optional

import gymnasium as gym
import numpy as np
import pytest
from mypy_extensions import DefaultNamedArg, NamedArg

from probing_environments.envs import (
    AdvantagePolicyLossPolicyUpdateEnv,
    PolicyAndValueEnv,
    RewardDiscountingEnv,
    ValueBackpropEnv,
    ValueLossOrOptimizerEnv,
)
from probing_environments.utils.type_hints import AgentType

EPS = 1e-1
GAMMA = 0.5
InitAgentType = Callable[
    [
        NamedArg(AgentType, "agent"),
        NamedArg(gym.Env, "env"),
        NamedArg(str, "run_name"),
        NamedArg(float, "learning_rate"),
        DefaultNamedArg(float, "gamma"),
        NamedArg(int, "num_envs"),
    ],
    AgentType,
]


def assert_predicted_value_isclose_expected_value(
    expected_value: float, predicted_value: float, err_msg: str
):
    """
    Check that the predicted value is close enough to the expected_value and return\
          an appropriate error message if it's not the case

    Args:
        expected_value (float): The expected value
        predicted_value (float): The value predicted by critic.
        err_msg (str): The error message pointing to where the cause may be if the\
              assertion fails in the specific test.
    """
    assert (
        pytest.approx(expected_value, abs=EPS, rel=EPS) == predicted_value
    ), f"{err_msg}. Expected a value of {expected_value}, got {predicted_value}"


def assert_action_proba_is_larger_than_threshold(
    expected_proba: float,
    expected_action: int,
    actions_probas: List[float],
    err_msg: str,
):
    """
    Check that the predicted probability for the expected_action is larger than a given\
          threshold and if not return a message to pinpoint where the bug most likely\
            originates from if possible.
    Args:
        expected_proba (float): The proability threhsold to be beaten.
        expected_action (int): The action for which we want to check probability.
        actions_probas (List[float]): The action probabilities to consider
        err_msg (str): The error message to pinpoint the probable source of the\
              potential problem
    """
    action_proba = actions_probas[expected_action]
    assert action_proba > expected_proba, (
        f"{err_msg}. Expected a probability larger than {expected_proba*100}% for"
        f" action {expected_action}, got {action_proba*100}%"
    )


def check_loss_or_optimizer_value_net(
    agent: AgentType,
    init_agent: InitAgentType,
    train_agent: Callable[[AgentType, float], AgentType],
    get_value: Callable[[AgentType, np.ndarray], np.ndarray],
    budget: Optional[int] = int(1e3),
    learning_rate: Optional[float] = 1e-3,
    num_envs: Optional[int] = 1,
):
    """
    Train and test your agent on ValueLossOrOptimizerEnv : Check for problems in the\
          loss calculation or optimizer of the value network.

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
    env = ValueLossOrOptimizerEnv
    agent = init_agent(
        agent=agent,
        env=env,
        run_name="check_loss_or_optimizer_value_net",
        learning_rate=learning_rate,
        num_envs=num_envs,
    )
    agent = train_agent(agent, budget)
    assert_predicted_value_isclose_expected_value(
        1,
        get_value(agent, env().reset()[0]),
        "There's most likely a problem with the value loss calculation or the"
        " optimizer",
    )


def check_backprop_value_net(
    agent: AgentType,
    init_agent: InitAgentType,
    train_agent: Callable[[AgentType, float], AgentType],
    get_value: Callable[[AgentType, np.ndarray], np.ndarray],
    budget: Optional[float] = int(2e3),
    learning_rate: Optional[float] = 1e-3,
    num_envs: Optional[int] = 1,
):
    """
    Train and test your agent on ValueBackpropEnv : Check for problems in the \
        backprop of your value net.

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
    env = ValueBackpropEnv
    agent = init_agent(
        agent=agent,
        env=env,
        num_envs=num_envs,
        run_name="check_backprop_value_net",
        learning_rate=learning_rate,
    )
    agent = train_agent(agent, budget)
    err_msg = "There is most lilely a problem with the backprop in your value network."
    assert_predicted_value_isclose_expected_value(
        0,
        get_value(agent, np.array([0])),
        err_msg,
    )
    assert_predicted_value_isclose_expected_value(
        1,
        get_value(agent, np.array([1])),
        err_msg,
    )


def check_reward_discounting(
    agent: AgentType,
    init_agent: InitAgentType,
    train_agent: Callable[[AgentType, float], AgentType],
    get_value: Callable[[AgentType, np.ndarray], np.ndarray],
    get_gamma: Callable[[AgentType], float],
    budget: Optional[float] = int(2e3),
    learning_rate: Optional[float] = 1e-3,
    num_envs: Optional[int] = 1,
):
    """
    Train and test yout agent on RewardDiscountingEnv: Check problems in the reward\
          discounting computation.

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
    agent = init_agent(
        agent=agent,
        env=RewardDiscountingEnv,
        run_name="check_reward_discounting",
        num_envs=num_envs,
        gamma=0.5,
        learning_rate=learning_rate,
    )
    agent = train_agent(agent, budget)
    err_msg = "There is most likely a problem with your reward discounting."
    assert_predicted_value_isclose_expected_value(
        expected_value=get_gamma(agent),
        predicted_value=get_value(agent, np.array([0])),
        err_msg=err_msg,
    )
    assert_predicted_value_isclose_expected_value(
        expected_value=1,
        predicted_value=get_value(agent, np.array([1])),
        err_msg=err_msg,
    )


def check_advantage_policy(
    agent: AgentType,
    init_agent: InitAgentType,
    train_agent: Callable[[AgentType, float], AgentType],
    get_policy: Callable[[AgentType, np.ndarray], np.ndarray],
    budget: Optional[float] = int(2e3),
    learning_rate: Optional[float] = 1e-3,
    num_envs: Optional[int] = 1,
):
    """
    Train and test your agent on AdvantagePolicyLossPolicyUpdateEnv: Check problems in\
          the advantage computation , the policy update or the policy loss.

    Args:
        agent (AgentType): The agent to be used
        init_agent (Callable[ [AgentType, gym.Env, DefaultNamedArg): Init your agent on\
              a given Env and gamma/discount factor. See template.
        train_agent (Callable[[AgentType, float], AgentType]): Train your agent for a \
            given budget. See template.
        get_policy (Callable[[AgentType, np.ndarray], List[float]]): Get action for a \
            given obs using your actor. See template.
        discrete (bool, optional): Wether or not to handle state as discrete. \
            Defaults to True.
    """
    env = AdvantagePolicyLossPolicyUpdateEnv
    agent = init_agent(
        agent=agent,
        env=env,
        run_name="check_advantage_policy",
        gamma=0.5,
        learning_rate=learning_rate,
        num_envs=num_envs,
    )
    agent = train_agent(agent, budget)
    err_msg = (
        "There is most likely a problem with your reward advantage computing or your"
        " policy loss or your policy update "
    )
    assert_action_proba_is_larger_than_threshold(
        expected_proba=0.90,
        expected_action=0,
        actions_probas=get_policy(agent, env().reset()[0]),
        err_msg=err_msg,
    )


def check_actor_and_critic_coupling(
    agent: AgentType,
    init_agent: InitAgentType,
    train_agent: Callable[[AgentType, float], AgentType],
    get_policy: Callable[[AgentType, np.ndarray], List[float]],
    get_value: Callable[[AgentType, np.ndarray], np.ndarray],
    budget: Optional[float] = int(2e3),
    learning_rate: Optional[float] = 1e-3,
    num_envs: Optional[int] = 1,
):
    """
    Train and test your agent on PolicyAndValueEnv: Check problems in the coupling of\
          actor and critic (possibly batching process for example).

    Args:
        agent (AgentType): The agent to be used
        init_agent (Callable[ [AgentType, gym.Env, DefaultNamedArg): Init your agent on\
              a given Env and gamma/discount factor. See template.
        train_agent (Callable[[AgentType, float], AgentType]): Train your agent for a \
            given budget. See template.
        get_policy (Callable[[AgentType, np.ndarray], List[float]]): Get action for a \
            given obs using your actor. See template.
        get_value (Callable[[AgentType, np.ndarray], np.ndarray]): Get value for a \
            given obs using your critic. See template.
        discrete (bool, optional): Wether or not to handle state as discrete. \
            Defaults to True.
    """
    env = PolicyAndValueEnv
    agent = init_agent(
        agent=agent,
        env=env,
        num_envs=num_envs,
        run_name="check_actor_and_critic_coupling",
        learning_rate=learning_rate,
    )
    agent = train_agent(agent, budget)
    assert_action_proba_is_larger_than_threshold(
        expected_proba=0.90,
        expected_action=0,
        actions_probas=get_policy(agent, np.array([0])),
        err_msg="",
    )
    assert_predicted_value_isclose_expected_value(
        expected_value=1,
        predicted_value=get_value(agent, np.array([0])),
        err_msg="",
    )
    assert_action_proba_is_larger_than_threshold(
        expected_proba=0.90,
        expected_action=1,
        actions_probas=get_policy(agent, np.array([1])),
        err_msg="",
    )
    assert_predicted_value_isclose_expected_value(
        expected_value=1,
        predicted_value=get_value(agent, np.array([1])),
        err_msg="",
    )
