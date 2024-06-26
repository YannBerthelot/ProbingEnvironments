"""
Premade tests including the initialization of the agent, the training and the
parameter tests.
"""
from typing import Any, Callable, List, Optional

import gymnasium as gym
import jax.numpy as jnp
import numpy as np
import pytest
from mypy_extensions import DefaultNamedArg, NamedArg

from probing_environments.envs import (
    AdvantagePolicyLossPolicyUpdateEnv,
    AdvantagePolicyLossPolicyUpdateEnvContinuous,
    PolicyAndValueEnv,
    PolicyAndValueEnvContinuous,
    RewardDiscountingEnv,
    TwoChoiceMDP,
    ValueBackpropEnv,
    ValueLossOrOptimizerEnv,
)
from probing_environments.gymnax_envs import (
    AdvantagePolicyLossPolicyUpdateEnv as AdvantagePolicyLossPolicyUpdateEnv_gx,
)
from probing_environments.gymnax_envs import PolicyAndValueEnv as PolicyAndValueEnv_gx
from probing_environments.gymnax_envs import RecurrentValueEnv as RecurrentValueEnv_gx
from probing_environments.gymnax_envs import (
    RewardDiscountingEnv as RewardDiscountingEnv_gx,
)
from probing_environments.gymnax_envs import ValueBackpropEnv as ValueBackpropEnv_gx
from probing_environments.gymnax_envs import (
    ValueLossOrOptimizerEnv as ValueLossOrOptimizerEnv_gx,
)
from probing_environments.gymnax_envs.average_reward.two_choice_MDP import (
    TwoChoiceMDP as TwoChoiceMDP_gx,
)
from probing_environments.gymnax_envs.continuous_actions import (
    AdvantagePolicyLossPolicyUpdateEnv as AdvantagePolicyLossPolicyUpdateEnv_continuous_gx,
)
from probing_environments.gymnax_envs.continuous_actions import (
    PolicyAndValueEnv as PolicyAndValueEnv_continuous_gx,
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
    gymnax: bool = False,
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
    if gymnax:
        env = ValueLossOrOptimizerEnv_gx
    else:
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
        get_value(agent, np.array([0])),
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
    gymnax: bool = False,
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
    if gymnax:
        env = ValueBackpropEnv_gx
    else:
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
    gymnax: bool = False,
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
    if gymnax:
        env = RewardDiscountingEnv_gx
    else:
        env = RewardDiscountingEnv
    agent = init_agent(
        agent=agent,
        env=env,
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
    gymnax: bool = False,
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
    if gymnax:
        env = AdvantagePolicyLossPolicyUpdateEnv_gx
    else:
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
        actions_probas=get_policy(agent, np.array([0])),
        err_msg=err_msg,
    )


def check_advantage_policy_continuous(
    agent: AgentType,
    init_agent: InitAgentType,
    train_agent: Callable[[AgentType, float], AgentType],
    get_action: Callable[[AgentType, np.ndarray], np.ndarray],
    budget: float = int(2e3),
    learning_rate: Optional[float] = 1e-3,
    num_envs: Optional[int] = 1,
    gymnax: bool = False,
    key: Optional[Any] = None,
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
    if gymnax:
        env = AdvantagePolicyLossPolicyUpdateEnv_continuous_gx
    else:
        env = AdvantagePolicyLossPolicyUpdateEnvContinuous
    agent = init_agent(
        agent=agent,
        env=env,
        run_name="check_advantage_policy",
        gamma=0.5,
        learning_rate=learning_rate,
        num_envs=num_envs,
        budget=budget,
    )
    agent = train_agent(agent, budget)
    err_msg = (
        "There is most likely a problem with your reward advantage computing or your"
        " policy loss or your policy update "
    )
    if gymnax:
        action = get_action(agent, np.array([1.0]), key)
    else:
        action = get_action(agent, np.array([1.0]))

    assert action >= 0.90, (
        err_msg + f"Expected action to be at least 0.9, got {action=}"
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
    gymnax: bool = False,
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
    if gymnax:
        env = PolicyAndValueEnv_gx
    else:
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


def check_actor_and_critic_coupling_continuous(
    agent: AgentType,
    init_agent: InitAgentType,
    train_agent: Callable[[AgentType, float], AgentType],
    get_action: Callable[[AgentType, np.ndarray], float],
    get_value: Callable[[AgentType, np.ndarray], np.ndarray],
    budget: Optional[float] = int(2e3),
    learning_rate: Optional[float] = 1e-3,
    num_envs: Optional[int] = 1,
    gymnax: bool = False,
    key: Optional[Any] = None,
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
    if gymnax:
        env = PolicyAndValueEnv_continuous_gx
    else:
        env = PolicyAndValueEnvContinuous
    agent = init_agent(
        agent=agent,
        env=env,
        num_envs=num_envs,
        run_name="check_actor_and_critic_coupling",
        learning_rate=learning_rate,
    )
    agent = train_agent(agent, budget)
    if gymnax:
        action = get_action(agent, np.array([1]), key)
    else:
        action = get_action(agent, np.array([1]))
    assert action > 0.0, f"Expected action to be at least 0.5, got {action=}"
    value = get_value(agent, np.array([1]))
    assert value >= 0.8, f"Expected value to be greater than 0.8, got {value=}"
    if gymnax:
        action = get_action(agent, np.array([-1]), key)
    else:
        action = get_action(agent, np.array([-1]))

    assert (
        action <= 0.0
    ), f"Expected action to be less than or equal to 0.5, got {action=}"
    value = get_value(agent, np.array([-1]))
    assert value >= 0.8, f"Expected value to be greater than 0.8, got {value=}"


def check_recurrent_agent(
    agent: AgentType,
    init_agent: InitAgentType,
    train_agent: Callable[[AgentType, float], AgentType],
    get_value_recurrent: Callable[
        [AgentType, np.ndarray, bool, np.ndarray], np.ndarray
    ],
    init_hidden_state: Callable[..., np.ndarray],
    compute_next_critic_hidden: Callable[
        [AgentType, np.ndarray, bool, np.ndarray], np.ndarray
    ],
    budget: Optional[float] = int(2e3),
    learning_rate: Optional[float] = 1e-3,
    num_envs: Optional[int] = 1,
    gymnax: bool = False,
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
    if gymnax:
        env = RecurrentValueEnv_gx
    else:
        env = None  # TODO : add gym version of the env
    agent = init_agent(
        agent=agent,
        env=env,
        run_name="check_recurrent_env",
        num_envs=num_envs,
        gamma=0.5,
        learning_rate=learning_rate,
    )
    agent = train_agent(agent, budget)
    err_msg = "There is most likely a problem with your reward discounting."

    hidden = init_hidden_state()
    obs = jnp.array([0.0])
    next_hidden = compute_next_critic_hidden(agent, obs, False, hidden)
    assert_predicted_value_isclose_expected_value(
        expected_value=0.0,
        predicted_value=get_value_recurrent(agent, np.array([2.0]), False, next_hidden),
        err_msg=err_msg,
    )
    hidden = init_hidden_state()
    obs = jnp.array([1.0])
    next_hidden = compute_next_critic_hidden(agent, obs, False, hidden)
    assert_predicted_value_isclose_expected_value(
        expected_value=1.0,
        predicted_value=get_value_recurrent(agent, np.array([2.0]), False, next_hidden),
        err_msg=err_msg,
    )


def check_average_reward(
    agent: AgentType,
    init_agent: InitAgentType,
    train_agent: Callable[[AgentType, float], AgentType],
    get_action: Callable[[AgentType, np.ndarray], float],
    get_value: Callable[[AgentType, np.ndarray], np.ndarray],
    budget: Optional[float] = int(2e3),
    learning_rate: Optional[float] = 1e-3,
    num_envs: Optional[int] = 1,
    gymnax: bool = False,
    key: Optional[Any] = None,
):
    """
    TODO : Do this
    """
    if gymnax:
        env = TwoChoiceMDP_gx
    else:
        env = TwoChoiceMDP
    agent = init_agent(
        agent=agent,
        env=env,
        num_envs=num_envs,
        run_name="check_average_reward",
        learning_rate=learning_rate,
    )
    agent = train_agent(agent, budget)
    if gymnax:
        action = get_action(agent, 0, key)
    else:
        action = get_action(agent, 0)
    assert action == 1, f"Expected action to be 1, got {action=}"
    value = get_value(agent, 0, env().TIME_LIMIT)
    assert value == pytest.approx(
        2 / 5, rel=0.5
    ), f"Expected value to be close to {value/env().TIME_LIMIT}, got {value=}"
