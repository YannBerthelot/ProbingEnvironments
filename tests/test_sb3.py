"Unit tests for the tests ... :)"
from typing import Any

import pytest
from stable_baselines3.a2c import A2C

from probing_environments.adaptors.sb3 import (
    get_gamma,
    get_policy,
    get_value,
    init_agent,
    train_agent,
)
from probing_environments.checks import (
    check_actor_and_critic_coupling,
    check_advantage_policy,
    check_backprop_value_net,
    check_loss_or_optimizer_value_net,
    check_reward_discounting,
)

AgentType = Any
AGENT = A2C
LEARNING_RATE = 1e-3
BUDGET = 2e3


def test_check_loss_or_optimizer_value_net():
    """
    Test that check_loss_or_optimizer_value_net works on failproof sb3.
    """
    for discrete in (True, False):
        check_loss_or_optimizer_value_net(
            AGENT,
            init_agent,
            train_agent,
            get_value,
            discrete=discrete,
            learning_rate=LEARNING_RATE,
            budget=BUDGET,
        )


def test_check_backprop_value_net():
    """
    Test that check_backprop_value_net works on failproof sb3.
    """
    for discrete in (True, False):
        check_backprop_value_net(
            AGENT,
            init_agent,
            train_agent,
            get_value,
            discrete=discrete,
            learning_rate=LEARNING_RATE,
            budget=BUDGET,
        )


def test_check_reward_discounting():
    """
    Test that check_reward_discounting works on failproof sb3.
    """
    for discrete in (True, False):
        check_reward_discounting(
            AGENT,
            init_agent,
            train_agent,
            get_value,
            get_gamma,
            discrete=discrete,
            learning_rate=LEARNING_RATE,
            budget=BUDGET,
        )


def test_check_advantage_policy():
    """
    Test that check_advantage_policy works on failproof sb3.
    """
    for discrete in (True, False):
        check_advantage_policy(
            AGENT,
            init_agent,
            train_agent,
            get_policy,
            discrete=discrete,
            learning_rate=LEARNING_RATE,
            budget=BUDGET,
        )


def test_check_actor_and_critic_coupling():
    """
    Test that check_actor_and_critic_coupling works on failproof sb3.
    """
    for discrete in (True, False):
        check_actor_and_critic_coupling(
            AGENT,
            init_agent,
            train_agent,
            get_policy,
            get_value,
            discrete=discrete,
            learning_rate=LEARNING_RATE,
            budget=BUDGET,
        )


def test_errors():
    """
    Run all tests on sb3 (that we assume to work) and make sure they pass and don't\
          return any bugs.
    """
    with pytest.raises(AssertionError):
        check_loss_or_optimizer_value_net(
            AGENT, init_agent, train_agent, get_value=lambda x, y: -1
        )
    with pytest.raises(AssertionError):
        check_backprop_value_net(
            AGENT, init_agent, train_agent, get_value=lambda x, y: -1
        )
    with pytest.raises(AssertionError):
        check_reward_discounting(
            AGENT,
            init_agent,
            train_agent,
            get_value=lambda x, y: -1,
            get_gamma=get_gamma,
        )

    with pytest.raises(AssertionError):
        check_advantage_policy(
            AGENT,
            init_agent,
            train_agent,
            get_policy=lambda x, y: [0.1, 0.9],
        )

    with pytest.raises(AssertionError):
        check_actor_and_critic_coupling(
            AGENT,
            init_agent,
            train_agent,
            get_policy=lambda x, y: [0.1, 0.9],
            get_value=lambda x, y: -1,
        )
