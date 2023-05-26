"Unit tests for the tests ... :)"
from typing import Any

import pytest
from rlberry.agents.torch import A2CAgent

from probing_environments.adaptors.rlberry import (
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
AGENT = A2CAgent


def test_check_loss_or_optimizer_value_net():
    """
    Test that check_loss_or_optimizer_value_net works
    """
    check_loss_or_optimizer_value_net(
        AGENT, init_agent, train_agent, get_value, discrete=False
    )


def test_check_backprop_value_net():
    """
    Test that check_backprop_value_net works
    """
    check_backprop_value_net(AGENT, init_agent, train_agent, get_value, discrete=False)


def test_check_reward_discounting():
    """
    Test that check_reward_discounting works
    """
    check_reward_discounting(
        AGENT, init_agent, train_agent, get_value, get_gamma, discrete=False
    )


def test_check_advantage_policy():
    """
    Test that check_advantage_policy works
    """
    check_advantage_policy(AGENT, init_agent, train_agent, get_policy, discrete=False)


def test_check_actor_and_critic_coupling():
    """
    Test that check_actor_and_critic_coupling works
    """
    check_actor_and_critic_coupling(
        AGENT, init_agent, train_agent, get_policy, get_value, discrete=False
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
