"Unit tests for the tests ... :)"
from typing import Any

import pytest
from stable_baselines3.a2c import A2C

from probing_environments.checks import (
    check_backprop_value_net,
    check_loss_or_optimizer_value_net,
    check_reward_discounting,
)
from probing_environments.premade_utils.sb3 import (
    get_gamma,
    get_value,
    init_agent,
    train_agent,
)

AgentType = Any
AGENT = A2C


def test_successfull():
    """
    Run all tests on sb3 (that we assume to work) and make sure they pass and don't\
          return any bugs.
    """

    check_loss_or_optimizer_value_net(AGENT, init_agent, train_agent, get_value)
    check_backprop_value_net(AGENT, init_agent, train_agent, get_value)
    check_reward_discounting(AGENT, init_agent, train_agent, get_value, get_gamma)


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