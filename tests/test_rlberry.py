"Unit tests for the tests ... :)"
from typing import Any

import pytest

from probing_environments.checks import (
    check_backprop_value_net,
    check_loss_or_optimizer_value_net,
    check_reward_discounting,
)
from probing_environments.premade_utils.rlberry import (
    get_gamma,
    get_value,
    init_agent,
    train_agent,
)

AgentType = Any


def test_successfull():
    """
    Run all tests on sb3 (that we assume to work) and make sure they pass and don't\
          return any bugs.
    """
    check_loss_or_optimizer_value_net(
        init_agent, train_agent, get_value, discrete=False
    )
    check_backprop_value_net(init_agent, train_agent, get_value, discrete=False)
    check_reward_discounting(
        init_agent, train_agent, get_value, get_gamma, discrete=False
    )


def test_errors():
    """
    Run all tests on sb3 (that we assume to work) and make sure they pass and don't\
          return any bugs.
    """
    with pytest.raises(AssertionError):
        check_loss_or_optimizer_value_net(
            init_agent, train_agent, get_value=lambda x, y: -1
        )
    with pytest.raises(AssertionError):
        check_backprop_value_net(init_agent, train_agent, get_value=lambda x, y: -1)
    with pytest.raises(AssertionError):
        check_reward_discounting(
            init_agent, train_agent, get_value=lambda x, y: -1, get_gamma=get_gamma
        )
