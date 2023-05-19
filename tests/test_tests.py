"Unit tests for the tests ... :)"
from typing import Any

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


def test_tests():
    """
    Run all tests on sb3 (that we assume to work) and make sure they pass and don't\
          return any bugs.
    """
    check_backprop_value_net(init_agent, train_agent, get_value)
    check_loss_or_optimizer_value_net(init_agent, train_agent, get_value)
    check_reward_discounting(init_agent, train_agent, get_value, get_gamma)
