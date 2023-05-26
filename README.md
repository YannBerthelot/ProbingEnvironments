# ProbingEnvironments
ProbingEnvironments is a library that provides Reinforcement Learning Environments allowing for easy debugging of DeepRL actor-critic algorithms. Tired of debugging your agent by running it on CartPole or another Gym Env and not being sure if it works or you have bugs that cancel one another? This library aims at providing testing envs to make sure that each individual part of your actor-critic algorithm works on simple cases, this allows you to narrow down your bug chase.

The goal of this library is either :
- To use the environments yourself to check your agent by hand
- To include the premade tests in your units tests, allowing to check your agent without relying on long training tests on more complex environments

Functionnalities :
- Simple environments (in the gym framework) allowing to identify the part of your actor-critic algorithm that seems to be faulty.
- Premade tests/checks that wraps the enviroments and your agent to easily use those environments by hand or in your unit tests.
- Premade adaptors to connect your agent to the tests (to adapt to the way you coded your agent without requiring refactoring) and a template to create yours.


# Installation 
```bash
pip install git+https://github.com/YannBerthelot/ProbingEnvironments
# if you need extras don't forget to install them in your virtualenv, e.g.
pip install stable-baselines3
```
OR
```bash
poetry add git+https://github.com/YannBerthelot/ProbingEnvironments
# OR, if you need extras (i.e. you are going to use your own adaptors) add @<version>[<extra_name>] e.g. for rlberry
poetry add "git+https://github.com/YannBerthelot/ProbingEnvironments@0.1.0[rlberry]"
```

Installation from PyPi is WIP.


# Extras list
- rlberry : rlberry
- sb3 : stable-baselines3

# How-to
- Install this library (with the required exgtras if the adaptators for your Agent are already provided)
- Create a unit test file in your project.
- Import pytest and the checks from ProbingEnvironments :
```python
import pytest
from probing_environments.checks import (
    check_advantage_policy,
    check_backprop_value_net,
    check_batching_process,
    check_loss_or_optimizer_value_net,
    check_reward_discounting,
)
```
- Import the adaptors for your library OR write them yourself (see template in adaptors/template.py):
```python
from probing_environments.adaptors.sb3 import (
    get_gamma,
    get_policy,
    get_value,
    init_agent,
    train_agent,
)
```
- Import your agent to be fed into the tests.
- You can then use the following tests in your unit tests (adapt the discrete parameter depending on if your agent handles Discrete or Box gym environments):
```python
def test_check_loss_or_optimizer_value_net():
    """
    Test that check_loss_or_optimizer_value_net works on failproof sb3.
    """
    check_loss_or_optimizer_value_net(
        AGENT, init_agent, train_agent, get_value, discrete=False
    )


def test_check_backprop_value_net():
    """
    Test that check_backprop_value_net works on failproof sb3.
    """
    check_backprop_value_net(AGENT, init_agent, train_agent, get_value, discrete=False)


def test_check_reward_discounting():
    """
    Test that check_reward_discounting works on failproof sb3.
    """
    check_reward_discounting(
        AGENT, init_agent, train_agent, get_value, get_gamma, discrete=False
    )


def test_check_advantage_policy():
    """
    Test that check_advantage_policy works on failproof sb3.
    """
    check_advantage_policy(AGENT, init_agent, train_agent, get_policy, discrete=False)


def test_check_actor_and_critic_coupling():
    """
    Test that test_check_actor_and_critic_coupling works on failproof sb3.
    """
    check_actor_and_critic_coupling(
        AGENT, init_agent, train_agent, get_policy, get_value, discrete=False
    )
```
- Run your tests and the (potential) error output should help you pinpoint where to start debugging !
- Keep them in your tests for non-regression testing

# Disclaimer
The idea for this library comes from this presentation from Andy L Jones : https://andyljones.com/posts/rl-debugging.html


# To-do

- [ ] Expand Readme with example of debugging
- [ ] Expand Readme with example of connector definition
- [ ] Fix the single action policy bug (sb3 policy returns an int instead of a list of float when probability is 100%)
- [ ] Further expand tests
- [ ] Fix the no-direct dependency issue when building for PyPi
- [ ] Release on Test-PyPi
- [ ] Init changelog and version automation
- [ ] Rework message codes so they are not cutoff on screen
- [ ] Rework the setup part of readme with extras for reproducibility
