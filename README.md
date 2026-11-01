# ProbingEnvironments
ProbingEnvironments is a library that provides Reinforcement Learning Environments allowing for easy debugging of DeepRL actor-critic algorithms. Tired of debugging your agent by running it on CartPole or another Gym Env and not being sure if it works or you have bugs that cancel one another? This library aims at providing testing envs to make sure that each individual part of your actor-critic algorithm works on simple cases, this allows you to narrow down your bug chase.

The goal of this library is either :
- To use the environments yourself to check your agent by hand
- To include the premade tests in your units tests, allowing to check your agent without relying on long training tests on more complex environments

Functionnalities :
- Simple environments, in both the gymnasium and the gymnax (1.0) APIs, allowing to identify the part of your actor-critic algorithm that seems to be faulty.
- Premade tests/checks that wraps the enviroments and your agent to easily use those environments by hand or in your unit tests.
- Premade adaptors to connect your agent to the tests (to adapt to the way you coded your agent without requiring refactoring) and a template to create yours.


# Installation
```bash
pip install probing-environments
```
Version 0.6.0 and later need gymnax 1.0 or newer. The premade Stable-Baselines3 adaptor also needs `pip install stable-baselines3`.

# How-to
Every snippet below runs as written; the test suite executes them (`tests/test_readme.py`).

## Use an environment directly
The gymnax environments follow the gymnax 1.0 API: `step` returns six values, with time limits reported as `truncated`, separately from `terminated`.
```python
import jax

from probing_environments.gymnax_envs import RewardDiscountingEnv

env = RewardDiscountingEnv()
params = env.default_params
obs, state = env.reset(jax.random.PRNGKey(0), params)
obs, state, reward, terminated, truncated, info = env.step(
    jax.random.PRNGKey(1), state, 0, params
)
```

## Check an agent
Each check trains your agent briefly on one environment and asserts that it reaches the known answer. A failure names the part of the agent most likely at fault.

Connect your agent with an adaptor: use a premade one (here Stable-Baselines3), or write your own from `probing_environments/adaptors/template.py`.
```python
from stable_baselines3 import PPO

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

AGENT = PPO
BUDGET = 2_000  # environment steps per check

check_loss_or_optimizer_value_net(
    AGENT, init_agent, train_agent, get_value, budget=BUDGET
)
check_backprop_value_net(AGENT, init_agent, train_agent, get_value, budget=BUDGET)
check_reward_discounting(
    AGENT, init_agent, train_agent, get_value, get_gamma, budget=BUDGET
)
check_advantage_policy(AGENT, init_agent, train_agent, get_policy, budget=BUDGET)
check_actor_and_critic_coupling(
    AGENT, init_agent, train_agent, get_policy, get_value, budget=BUDGET
)
```
- Put these calls in your unit tests; the (potential) error output should help you pinpoint where to start debugging.
- Keep them in your tests for non-regression testing.

# Disclaimer
The idea for this library comes from this presentation from Andy L Jones : https://andyljones.com/posts/rl-debugging.html


# To-do

- [ ] Expand Readme with example of debugging
- [ ] Expand Readme with example of connector definition
- [ ] Fix the single action policy bug (sb3 policy returns an int instead of a list of float when probability is 100%)
- [ ] Further expand tests
- [ ] Release on Test-PyPi
- [ ] Version automation
- [ ] Rework message codes so they are not cutoff on screen
- [ ] Rework the setup part of readme with extras for reproducibility
