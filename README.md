<h1 align="center">ProbingEnvironments</h1>

<p align="center">
  <b>Find the bug in your RL agent without a long training run.</b><br/>
  Tiny environments with known answers: train your agent on one for a few thousand steps, and a wrong answer points at the part that is broken.
</p>

<p align="center">
  <a href="https://yannberthelot.github.io/ProbingEnvironments/"><img alt="Documentation" src="https://img.shields.io/badge/docs-yannberthelot.github.io%2FProbingEnvironments-blue"></a>
  <a href="https://github.com/YannBerthelot/ProbingEnvironments/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/YannBerthelot/ProbingEnvironments/ci.yml?branch=main&label=CI"></a>
  <a href="https://codecov.io/gh/YannBerthelot/ProbingEnvironments"><img alt="Coverage" src="https://img.shields.io/codecov/c/github/YannBerthelot/ProbingEnvironments"></a>
  <a href="https://pypi.org/project/probing-environments/"><img alt="PyPI" src="https://img.shields.io/pypi/v/probing-environments"></a>
  <a href="https://pypi.org/project/probing-environments/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/probing-environments"></a>
  <a href="https://github.com/YannBerthelot/ProbingEnvironments/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-green"></a>
</p>

<p align="center">
  <a href="#install">Install</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#the-probes">The probes</a> ·
  <a href="#reading-a-failure">Reading a failure</a> ·
  <a href="#connecting-your-agent">Connecting your agent</a> ·
  <a href="https://yannberthelot.github.io/ProbingEnvironments/">Documentation</a>
</p>

A deep-RL agent that learns CartPole slowly, or not at all, can have a dozen
causes, and some bugs hide others. ProbingEnvironments takes them apart.
Each environment is small enough that the right value or action is known in
closed form, and each one tests a single part of an actor-critic agent: the
value loss, backpropagation, discounting, the policy update, the coupling
between actor and critic, memory. Run them in order, and the first one that
fails tells you where to look.

- **Known answers**: the probes use one- and two-step episodes whose correct
  values and actions follow from the rules of the environment, with nothing
  to tune and no baseline to run.
- **Cheap**: most checks train for a few thousand environment steps, seconds
  on a CPU, so they fit in a unit test suite and run on every change.
- **Two APIs**: [gymnax](https://github.com/RobertTLange/gymnax) 1.0
  environments for JAX agents (`jit` and `vmap` them with the agent), and
  [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) ones for the
  rest.
- **Premade checks**: each trains your agent through a small adaptor and
  fails with the expected and the measured value. An adaptor for
  [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) is
  included, and a template shows how to write yours.

[Ajax](https://github.com/YannBerthelot/Ajax)'s probe suite is built on these
environments.

## Install

```bash
pip install probing-environments
# with the premade Stable-Baselines3 adaptor
pip install "probing-environments[sb3]"
```

Version 0.6.0 and later need gymnax 1.0 or newer, and Python 3.11 to 3.14.
To install `main`, whose documentation is the site's `dev` version:

```bash
pip install "git+https://github.com/YannBerthelot/ProbingEnvironments"
```

## Quickstart

Every snippet in this README runs as written: the test suite executes them
(`tests/test_readme.py`).

### Check an agent

Each check trains the agent briefly on one environment and asserts that it
reaches the known answer.

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

Put these calls in your unit tests and keep them there: they catch
regressions as cheaply as they catch the first bug.

### Use an environment directly

The gymnax environments follow the gymnax 1.0 API: `step` returns six
values, with a time limit reported as `truncated`, apart from `terminated`.

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

The discrete environments are in `probing_environments.gymnax_envs`, the
continuous-action ones in `probing_environments.gymnax_envs.continuous_actions`
and the Gymnasium ones in `probing_environments.envs`.

## The probes

Run them in this order: each assumes the ones before it pass. The checks
train with a discount of γ = 0.5 (set by the check, or the adaptor's
default). The discrete checks accept a value within 0.1 of the answer, or
within 10% when that is larger.

| | Check | Environment | The agent must learn | If it fails, look at |
|---|---|---|---|---|
| 1 | `check_loss_or_optimizer_value_net` | `ValueLossOrOptimizerEnv`: one observation, one step, reward 1 | V = 1 | the value loss or the optimiser |
| 2 | `check_backprop_value_net` | `ValueBackpropEnv`: observation 0 or 1, reward equal to it, one step | V(0) = 0 and V(1) = 1 | backpropagation through the value network |
| 3 | `check_reward_discounting` | `RewardDiscountingEnv`: two steps, reward 1 at the end | V(first) = γ and V(second) = 1 | discounting and bootstrapping |
| 4 | `check_advantage_policy` | `AdvantagePolicyLossPolicyUpdateEnv`: two actions, one step, action 0 pays more | P(action 0) > 0.9 | the advantage, the policy loss or the policy update |
| 5 | `check_actor_and_critic_coupling` | `PolicyAndValueEnv`: a random observation, and the action that matches it pays | the matching action with P > 0.9, and V = 1 for both observations | the batching that feeds the networks, for example stale data in the value net |
| 6 | `check_recurrent_agent` | `RecurrentValueEnv` (gymnax): the first observation, 0 or 1, is the reward two steps later | V at the last step equals the first observation | the memory, or how the hidden state is carried |

Continuous-action agents run `check_advantage_policy_continuous` (the
action must reach 0.9) and `check_actor_and_critic_coupling_continuous` (the
action takes the sign of the observation, and V ≥ 0.8) on the continuous
versions of probes 4 and 5. `check_boundary_action_saturation` asks more of
probe 4: the action must reach 0.98 within its default 50,000 steps. An
on-policy agent that recomputes log-probabilities by inverting its tanh
stalls near 0.95.

`TwoChoiceMDP` (two reward cycles of gains 1/5 and 2/5) is there for
average-reward agents. Its check, `check_average_reward`, is unfinished and
does not run on the gymnax version.

## Reading a failure

A failing check raises `AssertionError` with the expected and the measured
value. Probes 1 to 4 also name the part most likely at fault, for example
*"There is most likely a problem with your reward discounting"*.

- **Start from the first failing probe.** A broken value loss can make later
  probes fail too, so their failures say little until it is fixed.
- **Read the measured value, not only the verdict.** V = 2 on probe 1 means
  the agent bootstraps past the end of the episode; V = 0.5 on both
  observations of probe 2 means it does not see the observation.
- **Give it a larger budget before blaming a bug.** A value that moves
  steadily towards the answer, but too slowly, points at the learning rate
  or the budget rather than at the code.

## Connecting your agent

A check drives your agent through a few functions, so it needs no change to
your code:

| Function | What it does |
|---|---|
| `init_agent(agent, env, run_name, gamma, learning_rate, num_envs, budget)` | builds the agent on the probe environment class it is given |
| `train_agent(agent, budget)` | trains it for `budget` environment steps |
| `get_value(agent, obs)` | the critic's value of an observation |
| `get_policy(agent, obs)` | the action probabilities (discrete actions) |
| `get_action(agent, obs[, key])` | the action, for continuous actions; the gymnax checks pass a PRNG key |
| `get_gamma(agent)` | the agent's discount factor |
| `init_hidden_state()`, `compute_next_critic_hidden(agent, obs, done, hidden)`, `get_value_recurrent(agent, obs, done, hidden)` | the memory, for probe 6 only |

`probing_environments.adaptors.sb3` implements them for Stable-Baselines3.
[`adaptors/template.py`](https://github.com/YannBerthelot/ProbingEnvironments/blob/main/src/probing_environments/adaptors/template.py)
documents `init_agent`, `train_agent`, `get_value`, `get_policy` and
`get_gamma` for writing your own; the continuous checks also need
`get_action`, and a `budget` argument on `init_agent`. With an adaptor for a JAX agent,
pass `gymnax=True` to a check to train on the gymnax version of its
environment.

## Proving that a probe catches its bug

A probe is only worth running if it can fail. `prove` trains your agent twice:
once healthy, and once with the bug the probe is named after, planted through
an ordinary setting so your library needs no edit. A probe is `proven` only if
the healthy run passes and the rule catches the faulted one.

```python
from stable_baselines3 import PPO

from probing_environments.adaptors.sb3 import SB3
from probing_environments.faults import no_discount
from probing_environments.probes import discounting
from probing_environments.runner import prove, provisional

agent = SB3(PPO, n_steps=32, learning_rate=1e-3)
proof = prove(discounting(gamma=0.8), agent, no_discount, provisional(at=2000))
print(proof.proven, proof.healthy.status, proof.faulted.status)
print(proof.faulted.report)
```

Every seed has to pass; there is no vote. A run ends in one of five states:
`passed`, `flagged`, `not_tested` (a counter did not move), `crashed` or
`diverged`. A run that crashed or trained nothing proves nothing.

`discounting` is the only probe here so far. This API is provisional: the rule
will change, and its name says so. To connect another library, give an agent
one method, `reader(probe, **settings)`, returning `read(seeds, at)`: the query
values and an `"updates"` counter, one number per seed.

## Documentation

The [documentation site](https://yannberthelot.github.io/ProbingEnvironments/)
has this guide and the
[changelog](https://yannberthelot.github.io/ProbingEnvironments/latest/changelog/),
one version per release; `dev` follows `main`.

## Roadmap

Planned for 0.7.0: environments that observe something informative and
declare the action box they pay on, more probes (time limits, credit through
a delayed reward, multi-dimensional actions, memory in the actor, stochastic
rewards, average reward), and a report that names the likely cause from the
probes' results.

## Related projects

- **[Debugging RL, Without the Agonizing Pain](https://andyljones.com/posts/rl-debugging.html)**,
  Andy L. Jones: the idea and the first five probes come from this post.
- **[bsuite](https://github.com/google-deepmind/bsuite)**: small experiments
  that measure an agent's capabilities, where these probes look for its bugs.
- **[gymnax](https://github.com/RobertTLange/gymnax)** and
  **[Gymnasium](https://github.com/Farama-Foundation/Gymnasium)**: the two
  environment APIs implemented here.
- **[Ajax](https://github.com/YannBerthelot/Ajax)**: JAX agents whose probe
  suite is built on these environments.
- **[TargetGym](https://github.com/YannBerthelot/TargetGym)**: setpoint-tracking
  environments with PID and MPC baselines, for when the probes pass.

## Contributing, citing, license

Bug reports and new probes are welcome.

```bash
git clone https://github.com/YannBerthelot/ProbingEnvironments.git
cd ProbingEnvironments
uv sync
make ci   # pre-commit, the small and medium tests with coverage, the docs
```

`make test` runs only the fast tier.

```bibtex
@misc{probingenvironments2023,
  title        = {ProbingEnvironments: Environments with Known Answers for Debugging Deep RL Agents},
  author       = {Yann Berthelot},
  year         = {2023},
  url          = {https://github.com/YannBerthelot/ProbingEnvironments},
}
```

ProbingEnvironments is MIT-licensed
([LICENSE](https://github.com/YannBerthelot/ProbingEnvironments/blob/main/LICENSE)).

## Disclaimer

The idea for this library comes from this presentation from Andy L Jones:
https://andyljones.com/posts/rl-debugging.html
