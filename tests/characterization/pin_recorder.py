"""Record how every environment behaves today, for the characterization pins.

Pins current behaviour, not verified against a spec. May encode bugs
(TESTING.md §7): the 0.7 refactor fixes several, and each fix shows up as a
change to ``pins.json``.

Everything is observed through the public surface that users and Ajax use:
``reset``, ``step``, the hooks Ajax's subclasses override or call
(``get_obs``, ``step_env``, ``is_terminated``, ``discount``), the spaces,
``default_params``, the state fields, ``env_list`` and the class hierarchy.

After a deliberate behaviour change:
- ``python tests/characterization/pin_recorder.py --diff`` lists every pin
  that changed, as material for the CHANGELOG's upgrade notes;
- ``python tests/characterization/pin_recorder.py`` rewrites ``pins.json``.
"""

import dataclasses
import importlib
import inspect
import json
import os
import sys
from pathlib import Path

# The pins assume CPU and 32-bit JAX; set before JAX is imported.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# pylint: disable=wrong-import-position
import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import environment

PINS = Path(__file__).with_name("pins.json")
N_KEYS = 32  # reset draws pinned per env (PLAN.md §3)
MAX_STEPS = 12  # enough to see TwoChoiceMDP's cycles of 5 steps
# Attributes users or checks read off an env instance (checks.py reads
# TIME_LIMIT); RecurrentValueEnv stores original_state on itself.
ATTRIBUTES = (
    "obs_shape",
    "action_shape",
    "original_state",
    "TIME_LIMIT",
    "obs_stack",
    "time_stack",
)

GYMNAX_MODULES = {
    "probing_environments.gymnax_envs": [
        "ValueLossOrOptimizerEnv",
        "ValueBackpropEnv",
        "RewardDiscountingEnv",
        "AdvantagePolicyLossPolicyUpdateEnv",
        "PolicyAndValueEnv",
        "RecurrentValueEnv",
    ],
    "probing_environments.gymnax_envs.continuous_actions": [
        "ValueLossOrOptimizerEnv",
        "ValueBackpropEnv",
        "RewardDiscountingEnv",
        "AdvantagePolicyLossPolicyUpdateEnv",
        "PolicyAndValueEnv",
    ],
    "probing_environments.gymnax_envs.average_reward.two_choice_MDP": ["TwoChoiceMDP"],
}
GYMNASIUM_MODULE = "probing_environments.envs"
GYMNASIUM_CLASSES = [
    "ValueLossOrOptimizerEnv",
    "ValueBackpropEnv",
    "RewardDiscountingEnv",
    "AdvantagePolicyLossPolicyUpdateEnv",
    "AdvantagePolicyLossPolicyUpdateEnvContinuous",
    "PolicyAndValueEnv",
    "PolicyAndValueEnvContinuous",
    "TwoChoiceMDP",
]
# Modules whose ``env_list`` Ajax iterates over.
ENV_LIST_MODULES = [
    "probing_environments.gymnax_envs",
    "probing_environments.gymnax_envs.continuous_actions",
]
ENV_IDS = [f"gymnax:{m}.{c}" for m, cs in GYMNAX_MODULES.items() for c in cs] + [
    f"gymnasium:{GYMNASIUM_MODULE}.{c}" for c in GYMNASIUM_CLASSES
]
# Pins that follow JAX's PRNG stream rather than an env's own code.
DRAW_PINS = {"reset_obs", "autoreset_obs", "episodes", "obs_through_get_obs_override"}


def check_jax_config():
    """Refuse to record or compare under a JAX setup the pins don't assume."""
    if jax.config.jax_enable_x64:
        raise RuntimeError(
            "The pins were recorded with JAX's x64 mode off: unset JAX_ENABLE_X64."
        )


def _num(x):
    """A JSON-stable copy of a number or array: floats rounded to 6 places."""
    a = np.asarray(x)
    if a.dtype == bool or np.issubdtype(a.dtype, np.integer):
        return a.tolist()
    return np.round(a.astype(float), 6).tolist()


def _kind(value):
    """What a value is, beyond its number: JSON can't tell 0 from 0.0 or a
    Python bool from an array."""
    if isinstance(value, (bool, int, float, str, type(None))):
        return type(value).__name__
    if isinstance(value, jax.Array):
        return f"jax.Array[{value.dtype.name}{list(value.shape)}]"
    if isinstance(value, np.ndarray):
        return f"ndarray[{value.dtype.name}{list(value.shape)}]"
    if isinstance(value, np.generic):
        return f"numpy.{value.dtype.name}"
    return type(value).__name__


def _dtypes(values):
    """The dtype names of some values."""
    return sorted({np.asarray(v).dtype.name for v in values})


def _flat(arrays):
    """Values of same-shaped observations as one flat list, plus that shape."""
    shapes = {np.shape(a) for a in arrays}
    assert len(shapes) == 1, shapes
    return [v for a in arrays for v in np.ravel(_num(a)).tolist()], list(shapes.pop())


def _signature(fn):
    """Parameter names, kinds and defaults, without annotations."""
    params = inspect.signature(fn).parameters.values()
    return [str(p.replace(annotation=inspect.Parameter.empty)) for p in params]


def _space(space):
    """The attributes that define a gymnax or Gymnasium space."""
    out = {"type": type(space).__name__}
    if isinstance(getattr(space, "spaces", None), dict):
        out["spaces"] = {k: _space(v) for k, v in space.spaces.items()}
        return out
    for attr in ("n", "start", "low", "high", "shape", "dtype"):
        if hasattr(space, attr):
            value = getattr(space, attr)
            if attr == "dtype":
                out[attr] = np.dtype(value).name
            elif attr == "shape":
                out[attr] = list(value)
            else:
                out[attr] = _num(value)
    return out


def _fields(dataclass_type):
    """Field names of a (flax) dataclass, with their defaults."""
    out = {}
    for f in dataclasses.fields(dataclass_type):
        if f.default is not dataclasses.MISSING:
            out[f.name] = _num(f.default)
        elif f.default_factory is not dataclasses.MISSING:
            out[f.name] = _num(f.default_factory())
        else:
            out[f.name] = "<required>"
    return out


def _leaves(state):
    """The state's field values, in field order."""
    return [_num(getattr(state, f.name)) for f in dataclasses.fields(state)]


def _attributes(env):
    """The ATTRIBUTES an env exposes, whether set on the instance or the class:
    tuples and builtins by value, anything else (a deque, a tracer leaked from
    a jitted reset) as "object"."""
    out = {}
    for name in ATTRIBUTES:
        value = getattr(env, name, "<missing>")
        if isinstance(value, tuple):
            out[name] = list(value)
        elif isinstance(value, (bool, int, float, str, type(None))):
            out[name] = value
        else:
            out[name] = "object"
    return out


def _gymnax_actions(space):
    """(label, action) pairs, as scalars and as shape-(1,) arrays."""
    if type(space).__name__ == "Discrete":
        n = int(space.n)
        return [(str(a), jnp.asarray(a, jnp.int32)) for a in range(n)] + [
            (f"[{a}]", jnp.asarray([a], jnp.int32)) for a in range(n)
        ]
    return [
        (f"[{v}]", jnp.asarray([v], jnp.float32)) for v in (-1.0, 0.0, 0.5, 1.0, 2.0)
    ] + [(str(v), jnp.asarray(v, jnp.float32)) for v in (-1.0, 1.0)]


def _gymnax_episode(env, key, action, params):
    """One episode under a fixed action, plus one step past its end to pin the
    auto-reset. Per step: the true next observation
    (``info["final_observation"]``), the returned observation, reward,
    terminated, truncated, discount and the returned state's fields."""
    _, state = env.reset(key, params)
    steps, ended = [], False
    for t in range(MAX_STEPS):
        obs, state, reward, terminated, truncated, info = env.step(
            jax.random.PRNGKey(1000 + t), state, action, params
        )
        steps.append(
            [
                _num(info["final_observation"]),
                _num(obs),
                _num(reward),
                bool(terminated),
                bool(truncated),
                _num(info["discount"]) if "discount" in info else None,
                _leaves(state),
            ]
        )
        if ended:
            break
        ended = bool(terminated or truncated)
    return steps


def _hooked(cls):
    """An instance of Ajax's ObserveOne pattern: a subclass overriding
    ``get_obs``. Observations must reach it through ``reset`` and ``step``."""

    class Hooked(cls):  # pylint: disable=too-few-public-methods
        """Observes 10 more than its parent."""

        def get_obs(self, state, params=None, key=None):
            return super().get_obs(state, params, key) + 10

    return Hooked()


def record_gymnax(module, name):
    """Everything a user or Ajax can observe of one gymnax environment."""
    cls = getattr(importlib.import_module(module), name)
    env = cls()
    params = env.default_params
    keys = [jax.random.PRNGKey(i) for i in range(N_KEYS)]
    resets = [env.reset(k, params) for k in keys]
    state = resets[0][1]
    action_space = env.action_space(params)
    actions = _gymnax_actions(action_space)
    first_action = actions[0][1]

    reset_obs, obs_shape = _flat([o for o, _ in resets])
    step_out = env.step(keys[0], state, first_action, params)
    autoreset_obs, _ = _flat(
        [env.step(k, state, first_action, params)[0] for k in keys]
    )
    vmapped_reset = jax.vmap(env.reset, in_axes=(0, None))(jnp.stack(keys), params)[0]
    vmapped_step = jax.vmap(env.step, in_axes=(0, None, None, None))(
        jnp.stack(keys), state, first_action, params
    )[0]

    # The hooks Ajax's subclasses call themselves, unjitted (map_ajaxUse.md §1).
    direct = env.step_env(keys[0], state, first_action, params)
    hooked = _hooked(cls)
    hooked_obs, hooked_state = hooked.reset(keys[0], params)
    hooked_step = hooked.step(keys[0], hooked_state, first_action, params)

    param_sets = {
        "default": params,
        # Ajax's override (map_ajaxUse.md §1).
        "max_steps=10000": params.replace(max_steps_in_episode=10_000),
    }
    if name == "TwoChoiceMDP":
        param_sets["max_steps=3"] = params.replace(max_steps_in_episode=3)
    starts = {}  # the first key that resets to each distinct observation
    for i, (obs, _) in enumerate(resets):
        starts.setdefault(json.dumps(_num(obs)), i)
    episodes = {}
    for p_name, p in param_sets.items():
        for i in starts.values():
            for label, action in actions:
                if p_name != "default" and action is not first_action:
                    continue
                episodes[f"params={p_name} start=key{i} action={label}"] = (
                    _gymnax_episode(env, keys[i], action, p)
                )

    record = {
        "name": env.name,
        "num_actions": int(env.num_actions),
        "default_params": {
            f.name: _num(getattr(params, f.name)) for f in dataclasses.fields(params)
        },
        "state_fields": _fields(type(state)),
        "action_space": _space(action_space),
        "observation_space": _space(env.observation_space(params)),
        "state_space": _space(env.state_space(params)),
        "get_obs_signature": _signature(cls.get_obs),
        "step_returns": len(step_out),
        "step_info_keys": sorted(step_out[5]),
        "dtypes": {
            "reset_obs": _dtypes([o for o, _ in resets]),
            "step_obs": _dtypes([step_out[5]["final_observation"]]),
            "reward": _dtypes([step_out[2]]),
            "terminated": _dtypes([step_out[3]]),
            "truncated": _dtypes([step_out[4]]),
            "discount": _dtypes([step_out[5]["discount"]]),
            "state": {
                f.name: _dtypes([getattr(state, f.name)])
                for f in dataclasses.fields(state)
            },
        },
        "attributes_after_reset": _attributes(env),
        "step_env_direct": {
            "returns": len(direct),
            "obs": [_num(direct[0]), _kind(direct[0])],
            "reward": [_num(direct[2]), _kind(direct[2])],
            "terminated": [_num(direct[3]), _kind(direct[3])],
            "info_keys": sorted(direct[4]),
        },
        "is_terminated_direct": [
            _num(v)
            for v in (
                env.is_terminated(state, params),
                env.is_terminated(direct[1], params),
            )
        ],
        "discount_direct": [
            _num(v)
            for v in (env.discount(state, params), env.discount(direct[1], params))
        ],
        # Ajax rebuilds states from x and time alone: type(state)(x=, time=).
        "obs_of_state_rebuilt_from_x_and_time": _num(
            env.get_obs(type(state)(x=state.x, time=state.time), params, keys[0])
        ),
        "obs_through_get_obs_override": {
            "reset": _num(hooked_obs),
            "step": _num(hooked_step[0]),
            "step_final": _num(hooked_step[5]["final_observation"]),
        },
        "reset_obs_shape": obs_shape,
        "reset_obs": reset_obs,
        "autoreset_obs": autoreset_obs,
        "vmapped_reset_matches": bool(
            np.array_equal(_flat(list(vmapped_reset))[0], reset_obs)
        ),
        "vmapped_step_matches": bool(
            np.array_equal(_flat(list(vmapped_step))[0], autoreset_obs)
        ),
        "episodes": episodes,
    }
    if type(action_space).__name__ == "Box":
        # APG-style agents differentiate the reward through the action.
        record["reward_gradient"] = {
            str(v): _num(
                jax.grad(
                    lambda a: jnp.sum(
                        jnp.asarray(env.step_env(keys[0], state, a, params)[2], float)
                    )
                )(jnp.asarray([v], jnp.float32))
            )
            for v in (-1.0, 0.0, 0.5, 1.0, 2.0)
        }
    return record


def _touches_global_rng(make, **reset_kwargs):
    """Whether ``reset`` moves numpy's global RNG, compared with a baseline
    draw from the same global seed."""
    env = make()
    np.random.seed(123)
    baseline = np.random.random()
    np.random.seed(123)
    env.reset(**reset_kwargs)
    return bool(np.random.random() != baseline)


def _gymnasium_actions(space):
    """(label, action) pairs, including the forms users pass by mistake."""
    if type(space).__name__ == "Discrete":
        n = int(space.n)
        return (
            [(str(a), a) for a in range(n)]
            + [(f"np.int64({a})", np.int64(a)) for a in range(n)]
            + [(f"[{a}]", np.array([a])) for a in range(n)]
        )
    return [
        (f"[{v}]", np.array([v], dtype=np.float32)) for v in (-1.0, 0.0, 0.5, 1.0, 2.0)
    ] + [(str(v), v) for v in (0.5, 1.0)]


def _gymnasium_step(env, action, kinds):
    """One step's pin, noting the output kinds in ``kinds``."""
    try:
        obs, reward, terminated, truncated, info = env.step(action)
    except Exception as error:  # pylint: disable=broad-exception-caught
        return ["raises", type(error).__name__], True
    kinds.add(
        json.dumps(
            [
                _kind(obs),
                _kind(reward),
                _kind(terminated),
                _kind(truncated),
                sorted(info),
            ]
        )
    )
    return [_num(obs), _num(reward), _num(terminated), _num(truncated)], bool(
        terminated or truncated
    )


def record_gymnasium(name):
    """Everything a user can observe of one Gymnasium environment."""
    cls = getattr(importlib.import_module(GYMNASIUM_MODULE), name)
    env = cls()
    resets = [env.reset(seed=s) for s in range(8)]
    actions = _gymnasium_actions(env.action_space)
    try:
        env.reset(seed=0, options={})
        reset_with_options = "ok"
    except Exception as error:  # pylint: disable=broad-exception-caught
        reset_with_options = type(error).__name__
    instances = {"default": cls}
    if name == "TwoChoiceMDP":
        instances["time_limit=3"] = lambda: cls(time_limit=3)

    episodes, kinds = {}, set()
    for i_name, make in instances.items():
        for label, action in actions:
            fresh = make()  # some envs carry state across episodes
            for episode in range(3):
                obs, _ = fresh.reset(seed=episode)
                steps = [[_num(obs)]]
                for _ in range(MAX_STEPS):
                    step, done = _gymnasium_step(fresh, action, kinds)
                    steps.append(step)
                    if done:
                        break
                episodes[f"{i_name} action={label} episode={episode}"] = steps
    # Two resets, then two steps: what the reward queue does with them.
    queued = cls()
    queued.reset(seed=0)
    queued.reset(seed=1)
    reset_reset_step_step = [
        _gymnasium_step(queued, actions[0][1], kinds)[0] for _ in range(2)
    ]

    return {
        "signatures": {
            "__init__": _signature(cls.__init__),
            "reset": _signature(cls.reset),
            "step": _signature(cls.step),
        },
        "metadata": json.loads(json.dumps(env.metadata)),
        "action_space": _space(env.action_space),
        "observation_space": _space(env.observation_space),
        "reset_obs": [_num(o) for o, _ in resets],
        "reset_kinds": sorted({json.dumps([_kind(o), sorted(i)]) for o, i in resets}),
        "reset_with_options": reset_with_options,
        "reset_moves_global_rng": {
            "seed=None": _touches_global_rng(cls),
            "seed=0": _touches_global_rng(cls, seed=0),
        },
        "attributes_after_reset": _attributes(env),
        "step_kinds": sorted(kinds),
        "episodes": episodes,
        "reset_reset_step_step": reset_reset_step_step,
    }


def record(env_id):
    """The current behaviour of the environment ``env_id`` (see ENV_IDS)."""
    kind, path = env_id.split(":")
    module, name = path.rsplit(".", 1)
    return record_gymnax(module, name) if kind == "gymnax" else record_gymnasium(name)


def _shipped_classes():
    """{env id: class} for every pinned env."""
    out = {}
    for env_id in ENV_IDS:
        module, name = env_id.split(":")[1].rsplit(".", 1)
        out[env_id] = getattr(importlib.import_module(module), name)
    return out


def record_env_lists():
    """Each ``env_list``, entry by entry, as the env id of the class it holds.
    Discrete and continuous classes share names, so names alone can't tell a
    wrong flavour."""
    ids = {cls: env_id for env_id, cls in _shipped_classes().items()}
    return {
        m: [
            ids.get(c, f"<unpinned {c.__module__}.{c.__qualname__}>")
            for c in importlib.import_module(m).env_list
        ]
        for m in ENV_LIST_MODULES
    }


def record_hierarchy():
    """For each env, the other shipped envs it subclasses: Ajax dispatches on
    ``issubclass`` (map_ajaxUse.md §1)."""
    classes = _shipped_classes()
    return {
        a: sorted(b for b, cb in classes.items() if b != a and issubclass(ca, cb))
        for a, ca in classes.items()
    }


def discover_env_ids():
    """The env ids of every environment class the modules expose."""
    found = set()
    for module in GYMNAX_MODULES:
        for name, value in vars(importlib.import_module(module)).items():
            if (
                isinstance(value, type)
                and issubclass(value, environment.Environment)
                and value.__module__.startswith("probing_environments")
                and not name.startswith("_")
            ):
                found.add(f"gymnax:{module}.{name}")
    for name, value in vars(importlib.import_module(GYMNASIUM_MODULE)).items():
        if (
            isinstance(value, type)
            and issubclass(value, gymnasium.Env)
            and value.__module__.startswith("probing_environments")
        ):
            found.add(f"gymnasium:{GYMNASIUM_MODULE}.{name}")
    return found


def record_all():
    """Every pin, in the layout of pins.json."""
    check_jax_config()
    return {
        "note": (
            "Characterization pins (TESTING.md §7): current behaviour, not a"
            " spec. May encode bugs. See tests/characterization/pin_recorder.py."
        ),
        "recorded_with": {
            "jax": jax.__version__,
            "jax_threefry_partitionable": bool(
                getattr(jax.config, "jax_threefry_partitionable", False)
            ),
        },
        "env_lists": record_env_lists(),
        "hierarchy": record_hierarchy(),
        "envs": {env_id: record(env_id) for env_id in ENV_IDS},
    }


def differences(old, new, path=""):
    """Every differing leaf between two pin trees, as (path, old, new)."""
    if isinstance(old, dict) and isinstance(new, dict):
        out = []
        for key in sorted(old.keys() | new.keys()):
            out += differences(old.get(key), new.get(key), f"{path}[{key!r}]")
        return out
    return [] if old == new else [(path, old, new)]


def _dump(obj, indent=0):
    """JSON with one line per short or flat list, so pin diffs stay readable."""
    pad = " " * (indent + 1)
    if isinstance(obj, dict):
        items = [
            f"{pad}{json.dumps(k)}: {_dump(v, indent + 1)}" for k, v in obj.items()
        ]
        return "{\n" + ",\n".join(items) + "\n" + " " * indent + "}"
    nested = isinstance(obj, list) and any(isinstance(e, (list, dict)) for e in obj)
    if nested and len(json.dumps(obj)) > 100:
        items = [pad + _dump(e, indent + 1) for e in obj]
        return "[\n" + ",\n".join(items) + "\n" + " " * indent + "]"
    return json.dumps(obj)


if __name__ == "__main__":
    current = record_all()
    if "--diff" in sys.argv:
        pinned = json.loads(PINS.read_text(encoding="utf-8"))
        del pinned["recorded_with"], current["recorded_with"]
        for where, before, after in differences(pinned, current):
            print(f"{where}\n  pinned: {before!r:.300}\n  now:    {after!r:.300}")
    else:
        PINS.write_text(_dump(current) + "\n", encoding="utf-8")
        print(f"wrote {PINS}")
