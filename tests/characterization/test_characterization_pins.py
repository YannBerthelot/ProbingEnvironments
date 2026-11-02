"""Characterization pins for every environment (TESTING.md §7).

Pins current behaviour, not verified against a spec. May encode bugs.

These tests protect the 0.7 refactor (REFACTOR.md R1a): a pure restructuring
must leave every pin unchanged, and every deliberate behaviour change must
show up here, be listed with ``pin_recorder.py --diff``, get an upgrade note
in the CHANGELOG, and be rewritten with ``pin_recorder.py``. Suspicious
behaviour they pin today is listed in REFACTOR.md §2.3, for example the
3-step first episode of the Gymnasium ``RewardDiscountingEnv``, the Gymnasium
envs moving numpy's global RNG, and the discrete ``PolicyAndValueEnv`` paying
0 for the right action given with shape (1,).
"""

import json

import pytest
from pin_recorder import (
    DRAW_PINS,
    ENV_IDS,
    PINS,
    check_jax_config,
    differences,
    discover_env_ids,
    record,
    record_env_lists,
    record_hierarchy,
)

pytestmark = pytest.mark.characterization

check_jax_config()
PINNED = json.loads(PINS.read_text(encoding="utf-8"))
REGENERATE = (
    "If the change is deliberate, list it with `python"
    " tests/characterization/pin_recorder.py --diff`, add an upgrade note to"
    " the CHANGELOG, then rerun the script without --diff."
)


def _report(what, pinned, recorded):
    found = differences(pinned, recorded)
    lines = [
        f"{where}: pinned {old!r:.200}, now {new!r:.200}" for where, old, new in found
    ]
    return f"{what} changed in {len(found)} place(s):\n" + "\n".join(lines[:5])


def test_characterization_every_shipped_env_is_pinned():
    assert discover_env_ids() == set(ENV_IDS) == set(PINNED["envs"])


def test_characterization_env_lists_are_unchanged():
    recorded = record_env_lists()
    assert recorded == PINNED["env_lists"], _report(
        "env_list", PINNED["env_lists"], recorded
    )


def test_characterization_class_hierarchy_is_unchanged():
    recorded = record_hierarchy()
    assert recorded == PINNED["hierarchy"], _report(
        "The class hierarchy (Ajax dispatches on issubclass)",
        PINNED["hierarchy"],
        recorded,
    )


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_characterization_env_behaviour_is_unchanged(env_id):
    recorded = record(env_id)
    pinned = PINNED["envs"][env_id]
    changed = {
        k for k in pinned.keys() | recorded.keys() if pinned.get(k) != recorded.get(k)
    }
    hint = ""
    if changed and changed <= DRAW_PINS:
        hint = (
            "\nOnly pins that follow JAX's PRNG stream changed: if JAX was"
            f" upgraded (pins recorded with {PINNED['recorded_with']}), its"
            " random stream may have changed rather than the env."
        )
    assert not changed, _report(env_id, pinned, recorded) + hint + "\n" + REGENERATE
