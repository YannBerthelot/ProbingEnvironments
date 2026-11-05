"""What a correct agent must read, derived by hand and never from a run.

Provisional: this holds the one probe carried through the whole path so far.
The answers move to ``env.known_answers`` when the ladder is rebuilt.
"""

import dataclasses
import math
from typing import Mapping

CEILING = 0.1  # however far the nearest wrong answer, a band is no wider


@dataclasses.dataclass(frozen=True)
class Query:
    """One value to read at ``obs``: its ``truth`` and named ``wrong`` answers.
    The names say what a bad reading looks like; only the truth judges."""

    name: str
    obs: float
    truth: float
    wrong: Mapping[str, float]

    @property
    def band(self) -> float:
        """Half the gap to the nearest wrong answer, at most ``CEILING``."""
        return min([CEILING, *(0.5 * abs(self.truth - w) for w in self.wrong.values())])


@dataclasses.dataclass(frozen=True)
class Probe:
    """``env`` names the class, the same in ``envs`` and ``gymnax_envs``.
    ``gamma`` is the spec: the answers hold at it, whatever an agent trained at."""

    id: str
    env: str
    gamma: float
    queries: tuple[Query, ...]


def discounting(gamma: float) -> Probe:
    """s0 -> s1 -> end, reward 1 on the last step and 0 before it, so
    V(1) = 1 and V(0) = gamma.

    The wrong answers: discounting applied twice, gamma^2; the complement used
    as the discount, 1 - gamma; no discounting, 1; a done flag stored one step
    early or late, which cuts the bootstrap, 0. If the end is ignored, the
    return runs on into the next episode's s0, so V(1) = 1 + gamma^2 V(1).
    The band narrows as gamma nears 0.5, where 1 - gamma nears the truth.
    """
    if not 0 < gamma < 1:
        raise ValueError(f"gamma must lie strictly between 0 and 1, got {gamma}")
    named = {
        "gamma squared": gamma**2,
        "1 - gamma": 1 - gamma,
        "no discount": 1.0,
        "done flag one step late": 0.0,
    }
    # At gamma 0.5, 1 - gamma is the truth: it names nothing there.
    v0 = {k: v for k, v in named.items() if not math.isclose(v, gamma)}
    v1 = {"done mask ignored": 1 / (1 - gamma**2)}
    queries = Query("V(0)", 0.0, gamma, v0), Query("V(1)", 1.0, 1.0, v1)
    return Probe("discounting", "RewardDiscountingEnv", gamma, queries)
