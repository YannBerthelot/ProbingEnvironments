"""Prove that a probe catches its bug, on any agent.

Provisional, and not a public promise: ``provisional`` holds only the clauses
every candidate rule shares, and the frozen rule replaces it. A rule is one
callable, so swapping it touches nothing else. ``at`` is in the agent's own
training units until that rule fixes the clock.
"""

from typing import Callable, Mapping, Optional, Protocol, Sequence

import numpy as np

from probing_environments.faults import Fault
from probing_environments.probes import Probe, Query
from probing_environments.results import Proof, Verdict

Readings = Mapping[str, np.ndarray]  # by query name, one value per seed
Read = Callable[[tuple[int, ...], int], Readings]  # (seeds, how far to train)
Rule = Callable[[Sequence[Query], Read], Verdict]
UPDATES = "updates"  # the readings' key for each seed's update counter
SEEDS = tuple(range(8))


class Agent(Protocol):  # pylint: disable=too-few-public-methods
    """What ``prove`` needs from a library."""

    def reader(self, probe: Probe, **settings: float) -> Read:
        """Train at the probe's gamma unless ``settings`` overrides it."""


def _bands(
    queries: Sequence[Query], tolerances: Optional[Mapping[str, float]]
) -> dict[str, float]:
    bands = {q.name: q.band for q in queries}
    for name, tolerance in (tolerances or {}).items():
        if tolerance > bands[name]:  # an unknown name raises too
            raise ValueError(f"{name}: a tolerance cannot exceed the band")
        bands[name] = tolerance
    return bands


def _line(q: Query, readings: np.ndarray, band: float) -> str:
    line = (
        f"  {q.name}: wants {q.truth:.3f} +- {band:.3f}; readings {readings.round(3)}"
    )
    median = float(np.median(readings))
    if abs(median - q.truth) > band and q.wrong and np.isfinite(median):
        name = min(q.wrong, key=lambda k: abs(q.wrong[k] - median))
        line += f"; median {median:.3f} is nearest {name!r} ({q.wrong[name]:.3f})"
    return line


def provisional(
    at: int,
    seeds: Sequence[int] = SEEDS,
    tolerances: Optional[Mapping[str, float]] = None,
) -> Rule:
    """Every seed must end in band, with a counter that moved and a query that
    was out of band before training, so it could have failed. There is no vote.
    A row that could not have failed is ``not_tested``, not a catch."""
    ids = tuple(seeds)
    if not ids:
        raise ValueError("a rule needs at least one seed")

    def row(readings: Readings, name: str) -> np.ndarray:
        return np.asarray(readings[name], float).reshape(len(ids))

    def rule(queries: Sequence[Query], read: Read) -> Verdict:
        bands = _bands(queries, tolerances)
        try:
            before, after = read(ids, 0), read(ids, at)
            start = [row(before, q.name) for q in queries]
            end = [row(after, q.name) for q in queries]
            counted = UPDATES in before and UPDATES in after
            moved = counted and (row(after, UPDATES) > row(before, UPDATES)).all()
        except Exception as error:  # pylint: disable=broad-exception-caught
            return Verdict("crashed", f"{type(error).__name__}: {error}")
        tols = [bands[q.name] for q in queries]
        lines = "\n".join(map(_line, queries, end, tols))
        # First: a run that did not train says nothing about what follows.
        if not moved:
            return Verdict("not_tested", f"no update counter moved\n{lines}")
        if not np.isfinite(end).all():
            return Verdict("diverged", f"a reading is not finite\n{lines}")
        off = [np.abs(r - q.truth) > t for r, q, t in zip(start, queries, tols)]
        on = [np.abs(r - q.truth) <= t for r, q, t in zip(end, queries, tols)]
        within = np.all(on, axis=0)
        report = f"{int(within.sum())} of {len(ids)} seeds within\n{lines}"
        if not within.all():
            return Verdict("flagged", report)
        if not (np.isfinite(start).all() and np.any(off, axis=0).all()):
            return Verdict("not_tested", f"it could not have failed\n{report}")
        return Verdict("passed", report)

    return rule


def prove(probe: Probe, agent: Agent, fault: Fault, rule: Rule) -> Proof:
    """Run the probe healthy and with ``fault``, judged by one rule against the
    probe's own answers."""
    healthy = rule(probe.queries, agent.reader(probe))
    faulted = rule(probe.queries, agent.reader(probe, **fault.settings))
    return Proof(probe.id, fault.id, healthy, faulted)
