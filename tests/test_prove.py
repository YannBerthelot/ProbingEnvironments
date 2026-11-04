"""A probe must show that it catches the bug it is named after.

Contract, taken from the owner's direction and decisions of 2026-11-04 and the
candidate probe rules, never from the code. The rule is provisional: it holds
only the clauses all candidates share, and the frozen rule replaces it.

The rule reads every query before training and at ``at``.
- A seed passes if every query is in band at ``at``, at least one query was
  out of band before training (it could have failed), and its update counter
  moved. Being on the band's edge is in band.
- A row passes only if every seed passes. There is no vote.
- Anything else is a named state, the first that applies: ``crashed`` if
  training raised; ``not_tested`` if a counter is missing or did not move;
  ``diverged`` if a reading after training is not finite; ``flagged`` if a seed
  ends out of band; ``not_tested`` if a row that ended in band could not have
  failed, or was unreadable before training.
- A tolerance may be tighter than the query's band, never wider, and must name
  a query. A rule needs a seed.

``prove`` requires a healthy run that passes and a faulted run the rule caught
(``flagged`` or ``diverged``). A crash, a run that did not train and a row that
could not have failed prove nothing. The truth always comes from the probe, so
a fault cannot change what it is judged against.

The band is half the gap to the nearest named wrong answer, at most 0.1.
``discounting`` is V(0) = gamma and V(1) = 1 (one reward of 1 on the second
step, then the end), and the rollouts below check that against the real env.
"""

from typing import Any, Callable, Mapping, Sequence

import jax
import numpy as np
import pytest

from probing_environments import envs, gymnax_envs
from probing_environments.faults import Fault, no_discount
from probing_environments.gymnax_envs import continuous_actions
from probing_environments.probes import Probe, Query, discounting
from probing_environments.results import Verdict
from probing_environments.runner import UPDATES, prove, provisional

# Band of V: half the gap to 0.5 is 0.25, so the ceiling sets it, 0.1.
V = Query("V", 0.0, 1.0, {"off": 0.5})
# Band of W: half the gap to 0.1 is 0.05.
W = Query("W", 1.0, 0.0, {"near": 0.1})
SEEDS = (0, 1, 2)
AT = 100
Row = Sequence[float]


def fake_read(
    before: Mapping[str, Row], after: Mapping[str, Row], updates: Row = (5, 5, 5)
) -> Callable[[tuple[int, ...], int], dict[str, np.ndarray]]:
    """A fake of the agent seam: canned readings, per seed, at each clock."""

    def read(seeds: tuple[int, ...], at: int) -> dict[str, np.ndarray]:
        out = {k: np.asarray(v, float) for k, v in (after if at else before).items()}
        return out | {UPDATES: np.asarray(updates if at else (0,) * len(seeds))}

    return read


def judge(before: Row, after: Row, **kw: Any) -> Verdict:
    return provisional(AT, SEEDS, **kw)((V,), fake_read({"V": before}, {"V": after}))


FRESH = (0.0, 0.0, 0.0)  # untrained: far from V's truth, 1


def test_passes_when_every_seed_lands_in_band() -> None:
    assert judge(FRESH, (1.0, 1.05, 0.95)).status == "passed"


def test_one_seed_out_of_band_flags_the_row_with_no_vote() -> None:
    read = fake_read({"V": (0.0,) * 8}, {"V": (1.0,) * 7 + (0.5,)}, (5,) * 8)
    assert provisional(AT, tuple(range(8)))((V,), read).status == "flagged"


@pytest.mark.parametrize(
    "reading, status",
    [(1.0625, "passed"), (1.0625001, "flagged"), (0.9375, "passed")],
)
def test_the_band_edge_is_in_band(reading: float, status: str) -> None:
    # 1/16 is exact in binary, so the edge is exact too.
    got = judge(FRESH, (reading,) * 3, tolerances={"V": 0.0625})
    assert got.status == status


def test_a_row_already_in_band_before_training_could_not_have_failed() -> None:
    assert judge((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)).status == "not_tested"


def test_one_query_out_of_band_before_training_is_enough() -> None:
    before = {"V": (0.0,) * 3, "W": (0.0,) * 3}  # W starts on its truth
    after = {"V": (1.0,) * 3, "W": (0.0,) * 3}
    assert provisional(AT, SEEDS)((V, W), fake_read(before, after)).status == "passed"


def test_the_could_have_failed_control_is_per_seed() -> None:
    assert judge((0.0, 0.0, 1.0), (1.0, 1.0, 1.0)).status == "not_tested"


def test_a_wrong_reading_outranks_the_could_have_failed_control() -> None:
    assert judge((1.0, 0.0, 0.0), (0.5, 1.0, 1.0)).status == "flagged"


@pytest.mark.parametrize("moved", [(5, 5, 0), (0, 0, 0)])
def test_a_counter_that_did_not_move_means_not_tested(moved: Row) -> None:
    read = fake_read({"V": FRESH}, {"V": (1.0,) * 3}, moved)
    assert provisional(AT, SEEDS)((V,), read).status == "not_tested"


def test_a_missing_counter_means_not_tested() -> None:
    def read(seeds: tuple[int, ...], at: int) -> dict[str, np.ndarray]:
        return {"V": np.full(len(seeds), 1.0 if at else 0.0)}

    assert provisional(AT, SEEDS)((V,), read).status == "not_tested"


@pytest.mark.parametrize("after", [(1.0, np.nan, 1.0), (1.0, np.inf, 1.0)])
def test_a_non_finite_reading_after_training_means_diverged(after: Row) -> None:
    assert judge(FRESH, after).status == "diverged"


def test_a_row_unreadable_before_training_is_not_tested() -> None:
    before = {"V": (np.nan, 0.0, 0.0), "W": (0.5, 0.5, 0.5)}  # W stays out of band
    after = {"V": (1.0,) * 3, "W": (0.0,) * 3}
    got = provisional(AT, SEEDS)((V, W), fake_read(before, after))
    assert got.status == "not_tested"


def test_a_wrong_reading_outranks_an_unreadable_start() -> None:
    assert judge((np.nan, 0.0, 0.0), (0.5, 1.0, 1.0)).status == "flagged"


def test_a_run_that_did_not_train_is_not_tested_even_if_it_is_not_finite() -> None:
    read = fake_read({"V": FRESH}, {"V": (np.nan,) * 3}, (0, 0, 0))
    assert provisional(AT, SEEDS)((V,), read).status == "not_tested"


def test_diverged_outranks_a_flag_on_another_seed() -> None:
    assert judge(FRESH, (np.nan, 0.0, 1.0)).status == "diverged"


def test_training_that_raises_is_crashed_and_named() -> None:
    def read(seeds: tuple[int, ...], at: int) -> dict[str, np.ndarray]:
        raise RuntimeError("buffer exploded")

    got = provisional(AT, SEEDS)((V,), read)
    assert got.status == "crashed"
    assert "buffer exploded" in got.report


def test_readings_of_the_wrong_length_are_crashed_not_broadcast() -> None:
    read = fake_read({"V": (0.0,)}, {"V": (1.0,)})
    assert provisional(AT, SEEDS)((V,), read).status == "crashed"


def test_a_counter_of_the_wrong_length_is_crashed_not_broadcast() -> None:
    read = fake_read({"V": FRESH}, {"V": (1.0,) * 3}, (5,))
    assert provisional(AT, SEEDS)((V,), read).status == "crashed"


def test_a_rule_needs_a_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        provisional(AT, ())


def test_a_tolerance_tighter_than_the_band_is_used() -> None:
    assert judge(FRESH, (1.05, 1.0, 1.0)).status == "passed"
    assert judge(FRESH, (1.05, 1.0, 1.0), tolerances={"V": 0.02}).status == "flagged"


def test_a_tolerance_wider_than_the_band_is_refused() -> None:
    with pytest.raises(ValueError, match="V"):
        judge(FRESH, FRESH, tolerances={"V": 0.11})


def test_a_tolerance_for_no_query_is_refused_not_ignored() -> None:
    with pytest.raises(KeyError):
        judge(FRESH, FRESH, tolerances={"typo": 0.01})


def test_it_reads_before_training_and_at_the_budget_with_these_seeds() -> None:
    asked = []
    inner = fake_read({"V": FRESH}, {"V": (1.0,) * 3})

    def read(seeds: tuple[int, ...], at: int) -> dict[str, np.ndarray]:
        asked.append((seeds, at))
        return inner(seeds, at)

    provisional(AT, SEEDS)((V,), read)
    assert sorted(asked) == [(SEEDS, 0), (SEEDS, AT)]


def test_the_report_names_the_nearest_wrong_answer() -> None:
    query = Query("V(0)", 0.0, 0.8, {"gamma squared": 0.64, "no discount": 1.0})
    read = fake_read({"V(0)": FRESH}, {"V(0)": (1.0, 1.0, 1.0)})
    report = provisional(AT, SEEDS)((query,), read).report
    assert "no discount" in report and "gamma squared" not in report


# --- prove ---------------------------------------------------------------------


class Agent:
    """A fake agent whose value is whatever gamma it trained at, with hooks to
    misbehave: ``deaf`` ignores the fault, ``healthy``/``faulted`` replace the
    trained readings, ``fresh`` is the reading before training."""

    def __init__(
        self,
        deaf: bool = False,
        healthy: Any = None,
        faulted: Any = None,
        counts: bool = True,
        fresh: Any = 0.0,
    ) -> None:
        self.deaf, self.healthy, self.faulted = deaf, healthy, faulted
        self.counts, self.fresh = counts, fresh

    def reader(self, probe: Probe, **settings: float) -> Any:
        g = probe.gamma if self.deaf else settings.get("gamma", probe.gamma)
        trained = self.faulted if settings else self.healthy
        values = {"V(0)": g, "V(1)": 1.0} if trained is None else trained

        def read(seeds: tuple[int, ...], at: int) -> dict[str, np.ndarray]:
            n = len(seeds)
            if isinstance(values, Exception):
                raise values
            fresh = self.fresh if settings else 0.0
            start = fresh if isinstance(fresh, dict) else dict.fromkeys(values, fresh)
            out = {k: np.full(n, v if at else start[k]) for k, v in values.items()}
            moved = self.counts or not settings
            return out | {UPDATES: np.full(n, 5 if at and moved else 0)}

        return read


PROBE, RULE = discounting(0.8), provisional(AT, SEEDS)


def proof(agent: Agent) -> Any:
    return prove(PROBE, agent, no_discount, RULE)


def test_a_probe_that_catches_its_fault_is_proven() -> None:
    got = proof(Agent())
    assert got.proven
    assert (got.healthy.status, got.faulted.status) == ("passed", "flagged")
    assert (got.probe, got.fault) == ("discounting", "no_discount")


def test_a_fault_that_passes_proves_nothing() -> None:
    assert not proof(Agent(deaf=True)).proven


def test_a_healthy_run_that_fails_proves_nothing() -> None:
    assert not proof(Agent(healthy={"V(0)": 0.5, "V(1)": 1.0})).proven


def test_a_faulted_run_that_crashed_proves_nothing() -> None:
    assert not proof(Agent(faulted=RuntimeError("no such setting"))).proven


def test_a_faulted_run_that_was_not_tested_proves_nothing() -> None:
    assert not proof(Agent(counts=False)).proven


def test_a_faulted_run_that_could_not_have_failed_proves_nothing() -> None:
    clean = {"V(0)": 0.8, "V(1)": 1.0}  # on the truth before and after training
    assert not proof(Agent(faulted=clean, fresh=clean)).proven


def test_a_faulted_run_that_never_trained_proves_nothing() -> None:
    agent = Agent(faulted={"V(0)": np.nan, "V(1)": np.nan}, fresh=np.nan, counts=False)
    assert not proof(agent).proven


def test_a_faulted_run_that_diverged_is_caught() -> None:
    assert proof(Agent(faulted={"V(0)": np.inf, "V(1)": 1.0})).proven


def test_a_fault_is_judged_against_the_probes_truth_not_its_own_gamma() -> None:
    # Against the gamma it trained at, 0.5 would pass; the probe's truth is 0.8.
    got = prove(PROBE, Agent(), Fault("custom", {"gamma": 0.5}), RULE)
    assert got.faulted.status == "flagged"


def test_no_discount_trains_at_gamma_one() -> None:
    assert dict(no_discount.settings) == {"gamma": 1.0}


# --- the discounting probe ------------------------------------------------------


@pytest.mark.parametrize(
    "gamma, v0_band",
    [
        (0.8, 0.08),  # nearest wrong answer gamma^2 = 0.64: half of 0.16
        (0.62, 0.1),  # nearest is 0.3844, gap 0.2356: the ceiling sets it
        (0.5, 0.1),  # gamma^2 = 0.25, gap 0.25; 1 - gamma = 0.5 names nothing
    ],
)
def test_discounting_answers_and_bands(gamma: float, v0_band: float) -> None:
    v0, v1 = discounting(gamma).queries
    assert (v0.name, v0.obs, v1.name, v1.obs) == ("V(0)", 0.0, "V(1)", 1.0)
    assert v0.truth == gamma and v1.truth == 1.0
    assert v0.band == pytest.approx(v0_band)
    assert v1.band == 0.1  # 1 / (1 - gamma^2) is at least 4/3 here: gap over 0.2


def test_discounting_names_keep_ajaxs_answers_at_its_gamma() -> None:
    v0, v1 = discounting(0.62).queries
    assert v0.wrong == pytest.approx(
        {
            "gamma squared": 0.3844,
            "1 - gamma": 0.38,
            "no discount": 1.0,
            "done flag one step late": 0.0,
        }
    )
    assert v1.wrong == pytest.approx({"done mask ignored": 1 / (1 - 0.3844)})


def test_a_wrong_answer_equal_to_the_truth_is_dropped() -> None:
    assert "1 - gamma" not in discounting(0.5).queries[0].wrong


@pytest.mark.parametrize("gamma", [0.0, 1.0, -0.1, 1.5])
def test_discounting_needs_a_gamma_strictly_between_0_and_1(gamma: float) -> None:
    with pytest.raises(ValueError, match="gamma"):
        discounting(gamma)


@pytest.mark.parametrize(
    "env_class",
    [gymnax_envs.RewardDiscountingEnv, continuous_actions.RewardDiscountingEnv],
)
def test_discounting_truths_are_the_real_envs_discounted_returns(
    env_class: Any,
) -> None:
    gamma = 0.8
    env, params = env_class(), env_class().default_params
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key, params)
    trace, done = [], False
    while not done:
        action = env.action_space(params).sample(key)
        next_obs, state, reward, terminated, truncated, _ = env.step(
            key, state, action, params
        )
        trace.append((float(obs[0]), float(reward)))
        obs, done = next_obs, bool(terminated or truncated)

    returns = {}
    so_far = 0.0
    for x, reward in reversed(trace):  # discounted return from each observation
        so_far = reward + gamma * so_far
        returns[x] = so_far

    v0, v1 = discounting(gamma).queries
    assert returns == {0.0: pytest.approx(v0.truth), 1.0: pytest.approx(v1.truth)}
    assert discounting(gamma).env == env_class.__name__


def test_discounting_truths_hold_on_the_gymnasium_env_from_its_second_episode() -> None:
    # Its first episode runs one step long, 0 -> 0 -> 1 -> end: reset() does not
    # clear its observation stack. A finding, left for the generated envs.
    gamma = 0.8
    env = envs.RewardDiscountingEnv()
    for _ in range(2):
        obs, _ = env.reset()
        trace, done = [], False
        while not done:
            next_obs, reward, done, _, _ = env.step(0)
            trace.append((float(obs[0]), float(reward)))
            obs = next_obs

    v0, v1 = discounting(gamma).queries
    assert trace == [(v0.obs, 0.0), (v1.obs, 1.0)]  # V(1) = 1, V(0) = gamma * 1
    assert (v0.truth, v1.truth) == (gamma, 1.0)
