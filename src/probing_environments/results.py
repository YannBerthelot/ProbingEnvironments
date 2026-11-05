"""What a rule says about a row. Provisional, like ``runner``."""

import dataclasses
from typing import Literal

Status = Literal["passed", "flagged", "not_tested", "crashed", "diverged"]
CAUGHT: tuple[Status, ...] = ("flagged", "diverged")


@dataclasses.dataclass(frozen=True)
class Verdict:
    """``passed`` is the only clean state. ``flagged`` is a reading a person
    should look at; the other three say why there is no verdict on the agent."""

    status: Status
    report: str


@dataclasses.dataclass(frozen=True)
class Proof:
    """A probe and a fault, run healthy and faulted through one agent."""

    probe: str
    fault: str
    healthy: Verdict
    faulted: Verdict

    @property
    def proven(self) -> bool:
        """The healthy run passed and the rule caught the fault. A run that
        crashed or trained nothing shows nothing about the probe."""
        return self.healthy.status == "passed" and self.faulted.status in CAUGHT
