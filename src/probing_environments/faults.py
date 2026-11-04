"""Bugs planted through ordinary settings, with no edit to the code under
test, so one fault runs on every library. Provisional, like ``runner``."""

import dataclasses
from typing import Mapping


@dataclasses.dataclass(frozen=True)
class Fault:
    """``settings`` maps a role to its value; each adaptor knows its library's
    name for a role."""

    id: str
    settings: Mapping[str, float]


no_discount = Fault("no_discount", {"gamma": 1.0})
