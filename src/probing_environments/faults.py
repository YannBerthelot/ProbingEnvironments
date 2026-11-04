"""Bugs planted through ordinary settings, with no edit to the code under
test, so one fault runs on every library. Provisional, like ``runner``."""

import dataclasses
from typing import Mapping


@dataclasses.dataclass(frozen=True)
class Fault:
    """``settings`` go to the agent's ``reader``. So far they are named as SB3
    and Ajax both name them; a library that differs maps them in its adaptor."""

    id: str
    settings: Mapping[str, float]


no_discount = Fault("no_discount", {"gamma": 1.0})
