from typing import Callable, Any

import numpy as np
import pytest

EPS = 1e-1
GAMMA = 0.5

AgentType = Any


def env_1(agent: AgentType, get_value: Callable[[AgentType, np.ndarray], np.ndarray]):
    agent.learn(1e3)
    assert pytest.approx(1, abs=EPS) == get_value(agent, agent.get_env().reset())


def env_2(agent: AgentType, get_value: Callable[[AgentType, np.ndarray], np.ndarray]):
    agent.learn(1e3)
    assert pytest.approx(0, abs=EPS) == get_value(agent, np.array(0))
    assert pytest.approx(1, abs=EPS) == get_value(agent, np.array(1))


def env_3(agent, get_value):
    agent.learn(1e3)
    assert pytest.approx(GAMMA, rel=EPS) == get_value(agent, np.array(
        [0]))  # Comment gérer le gamma pour l'imposer sur l'agent?
    assert pytest.approx(1, rel=EPS) == get_value(agent, np.array([1]))
