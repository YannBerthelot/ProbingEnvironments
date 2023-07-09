from collections import deque
from copy import deepcopy

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ValueLossOrOptimizerEnv(gym.Env):
    """
    One action, zero observation, one timestep long, +1 reward every timestep:\
    This isolates the value network. If my agent can't learn that the value of\
    the only observation it ever sees it 1, there's a problem with the value \
        loss calculation or the optimizer.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Box(0, 1, shape=(1,))

    def step(self, action):
        return np.array([0]), 1, True, False, {}

    def reset(self, seed=None):

        return np.array([0]), {}


def get_random_obs():
    return np.random.choice([0, 1])


class ValueBackpropEnv(gym.Env):
    """
    One action, random +1/-1 observation, one timestep long, obs-dependent \
    +1/-1 reward every time: If my agent can learn the value in ProbeEnv1 but not \
    this one - meaning it can learn a constant reward but not a \
    predictable one! - it must be that backpropagation through my network is broken.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, num_envs: int = 1, sequential=False):
        """
        Args:
            num_envs (int, optional): Number of vectorized environments. Defaults to 1.
            sequential (bool, optional): Are the vectorized environments processed \
                sequentially (True) or in parralel (False). Defaults to False.
        """
        super().__init__()
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Box(0, 1, shape=(1,))
        if sequential:
            self.obs_stack = deque([None for _ in range(num_envs)], maxlen=num_envs)
        else:
            self.obs_stack = deque([], maxlen=num_envs)

    def step(self, action):
        reward = self.obs_stack.popleft()
        return np.array([1]), reward, True, False, {}

    def reset(self, seed=None):
        np.random.seed(seed)
        random_obs = np.copy(get_random_obs())
        self.obs_stack.append(random_obs)
        return np.array([random_obs]), {}


class RewardDiscountingEnv(gym.Env):
    """
    One action, zero-then-one observation, two timesteps long, +1 reward at \
    the end: If my agent can learn the value in (2.) but not this one, it must\
     be that my reward discounting is broken.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, num_envs: int = 1, sequential=False):
        """
        Args:
            num_envs (int, optional): Number of vectorized environments. Defaults to 1.
            sequential (bool, optional): Are the vectorized environments processed \
                sequentially (True) or in parralel (False). Defaults to False.
        """
        super().__init__()
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Box(0, 1, shape=(1,))
        if sequential:
            self.obs_stack = deque([0 for _ in range(num_envs)], maxlen=num_envs)
        else:
            self.obs_stack = deque([0], maxlen=num_envs)

    def step(self, action):
        t = self.obs_stack.popleft()
        is_done = t == 2
        if is_done:
            self.obs_stack.append(1)
            obs_stack = deepcopy(self.obs_stack)
            assert obs_stack.pop() == 1
        else:
            self.obs_stack.append(t + 1)
            obs_stack = deepcopy(self.obs_stack)
            assert obs_stack.pop() == t + 1
        return np.array([t]), int(is_done), is_done, False, {}

    def reset(self, seed=None):
        return np.array([0]), {}


class AdvantagePolicyLossPolicyUpdateEnv(gym.Env):
    """
    Two actions, zero observation, one timestep long, action-dependent +1/-1\
    reward: The first env to exercise the policy! If my agent can't learn \
    to pick the better action, there's something wrong with either my \
    advantage calculations, my policy loss or my policy update. That's three \
    things, but it's easy to work out by hand the expected values for each one \
    and check that the values produced by your actual code line up with them.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, *args, **kwargs):
        """
        Args:
            num_envs (int, optional): Number of vectorized environments. Defaults to 1.
            sequential (bool, optional): Are the vectorized environments processed \
                sequentially (True) or in parralel (False). Defaults to False.
        """
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(0, 1, shape=(1,))

    def step(self, action):
        return np.array([0]), 1 if action == 0 else -1, True, False, {}

    def reset(self, seed=None):
        np.random.seed(seed)
        return np.array([0]), {}


class PolicyAndValueEnv(gym.Env):
    """
    Two actions, random +1/-1 observation, one timestep long, action-and-obs \
    dependent +1/-1 reward: Now we've got a dependence on both obs and action.\
    The policy and value networks interact here, so there's a couple of things \
    to verify: that the policy network learns to pick the right action in each \
    of the two states, and that the value network learns that the value of \
    each state is +1. If everything's worked up until now, then if - for \
    example - the value network fails to learn here, it likely means your \
    batching process is feeding the value network stale experience.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, num_envs: int = 1, sequential=False):
        """
        Args:
            num_envs (int, optional): Number of vectorized environments. Defaults to 1.
            sequential (bool, optional): Are the vectorized environments processed \
                sequentially (True) or in parralel (False). Defaults to False.
        """
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(0, 1, shape=(1,))
        if sequential:
            self.obs_stack = deque([None for _ in range(num_envs)], maxlen=num_envs)
        else:
            self.obs_stack = deque([], maxlen=num_envs)

    def step(self, action):
        random_obs = self.obs_stack.popleft()
        reward = (
            1
            if ((random_obs == 0 and action == 0) or (random_obs == 1 and action == 1))
            else -1
        )
        return np.array([random_obs]), reward, True, False, {}

    def reset(self, seed=None):
        np.random.seed(seed)
        random_obs = np.copy(get_random_obs())
        self.obs_stack.append(random_obs)
        return np.array([random_obs]), {}
