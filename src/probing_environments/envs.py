import random

import gym
import numpy as np
from gym import spaces


class ProbeEnv1(gym.Env):
    """
    One action, zero observation, one timestep long, +1 reward every timestep:\
    This isolates the value network. If my agent can't learn that the value of\
    the only observation it ever sees it 1, there's a problem with the value \
        loss calculation or the optimizer.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Discrete(1)

    def step(self, action):
        return np.array((0)), 1, True, {}

    def reset(self, seed=None):
        return np.array((0))


def get_random_obs():
    return random.choice([0, 1])


class ProbeEnv2(gym.Env):
    """
    One action, random +1/-1 observation, one timestep long, obs-dependent \
    +1/-1 reward every time: If my agent can learn the value in ProbeEnv1 but not \
    this one - meaning it can learn a constant reward but not a \
    predictable one! - it must be that backpropagation through my network is broken.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Discrete(2)
        self.random_obs = get_random_obs()

    def step(self, action):
        return np.array(get_random_obs()), self.random_obs, True, {}

    def reset(self, seed=None):
        # Reset the state of the environment to an initial state
        np.random.seed(seed)
        self.random_obs = get_random_obs()
        return np.array(self.random_obs)


class ProbeEnv3(gym.Env):
    """
    One action, zero-then-one observation, two timesteps long, +1 reward at \
    the end: If my agent can learn the value in (2.) but not this one, it must\
     be that my reward discounting is broken.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Discrete(2)
        self.t = 0

    def step(self, action):
        self.t += 1
        return np.array([self.t]), int(self.t == 2), self.t == 2, {}

    def reset(self, seed=None):
        self.t = 0
        # Reset the state of the environment to an initial state
        return np.array([self.t])


# class ProbeEnv4(gym.Env):
#     """
#     Two actions, zero observation, one timestep long, action-dependent +1/-1\
#     reward: The first env to exercise the policy! If my agent can't learn \
#     to pick the better action, there's something wrong with either my \
#     advantage calculations, my policy loss or my policy update. That's three \
#     things, but it's easy to work out by hand the expected values for each one \
#     and check that the values produced by your actual code line up with them.
#     """

#     metadata = {"render.modes": ["human"]}

#     def __init__(self):
#         super().__init__()
#         # Define action and observation space
#         # They must be gym.spaces objects
#         # Example when using discrete actions:
#         self.action_space = spaces.Discrete(2)
#         # Example for using image as input:
#         self.observation_space = spaces.Discrete(1)

#     def step(self, action):
#         return np.array([0]), 1 if action == 0 else -1, True, {}

#     def reset(self, seed=None):
#         # Reset the state of the environment to an initial state
#         return np.array([0])


# class ProbeEnv5(gym.Env):
#     """
#     Two actions, random +1/-1 observation, one timestep long, action-and-obs \
#     dependent +1/-1 reward: Now we've got a dependence on both obs and action.\
#     The policy and value networks interact here, so there's a couple of things \
#     to verify: that the policy network learns to pick the right action in each \
#     of the two states, and that the value network learns that the value of \
#     each state is +1. If everything's worked up until now, then if - for \
#     example - the value network fails to learn here, it likely means your \
#     batching process is feeding the value network stale experience.
#     """

#     metadata = {"render.modes": ["human"]}

#     def __init__(self):
#         super().__init__()
#         self.action_space = spaces.Discrete(2)
#         self.observation_space = spaces.Discrete(1)
#         self.random_obs = None

#     def step(self, action):
#         reward = (
#             1
#             if (
#                 (self.random_obs == -1 and action == 0)
#                 or (self.random_obs == 1 and action == 1)
#             )
#             else -1
#         )
#         return np.array([self.random_obs]), reward, True, False, None

#     def reset(self, seed=None):
#         np.random.seed(seed)
#         self.random_obs = get_random_obs()
#         # Reset the state of the environment to an initial state
#         return np.array([self.random_obs]), None


# class ProbeEnv6(gym.Env):
#     """
#     One action, zero-then-one observation, two timesteps long, +1 reward at \
#     the end: If my agent can learn the value in (2.) but not this one, it must\
#      be that my reward discounting is broken.
#     """

#     metadata = {"render.modes": ["human"]}

#     def __init__(self):
#         super().__init__()
#         # Define action and observation space
#         # They must be gym.spaces objects
#         # Example when using discrete actions:
#         self.action_space = spaces.Discrete(1)
#         # Example for using image as input:
#         self.observation_space = spaces.Discrete(1)
#         self.t = 0

#     def step(self, action):
#         self.t += 1
#         return np.array([self.t]), int(self.t == 2), self.t == 2, False, None

#     def reset(self):
#         self.t = 0
#         # Reset the state of the environment to an initial state
#         return np.array([self.t]), None
