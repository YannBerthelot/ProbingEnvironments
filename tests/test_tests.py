import pytest
import gym
import numpy as np
from stable_baselines3.a2c import A2C
from probing_environments.premade import env_1, env_2, env_3
from probing_environments.envs import ProbeEnv1, ProbeEnv2, ProbeEnv3
from stable_baselines3.common.utils import obs_as_tensor
import torch

def get_value(agent, obs):
    return agent.policy.predict_values(torch.tensor(np.array([obs]))).detach().numpy()[0][0]

def test_tests():
    env_1(A2C("MlpPolicy", ProbeEnv1()), get_value)
    env_2(A2C("MlpPolicy", ProbeEnv2()), get_value)
    env_3(A2C("MlpPolicy", ProbeEnv3(), gamma=0.5), get_value)
