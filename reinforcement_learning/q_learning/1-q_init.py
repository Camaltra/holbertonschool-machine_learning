import gym
import numpy as np


def q_init(env: gym.Wrapper) -> np.ndarray:
    return np.zeros((env.observation_space.n, env.action_space.n))
