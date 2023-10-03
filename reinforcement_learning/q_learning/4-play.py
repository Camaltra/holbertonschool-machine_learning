import gym
import numpy as np


def play(env: gym.Wrapper, Q: np.ndarray, max_steps: int = 100) -> int:
    state = env.reset()[0]
    total_reward = 0
    print(env.render())
    for _ in range(max_steps):
        action = np.argmax(Q[state])
        state, reward, done, *_ = env.step(action)
        print(env.render())
        total_reward += reward

        if done:
            break

    return total_reward
