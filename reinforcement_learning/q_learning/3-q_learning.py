import gym
import numpy as np


def train(
    env: gym.Wrapper,
    Q: np.ndarray,
    episodes: int = 5000,
    max_steps: int = 100,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: int = 1,
    min_epsilon: float = 0.1,
    epsilon_decay: float = 0.05,
) -> tuple[np.ndarray, list[float]]:
    rewards_history = []
    for episode in range(episodes):
        state = env.reset()[0]
        current_reward = 0
        for _ in range(max_steps):
            if np.random.uniform() > epsilon:
                action = np.argmax(Q[state])
            else:
                action = np.random.randint(0, len(Q[state]))

            new_state, reward, done, _, info = env.step(action)

            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (
                reward + gamma * np.max(Q[new_state, :])
            )
            current_reward += reward
            state = new_state

            if done:
                break

        epsilon = min_epsilon + (1 - min_epsilon) * np.exp(-epsilon_decay * episode)

        rewards_history.append(current_reward)
    return Q, rewards_history
