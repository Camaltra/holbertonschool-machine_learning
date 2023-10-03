import numpy as np


def epsilon_greedy(Q: np.ndarray, state: int, epsilon: float) -> int:
    if np.random.uniform() > epsilon:
        return np.argmax(Q[state])
    return np.random.randint(0, len(Q[state]))
