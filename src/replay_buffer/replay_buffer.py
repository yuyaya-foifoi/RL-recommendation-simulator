import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, buffer_size: int) -> None:
        self.buffer = deque(maxlen=buffer_size)

    def add(
        self,
        state: np.array,
        action: int,
        reward: float,
        next_state: np.array,
        prob: float,
    ) -> None:

        data = (state, action, reward, next_state, prob)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self, batch_size):
        data = random.sample(self.buffer, batch_size)

        state = np.stack([x[0] for x in data])
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        prob = np.array([x[4] for x in data])

        return (
            torch.from_numpy(state).float(),
            torch.from_numpy(action),
            torch.from_numpy(reward).float(),
            torch.from_numpy(next_state).float(),
            torch.from_numpy(prob).float(),
        )
