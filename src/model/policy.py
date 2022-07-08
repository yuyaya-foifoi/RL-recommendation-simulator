import numpy as np
import torch
import torch.nn as nn

from configs.config import CFG_DICT


class PolicyNet(nn.Module):
    def __init__(self, state_size: int, action_size: int) -> None:
        super(PolicyNet, self).__init__()
        hidden_dim = CFG_DICT["ACTOR_CRITIC"]["POLICY_HIDDEN_DIM"]
        self.l1 = nn.Linear(state_size, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_size)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: np.array) -> torch.tensor:

        if not len(x.shape) > 1:
            x = torch.unsqueeze(x, 0)

        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.softmax(self.l3(x))

        return x
