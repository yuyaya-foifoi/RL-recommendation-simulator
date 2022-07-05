import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import CFG_DICT


class PolicyNet(nn.Module):
    def __init__(self, state_size: int, action_size: int) -> None:
        super(PolicyNet, self).__init__()
        hidden_dim = CFG_DICT["ACTOR_CRITIC"]["POLICY_HIDDEN_DIM"]
        self.l1 = nn.Linear(state_size, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_size)

    def forward(self, x: np.array) -> torch.tensor:

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.softmax(self.l3(x), dim=0)

        return x
