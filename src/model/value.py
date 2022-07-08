import torch
import torch.nn as nn

from configs.config import CFG_DICT


class ValueNet(nn.Module):
    def __init__(self, state_size: int, action_size: int) -> None:
        super(ValueNet, self).__init__()
        hidden_dim = CFG_DICT["ACTOR_CRITIC"]["VALUE_HIDDEN_DIM"]
        self.l1 = nn.Linear(state_size, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()

    def forward(self, x) -> torch.tensor:
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.l3(x)
        return x
