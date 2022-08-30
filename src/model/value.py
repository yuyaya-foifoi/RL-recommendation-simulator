import logging

import torch
import torch.nn as nn

from configs.config import CFG_DICT

logger = logging.getLogger(__name__)


class ValueNet(nn.Module):
    def __init__(self, state_size: int) -> None:
        super(ValueNet, self).__init__()
        logger.info("Value Net, state dim : {}".format(state_size))
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


class DDPGValueNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(DDPGValueNet, self).__init__()
        logger.info("Value Net, state dim : {}".format(state_size))
        hidden_dim = CFG_DICT["ACTOR_CRITIC"]["VALUE_HIDDEN_DIM"]
        self.net = nn.Sequential(
            nn.Linear(state_size + 1, hidden_dim),  # state_size + action
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        vector = torch.hstack((state, action))
        return self.net(vector.float())
