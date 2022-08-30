import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from configs.config import CFG_DICT

logger = logging.getLogger(__name__)


class PolicyNet(nn.Module):
    def __init__(self, state_size: int, action_size: int) -> None:
        super(PolicyNet, self).__init__()
        logger.info("Policy Net, state dim : {}".format(state_size))
        logger.info("Policy Net, action dim : {}".format(action_size))
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


class SACPolicyNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(SACPolicyNet, self).__init__()

        logger.info("Policy Net, state dim : {}".format(state_size))
        logger.info("Policy Net, action dim : {}".format(action_size))
        hidden_dim = CFG_DICT["SOFT_ACTOR_CRITIC"]["POLICY_HIDDEN_DIM"]

        self.max = torch.from_numpy(
            np.array(CFG_DICT["DATASET"]["NUM_ITEMS"] - 1)
        )
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.linear_mu = nn.Linear(hidden_dim, action_size)
        self.linear_std = nn.Linear(hidden_dim, action_size)

    def forward(self, x: np.array) -> torch.tensor:

        if not len(x.shape) > 1:
            x = torch.unsqueeze(x, 0)

        x = self.net(x)
        mu = self.linear_mu(x)
        std = self.linear_std(x)
        std = F.softplus(std) + 1e-3

        dist = Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        log_prob -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(
            dim=-1, keepdim=True
        )

        action = torch.sigmoid(action) * self.max
        return action, log_prob


class DDPGPolicyNet(nn.Module):
    """
    方策勾配法を使っていないことに注意

    一般的な方策関数 :
      1) 状態を入力
      2) 行動の確率分布を出力
      3) 行動をサンプリング
      ->この手順によって, 方策勾配法に基づいて方策関数のパラメタを更新できる
      方策勾配法 : G_{t}が大きければ \pi(a_{t} | s_{t})が大きくなるように学習する

    決定論的な方策 :
      Q(s, \mu_{s})を最大にするように更新する
    """

    def __init__(self, state_size, action_size, max):
        super(DDPGPolicyNet, self).__init__()
        logger.info("Policy Net, state dim : {}".format(state_size))
        logger.info("Policy Net, action dim : {}".format(action_size))
        hidden_dim = CFG_DICT["DDPG"]["POLICY_HIDDEN_DIM"]
        self.min = torch.from_numpy(np.array([0]))
        self.max = torch.from_numpy(
            np.array(CFG_DICT["DATASET"]["NUM_ITEMS"] - 1)
        )
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_size),
            nn.Sigmoid(),
        )

    def mu(self, x):
        return self.net(x.float()) * self.max

    def forward(self, x, epsilon=0.0):
        mu = self.mu(x)
        # 学習テクニック : 探索ノイズ
        # DQNでは探索を促進するために \eps-Greedyを採用していた
        # DDPGでは探索を促進するために方策関数の出力のアクションにノイズを載せる
        # 論文ではOUノイズを採用している
        mu = mu + torch.normal(0, epsilon, mu.size(), device=mu.device)
        action = torch.max(torch.min(mu, self.max), self.min)
        return action
