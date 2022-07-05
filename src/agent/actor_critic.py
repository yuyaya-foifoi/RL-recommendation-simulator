import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from configs.config import CFG_DICT
from src.model.policy import PolicyNet
from src.model.value import ValueNet


class ActorCriticAgent:
    def __init__(self, device) -> None:

        self.device = device

        # 環境に関する情報
        self.action_size = CFG_DICT["DATASET"]["NUM_ITEMS"]
        self.state_size = CFG_DICT["DATASET"]["NUM_ITEMS"] + 5

        self.gamma = CFG_DICT["ACTOR_CRITIC"]["GAMMA"]
        self.lr_pl_net = CFG_DICT["ACTOR_CRITIC"]["POLICY_LR"]
        self.lr_v_net = CFG_DICT["ACTOR_CRITIC"]["VALUE_LR"]

        self.pl_net = PolicyNet(self.state_size, self.action_size).to(
            self.device
        )
        self.v_net = ValueNet(self.state_size, self.action_size).to(
            self.device
        )

        self.optimizer_pl = torch.optim.Adam(
            self.pl_net.parameters(), lr=self.lr_pl_net
        )
        self.optimizer_v = torch.optim.Adam(
            self.v_net.parameters(), lr=self.lr_v_net
        )

        self.loss_function = nn.SmoothL1Loss()

    def get_action(self, state, size):
        probs = self.pl_net(state)
        m = Categorical(probs)
        action = m.sample((size,))
        return probs, action

    def get_target_probs(self, state, action):

        probs = self.pl_net(state)
        n_batch, _ = probs.shape

        return probs[np.arange(n_batch), action].detach()

    def update(self, state, reward, next_state, prob_behavior, prob_target):

        self.optimizer_v.zero_grad()
        self.optimizer_pl.zero_grad()

        rho = prob_target / prob_behavior

        td_target = rho * (reward + self.gamma * self.v_net(next_state))
        td_target.detach()
        v = rho * self.v_net(state)

        loss_v = self.loss_function(v, td_target)

        delta = td_target - v
        loss_pl = (-torch.log(prob_behavior) * delta).mean()

        loss_v.backward(retain_graph=True)
        loss_pl.backward(retain_graph=True)

        self.optimizer_v.step()
        self.optimizer_pl.step()
