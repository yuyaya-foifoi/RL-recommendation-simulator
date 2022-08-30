import copy

import torch

from configs.config import CFG_DICT
from src.model.policy import SACPolicyNet
from src.model.value import ValueNet
from src.utils.loss import get_loss_function


class SAC:
    def __init__(self, device):

        super().__init__()

        self.device = device

        self.action_size = CFG_DICT["DATASET"]["NUM_ITEMS"]
        self.state_size = CFG_DICT["DATASET"]["NUM_ITEMS"] + 5
        self.max_action = CFG_DICT["DATASET"]["NUM_ITEMS"]
        self.gamma = CFG_DICT["SOFT_ACTOR_CRITIC"]["GAMMA"]

        self.lr_pl_net = CFG_DICT["SOFT_ACTOR_CRITIC"]["POLICY_LR"]
        self.lr_v_net = CFG_DICT["SOFT_ACTOR_CRITIC"]["VALUE_LR"]

        self.q_net1 = ValueNet(self.state_size).to(self.device)
        self.q_net2 = ValueNet(self.state_size).to(self.device)
        self.policy = SACPolicyNet(self.state_size, self.action_size).to(
            self.device
        )

        self.target_q_net1 = copy.deepcopy(self.q_net1)
        self.target_q_net2 = copy.deepcopy(self.q_net2)
        self.target_policy = copy.deepcopy(self.policy)

        self.optimizer_pl = torch.optim.Adam(
            self.policy.parameters(), lr=self.lr_pl_net
        )
        self.optimizer_v_1 = torch.optim.Adam(
            self.q_net1.parameters(), lr=self.lr_v_net
        )
        self.optimizer_v_2 = torch.optim.Adam(
            self.q_net2.parameters(), lr=self.lr_v_net
        )

        self.cnt = 0
        self.alpha = CFG_DICT["SOFT_ACTOR_CRITIC"]["ALPHA"]
        self.tau = CFG_DICT["SOFT_ACTOR_CRITIC"]["TAU"]
        self.loss_function = get_loss_function(
            CFG_DICT["SOFT_ACTOR_CRITIC"]["LOSS"]
        )

    def get_action(self, x, size):
        action = self.policy(x)[0]
        action = torch.squeeze(action[:, :size], 0)
        return None, action

    def update(self, state, reward, next_state):
        self.cnt += 1
        self.optimizer_v_1.zero_grad()
        self.optimizer_v_2.zero_grad()

        q_loss_total = self.update_v(state, reward, next_state)
        q_loss_total.backward()
        self.optimizer_v_1.step()
        self.optimizer_v_2.step()

        if self.cnt % 2 == 0:
            self.optimizer_pl.zero_grad()
            policy_loss = self.update_p(state, reward, next_state)
            policy_loss.backward()
            self.optimizer_pl.step()

        self.update_target_w()

    def update_v(self, state, reward, next_state):

        action_values1 = self.q_net1(state)
        action_values2 = self.q_net2(state)

        target_actions, target_log_probs = self.target_policy(next_state)

        next_actions_values = torch.min(
            self.target_q_net1(next_state),
            self.target_q_net2(next_state),
        )

        expected_action_values = torch.unsqueeze(reward, 1) + self.gamma * (
            next_actions_values - self.alpha * target_log_probs
        )

        q_loss1 = self.loss_function(action_values1, expected_action_values)
        q_loss2 = self.loss_function(action_values2, expected_action_values)

        q_loss_total = q_loss1 + q_loss2

        return q_loss_total

    def update_p(self, state, reward, next_state):

        actions, log_probs = self.policy(state)

        action_values = torch.min(
            self.target_q_net1(next_state), self.target_q_net2(next_state)
        )

        policy_loss = (self.alpha * log_probs - action_values).mean()
        return policy_loss

    def polyak_average(self, net, target_net, tau):
        for qp, tp in zip(net.parameters(), target_net.parameters()):
            tp.data.copy_(tau * qp.data + (1 - tau) * tp.data)

    def update_target_w(self):

        self.polyak_average(self.q_net1, self.target_q_net1, tau=self.tau)
        self.polyak_average(self.q_net2, self.target_q_net2, tau=self.tau)
        self.polyak_average(self.policy, self.target_policy, tau=self.tau)
