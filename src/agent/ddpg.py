import copy

import torch

from configs.config import CFG_DICT
from src.model.policy import DDPGPolicyNet
from src.model.value import DDPGValueNet
from src.utils.loss import get_loss_function


class DDPG:
    """
    DDPGの問題点 :
      1) 行動価値を過大評価する -> TD3ではDoubleDQNのような手法を提案
      2) Q関数に似たような値ばかり渡される -> Q関数が過学習になる. TD3では方策関数の
    """

    def __init__(self, device):
        super(DDPG, self).__init__()

        self.device = device

        self.action_size = CFG_DICT["DATASET"]["NUM_ITEMS"]
        self.state_size = CFG_DICT["DATASET"]["NUM_ITEMS"] + 5
        self.max_action = CFG_DICT["DATASET"]["NUM_ITEMS"]
        self.gamma = CFG_DICT["DDPG"]["GAMMA"]

        self.lr_pl_net = CFG_DICT["DDPG"]["POLICY_LR"]
        self.lr_v_net = CFG_DICT["DDPG"]["VALUE_LR"]

        self.q_net = DDPGValueNet(self.state_size, self.action_size).to(
            self.device
        )
        self.policy = DDPGPolicyNet(
            self.state_size, self.action_size, self.max_action
        ).to(self.device)

        self.target_q_net = copy.deepcopy(self.q_net)
        self.target_policy = copy.deepcopy(self.policy)

        self.optimizer_pl = torch.optim.Adam(
            self.policy.parameters(), lr=self.lr_pl_net
        )
        self.optimizer_v = torch.optim.Adam(
            self.q_net.parameters(), lr=self.lr_v_net
        )

        self.cnt = 0
        self.alpha = CFG_DICT["SOFT_ACTOR_CRITIC"]["ALPHA"]
        self.tau = CFG_DICT["SOFT_ACTOR_CRITIC"]["TAU"]
        self.loss_function = get_loss_function(
            CFG_DICT["SOFT_ACTOR_CRITIC"]["LOSS"]
        )

    def get_action(self, x, size):
        action = self.policy.mu(x)
        return None, action[:size]

    def polyak_average(self, net, target_net, tau):
        for qp, tp in zip(net.parameters(), target_net.parameters()):
            tp.data.copy_(tau * qp.data + (1 - tau) * tp.data)

    def update(self, state, action, reward, next_state):
        self.cnt += 1
        self.optimizer_v.zero_grad()

        q_loss_total = self.update_v(state, action, reward, next_state)
        q_loss_total.backward()
        self.optimizer_v.step()

        if self.cnt % 2 == 0:
            self.optimizer_pl.zero_grad()
            policy_loss = self.update_p(state)
            policy_loss.backward()
            self.optimizer_pl.step()

        self.update_target_w()

    def update_target_w(self):

        self.polyak_average(self.q_net, self.target_q_net, tau=self.tau)
        self.polyak_average(self.policy, self.target_policy, tau=self.tau)

    def update_v(self, state, action, reward, next_state):

        action_values = self.q_net(state, torch.unsqueeze(action, 1))
        next_actions = self.target_policy.mu(next_state)
        next_action_values = self.target_q_net(
            next_state, torch.unsqueeze(next_actions[:, 0], 1)
        )

        expected_action_values = (
            torch.unsqueeze(reward, 1) + self.gamma * next_action_values
        )
        q_loss = self.loss_function(action_values, expected_action_values)

        return q_loss

    def update_p(self, state):
        mu = self.policy.mu(state)
        policy_loss = -self.q_net(state, torch.unsqueeze(mu[:, 0], 1)).mean()
        return policy_loss
