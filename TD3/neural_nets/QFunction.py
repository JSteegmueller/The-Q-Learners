import torch
import torch.nn.functional as F
from torch import nn


class DQFunction(torch.nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_size, learning_rate):
        super().__init__()

        obs_act_dim = observation_dim + action_dim

        # Q1
        self.layer_11 = nn.Linear(obs_act_dim, hidden_size)
        self.layer_12 = nn.Linear(hidden_size, hidden_size)
        self.layer_13 = nn.Linear(hidden_size, 1)

        # Q2
        self.layer_21 = nn.Linear(obs_act_dim, hidden_size)
        self.layer_22 = nn.Linear(hidden_size, hidden_size)
        self.layer_23 = nn.Linear(hidden_size, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), learning_rate)

    def forward(self, observation, action):
        obs_and_act = torch.cat([observation, action], 1)
        return self._q1(obs_and_act), self._q2(obs_and_act)

    def fit(self, observation, action, q_target):
        q1, q2 = self.forward(observation, action)
        loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def q1(self, observation, action):
        obs_and_act = torch.cat([observation, action], 1)
        return self._q1(obs_and_act)

    def _q1(self, obs_and_act):
        l11 = F.relu(self.layer_11(obs_and_act))
        l12 = F.relu(self.layer_12(l11))
        l13 = self.layer_13(l12)
        return l13

    def _q2(self, obs_and_act):
        l21 = F.relu(self.layer_21(obs_and_act))
        l22 = F.relu(self.layer_22(l21))
        l23 = self.layer_23(l22)
        return l23
