import torch
import torch.nn.functional as F
from torch import nn


class PolicyFunction(torch.nn.Module):
    def __init__(self, observation_dim, hidden_size, action_dim, lr, max_action):
        super().__init__()

        self.max_action = max_action
        self.layer1 = nn.Linear(observation_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, action_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr)

    def forward(self, state):
        l1 = F.relu(self.layer1(state))
        l2 = F.relu(self.layer2(l1))
        l3 = torch.tanh(self.layer3(l2))

        return l3 * self.max_action

    def fit(self, policy_loss):
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
