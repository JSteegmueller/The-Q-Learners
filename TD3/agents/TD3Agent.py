import copy
from typing import SupportsFloat

import numpy as np
import torch
from gymnasium import spaces
from gymnasium.core import ObsType
from numpy import ndarray, number
from torch import Tensor

from HyperHyper import AgentParams, torch_device
from helper.Memory import Memory
from helper.UnsupportedSpace import UnsupportedSpace
from helper.Util import soft_update_params
from neural_nets.PolicyFunction import PolicyFunction
from neural_nets.QFunction import DQFunction


# import glfw


class TD3Agent(object):
    """
    Agent implementing Q-learning with NN function approximation.
    """

    def __init__(self, observation_space, action_space, hyper: AgentParams):

        if not isinstance(observation_space, spaces.box.Box):
            raise UnsupportedSpace('Observation space {} incompatible ' \
                                   'with {}. (Require: Box)'.format(observation_space, self))
        if not isinstance(action_space, spaces.box.Box):
            raise UnsupportedSpace('Action space {} incompatible with {}.' \
                                   ' (Require Box)'.format(action_space, self))

        self._hyper = hyper
        self._observation_space = observation_space
        self._obs_dim = self._observation_space.shape[0]
        self._action_space = action_space
        self._action_n = action_space.shape[0]
        self._max_action = torch.FloatTensor(self._action_space.high).to(torch_device)
        self.buffer = Memory(max_size=self._hyper.buffer_size)

        # Q Network
        self.dq = DQFunction(observation_dim=self._obs_dim,
                             action_dim=self._action_n,
                             hidden_size=self._hyper.hidden_sizes,
                             learning_rate=self._hyper.learning_rate_critic).to(torch_device)
        self.dq_target: DQFunction

        self.policy = PolicyFunction(observation_dim=self._obs_dim, hidden_size=self._hyper.hidden_sizes,
                                     action_dim=self._action_n, lr=self._hyper.learning_rate_actor,
                                     max_action=self._max_action).to(torch_device)
        self.policy_target: DQFunction
        self._copy_nets()

        self.train_iter = 0
        self._episode_count = 0

    def act(self, observation, enable_noise=True, random=False):

        if random:
            return self._action_space.sample()

        action = self.policy.forward(torch.FloatTensor(observation).to(torch_device)).cpu().detach().numpy()
        noise = np.random.normal(0, self._hyper.explore_noise, size=action.shape)
        action += enable_noise * noise
        action = np.clip(action, -self._max_action.cpu().numpy(), self._max_action.cpu().numpy())
        return action

    def train(self, t) -> list[[number, number, number]]:
        self._episode_count += 1
        losses = []
        if self.buffer.size < self._hyper.buffer_threshold: return []

        for i in range(t):
            self.train_iter += 1
            # sample from the replay buffer
            states, actions, rewards, next_states, dones = self.buffer.sample(batch=self._hyper.batch_size)
            with torch.no_grad():
                # Calculate noise added to the next action
                noise: Tensor = torch.clip(torch.rand_like(actions) * self._hyper.target_noise,
                                           -self._hyper.noise_clip,
                                           self._hyper.noise_clip)

                # calculate next actions with noise
                next_actions = torch.clip(self.policy_target(next_states) + noise,
                                          -self._max_action,
                                          self._max_action)

                next_q1, next_q2 = self.dq_target.forward(next_states, next_actions)

                target_qs = rewards + (1 - dones) * self._hyper.discount * torch.min(next_q1, next_q2)

            # assign q_loss_value and actor_loss to be stored in the statistics
            dq_loss = self.dq.fit(states, actions, target_qs)
            actor_loss = -self.dq.q1(states, self.policy(states)).mean()
            if self.train_iter % self._hyper.update_target_every == 0:
                self.policy.fit(actor_loss)
                self._soft_update(self._hyper.tau)
            losses.append([dq_loss, actor_loss.item()])
        return losses

    def store_transition(self, transition: tuple[ObsType, ndarray, SupportsFloat, ObsType, bool]):
        self.buffer.add_transition(transition)

    def state(self) -> dict:
        return {"dq": self.dq.state_dict(),
                "dq_opti": self.dq.optimizer.state_dict(),
                "policy": self.policy.state_dict(),
                "policy_opti": self.policy.optimizer.state_dict()
                }

    def restore_state(self, state: dict):
        self.dq.load_state_dict(state["dq"])
        self.dq.optimizer.load_state_dict(state["dq_opti"])

        self.policy.load_state_dict(state["policy"])
        self.policy.optimizer.load_state_dict(state["policy_opti"])

        self._copy_nets()

    def _copy_nets(self):
        self.dq_target = copy.deepcopy(self.dq)
        self.policy_target = copy.deepcopy(self.policy)

    def _soft_update(self, tau):
        soft_update_params(self.dq_target, self.dq, tau)
        soft_update_params(self.policy_target, self.policy, tau)
