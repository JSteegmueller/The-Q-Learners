import numpy as np
import torch

from HyperHyper import torch_device


# class to store transitions
class Memory:
    def __init__(self, max_size):
        self.transitions = np.asarray([])
        self.size = 0
        self.current_idx = 0
        self.max_size = max_size

    def add_transition(self, transitions_new):
        if self.size == 0:
            blank_buffer = [np.asarray(transitions_new, dtype=object)] * self.max_size
            self.transitions = np.asarray(blank_buffer)

        self.transitions[self.current_idx, :] = np.asarray(transitions_new, dtype=object)
        self.size = min(self.size + 1, self.max_size)
        self.current_idx = (self.current_idx + 1) % self.max_size

    def sample(self, batch=1):
        if batch > self.size:
            batch = self.size
        idx = np.random.randint(0, self.size, size=batch)

        states = torch.FloatTensor(np.stack(self.transitions[idx, 0])).to(torch_device)
        actions = torch.FloatTensor(np.stack(self.transitions[idx, 1])).to(torch_device)
        rewards = torch.FloatTensor(np.stack(self.transitions[idx, 2])[:, None]).to(torch_device)
        next_states = torch.FloatTensor(np.stack(self.transitions[idx, 3])).to(torch_device)
        dones = torch.FloatTensor(np.stack(self.transitions[idx, 4])[:, None]).to(torch_device)
        return states, actions, rewards, next_states, dones

    def get_all_transitions(self):
        return self.transitions[0:self.size]
