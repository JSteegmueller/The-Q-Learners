import numpy as np
import torch

from HyperHyper import torch_device


def to_torch(x):
    return torch.from_numpy(x.astype(np.float32)).to(torch_device)


def soft_update_params(target_net, source_net, tau):
    for param, target_param in zip(source_net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
