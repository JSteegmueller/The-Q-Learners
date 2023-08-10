from enum import Enum

import torch

torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AgentParams:
    def __init__(self):
        # noise
        self.explore_noise = 0.2
        self.target_noise = 0.2
        self.noise_clip = 0.5
        self.noise_decay = 0

        # q value discount
        self.discount: float = 0.99

        # tau for policy update (soft update)
        self.tau = 0.005

        # learning rates
        self.learning_rate_actor: float = 0.001
        self.learning_rate_critic: float = 0.001

        # size of replay buffer and batch
        self.buffer_size: int = int(1e5)
        self.buffer_threshold: int = 0
        self.batch_size: int = 100

        # how often to update target functions (every x gradients)
        self.update_target_every: int = 2

        # Network hidden sizes
        self.hidden_sizes: int = 128

        self.random_timesteps: int = 0


class GymParams:
    def __init__(self):
        self.env_name: GymEnv = GymEnv.HOCKEY
        self.env_params_train: dict = {}
        self.env_params_play: dict = {"render_mode": "human"}
        self.render: bool = False

        # Max episode
        self.episodes: int = 15000

        # Logger
        self.log_interval: int = 50
        self.save_interval: int = 500
        self.random_seed: any = None


class HyperParams:
    def __init__(self):
        self.agent: AgentParams = AgentParams()
        self.gym: GymParams = GymParams()
        self.hyper_id = ""


class GymEnv(Enum):
    PENDULUM = "Pendulum-v1"
    CAR = "MountainCarContinuous-v0"
    HOCKEY = "Hockey-v0"
    LUNAR = "LunarLander-v2"
