import pickle

import laserhockey.hockey_env as h_env
import numpy as np
import torch

from RL2023HockeyTournamentClient.client.backend import Client
from RL2023HockeyTournamentClient.client.remoteControllerInterface import RemoteControllerInterface
from agents.TD3Agent import TD3Agent
from helper.Logger import RunInfo


class RemoteBasicOpponent(RemoteControllerInterface):

    def __init__(self):
        with open(f'results/TD3_Hockey-v0_normal2_1_1.pickle', 'rb') as f:
            data: RunInfo | None = pickle.load(f)
        if data is None: return
        state = torch.load(f'results/TD3_Hockey-v0_normal2_1_1.torch', map_location=torch.device('cpu'))
        hyper = data.hyper
        # Initialize the Gym environment
        env = h_env.HockeyEnv_BasicOpponent(mode=h_env.HockeyEnv.NORMAL, weak_opponent=True)
        self.agent = TD3Agent(env.observation_space, env.action_space, hyper.agent)
        self.agent.restore_state(state)

        RemoteControllerInterface.__init__(self, identifier='JTD3')

    def remote_act(self,
                   obs: np.ndarray,
                   ) -> np.ndarray:
        return self.agent.act(obs)


if __name__ == '__main__':
    controller = RemoteBasicOpponent()

    # Play n (None for an infinite amount) games and quit
    client = Client(username='The Q-Learners',
                    password='eeghieV4ka',
                    controller=controller,
                    output_path='logs/basic_opponents',  # rollout buffer with finished games will be saved in here
                    interactive=False,
                    op='start_queuing',
                    # server_addr='localhost',
                    num_games=None)
