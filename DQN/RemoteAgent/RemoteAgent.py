import sys
sys.path.append("..")
import numpy as np
import torch

from laserhockey.hockey_env import BasicOpponent
from client.remoteControllerInterface import RemoteControllerInterface
from client.backend.client import Client
from DuelingDQNAgent import load_agent
import argparse

torch.set_num_threads(4)
torch.set_num_interop_threads(4)

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument('agent', type=str)

# constants
state_scaling = np.array([ 1.0,  1.0, 0.5, 4.0, 4.0, 4.0,  
                           1.0,  1.0, 0.5, 4.0, 4.0, 4.0,  
                           2.0,  2.0, 10.0, 10.0, 4.0, 4.0])

action_set = np.array([
[0, 0, 0, 0],       # 0 stand
[-1, 0, 0, 0],      # 1 left
[1, 0, 0, 0],       # 2 right
[0, -1, 0, 0],      # 3 down
[0, 1, 0, 0],       # 4 up
[0, 0, -1, 0],      # 5 clockwise
[0, 0, 1, 0],       # 6 counter-clockwise
[-1, -1, 0, 0],     # 7 left down
[-1, 1, 0, 0],      # 8 left up
[1, -1, 0, 0],      # 9 right down
[1, 1, 0, 0],       # 10 right up
[-1, -1, -1, 0],    # 11 left down clockwise
[-1, -1, 1, 0],     # 12 left down counter-clockwise
[-1, 1, -1, 0],     # 13 left up clockwise
[-1, 1, 1, 0],      # 14 left up counter-clockwise
[1, -1, -1, 0],     # 15 right down clockwise
[1, -1, 1, 0],      # 16 right down counter-clockwise
[1, 1, -1, 0],      # 17 right up clockwise
[1, 1, 1, 0],       # 18 right up counter-clockwise
[0, 0, 0, 1],       # 19 shoot
])

# controller class
class RemoteAgent(RemoteControllerInterface):

    def __init__(self, path):
        self.path = path
        self.agent = load_agent(path)
        RemoteControllerInterface.__init__(self,
                                           identifier='DDQN')

    def remote_act(self, 
            obs : np.ndarray,
           ) -> np.ndarray:

        return action_set[self.agent(obs / state_scaling).item()]

    def before_game_starts(self):
        try:
            new_agent = load_agent(self.path)
            self.agent = new_agent
        except:
            print('Could not load agent')


# main
if __name__ == '__main__':
    args = parser.parse_args()
    controller = RemoteAgent(args.agent)

    client = Client(username='The Q-Learners',
                    password='eeghieV4ka',
                    controller=controller, 
                    output_path='remote_experiences/',
                    interactive=False,
                    op='start_queuing',
                    num_games=None)
