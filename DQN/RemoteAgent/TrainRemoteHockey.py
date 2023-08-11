"""
Trainer Classes for the Hockey Environment.
"""
import os
import sys
sys.path.append("..")
import argparse
import numpy as np
import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(4)

from tqdm import tqdm
from time import sleep

import laserhockey.hockey_env as h_env
from DuelingDQNAgent import DuelingDQNEnsembleAgent, \
        DuelingDQNAgent, save_agent, load_agent
from ExperienceBuffer import ExperienceBuffer

# constants
state_scaling = np.array([ 1.0,  1.0, 0.5, 4.0, 4.0, 4.0,  
                           1.0,  1.0, 0.5, 4.0, 4.0, 4.0,  
                           2.0,  2.0, 10.0, 10.0, 4.0, 4.0])

action_set = [
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
]

class RemoteTrainer:
    """
    A trainer that takes experiences from remotely played games
    """
    def __init__(self,
                 path_to_base_agent,
                 gamma = 0.99,
                 replay_factor = 1,
                 learning_rate = 0.000025,):
        self.agent = load_agent(path_to_base_agent)
        self.agent.gamma = gamma
        self.agent.experience_buffer = \
            ExperienceBuffer(self.agent.state_space_dim,
                             "cpu",
                             1_000_000)
        self.agent.QFunction.update_learning_rate(learning_rate)
        self.loaded_experiences = set()
        self.new_games_since_last_update = 0
        self.replay_factor = replay_factor
        self.first_load = True
    
    def update_experience_buffer(self,
                                 path_to_experiences):
        # go through experience files
        self.new_games_since_last_update = 0
        for filename in os.listdir(path_to_experiences):
            if not ".DS_Store" in filename and not filename in self.loaded_experiences:
                # file identifier to loaded_experiences
                self.loaded_experiences.add(filename)
                if not self.first_load:
                    self.new_games_since_last_update += 4

                # load experience
                experience = os.path.join(path_to_experiences, filename)
                experience = np.load(experience, 
                                     allow_pickle=True)["arr_0"]
                transitions = experience.item()["transitions"]

                # get the identifiers
                player_one = experience.item()["player_one"] 
                player_two = experience.item()["player_two"]

                # do not add StrongBasicOpponent or WeakBasicOpponent
                #if "StrongBasicOpponent" in \
                #        {player_one, player_two} \
                #        or "WeakBasicOpponent" in \
                #        {player_one, player_two}:
                #    continue
                strange_data = 0
                for state, action, next_state, reward, done, trunc, info in transitions:

                    # cast to numpy arrays
                    state = np.array(state)
                    action = action_set.index(action)           
                    next_state = np.array(next_state)
                    
                    # add experience to buffer
                    self.agent.experience_buffer.add_experience(
                        state / state_scaling,
                        action,
                        reward,
                        next_state / state_scaling,
                        float(done))                
        self.first_load = False

    def train(self, 
              q_target_update_interval,
              updated_agent_path,):

        # training loop
        if self.new_games_since_last_update == 0:
            return

        iterator = tqdm(range(self.new_games_since_last_update * self.replay_factor))
        for i in iterator:
            # optimize agent
            losses = self.agent.train()
            mean_loss = np.mean(losses[-100:])
            iterator.set_description("Training Started! loss: {:.3f}".format(mean_loss))
            # update target network
            if i % q_target_update_interval == 0:
                self.agent.update_frozen_QFunction()

        # save agent
        save_agent(self.agent, updated_agent_path)

    
    def continuos_update_and_train(self,
                                   path_to_experiences,
                                   q_target_update_interval,
                                   updated_agent_path,):
        while True:
            self.update_experience_buffer(path_to_experiences)
            self.train(q_target_update_interval,
                       updated_agent_path,)
            sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_base_agent",
                        type=str,
                        required=True,
                        help="path to the base agent")
    parser.add_argument("--path_to_experiences",
                        type=str,
                        required=True,
                        help="path to the experiences")
    parser.add_argument("--gamma",
                        type=float,
                        default=0.99,
                        help="discount factor")
    parser.add_argument("--replay_factor",
                        type=int,
                        default=1,
                        help="number of times to replay each experience")
    parser.add_argument("--learning_rate",
                        type=float,
                        default=0.00025,
                        help="learning rate")
    parser.add_argument("--q_target_update_interval",
                        type=int,
                        default=50,
                        help="number of iterations between target network updates")
    parser.add_argument("--updated_agent_path",
                        type=str,
                        required=True,
                        help="path to save updated agent")
    args = parser.parse_args()

    trainer = RemoteTrainer(args.path_to_base_agent,
                            args.gamma,
                            args.replay_factor,
                            args.learning_rate)

    trainer.continuos_update_and_train(args.path_to_experiences,
                                       args.q_target_update_interval,
                                       args.updated_agent_path)
