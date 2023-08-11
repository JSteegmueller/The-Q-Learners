## Simple training run for the Dueling DQN agent

import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import json
import sys
sys.path.append("../..")

import TrainHockey 
from DuelingDQNAgent import save_agent, load_agent
from GenerateReport import generate_summary
import laserhockey.hockey_env as h_env

torch.set_num_threads(24)
torch.set_num_interop_threads(24)

###############################
# Making extensive agent pool #
###############################

# adding act function
def add_act_function(agent):
    agent.act = lambda state:\
        TrainHockey.action_set[
                agent(state / TrainHockey.state_scaling).item()
        ]

# add a opponent from path to opponent list
def opp_to_list(list, path, times = 1):
    agent = load_agent(path)
    add_act_function(agent)
    list += [agent] * times

# weak and strong opponent
opponent_pool = [h_env.BasicOpponent(weak=False),
                 h_env.BasicOpponent(weak=True),] * 10 

# agent that is overfitted on the easy opponent 
opp_to_list(opponent_pool, "../Reproduce80EasyRun/DuelingDQNAgent", 5)

# agent that is overfitted on the hard opponent
opp_to_list(opponent_pool, "../Reproduce80Run/DuelingDQNAgent", 5)

# agent that was trained to beat a lot of coppies of itself
opp_to_list(opponent_pool, "../Reproduce80SelfplayRun2/DuelingDQNAgent", 5)

# agent that was trained on various other agents 
opp_to_list(opponent_pool, "../Reproduce80SelfplayRun3/DuelingDQNAgent", 5)

# agent that was trained on even more agents
opp_to_list(opponent_pool, "../Reproduce80SelfplayRun4/DuelingDQNAgent", 20)

# agent that was trained on even more agents
opp_to_list(opponent_pool, "../Reproduce80SelfplayRun5/DuelingDQNAgent", 20)

## Add some more agents to the pool that resulted from parameter searches with the hard agent
# network depth
for n in [2, 3, 4, 5]:
    opp_to_list(opponent_pool, f"../DepthRuns/depth_{n}/DuelingDQNAgent")

# forward view
for n in [1, 2, 3, 4, 5]:
    opp_to_list(opponent_pool, f"../ForwardViewRuns/n_{n}/DuelingDQNAgent")

# batch size
for n in [16, 32, 64]:
    opp_to_list(opponent_pool, f"../BatchSizeRuns/size_{n}/DuelingDQNAgent")

# q update
for n in [10, 25, 50, 100, 500, 1000]:
    opp_to_list(opponent_pool, f"../QUpdateRuns/qupdate_{n}/DuelingDQNAgent")

# set hyper parameters
hyper_parameters = {
        "seed": 42,
        "max_transitions": 100,
        "mode" : 0,
        "gamma": 0.99,
        "feature_space_dim": 1024,
        "segment_width": 1024,
        "segment_depth": 2,
        "activation_function": [torch.nn.ReLU()] * 4,
        "loss_function": torch.nn.SmoothL1Loss(),
        "optimizer": torch.optim.Adam,
        "learning_rate": 0.0002,
        "learning_rate_halving_episodes": [75_000, 80_000, 85_000],
        "optimization_steps": 10,
        "batch_size": 128,
        "device": "cpu",
        "epsilon0": 0.2,
        "eps_decay": 0.99995,
        "weak_opponent": False,
        "add_self_episodes": [i for i in range(50_000,
                                               90_000,
                                               10_000)],
        "opponent_pool": opponent_pool,
        "save_at_checkpoint": None,
        "forward_view_n": 3,
        "experience_buffer_size": 100_000
        }

trainer = TrainHockey.BasicHockeyTrainer(**hyper_parameters)

training_parameters = {
        "num_episodes": 90_000,
        "checkpoint_interval": 500,
        "q_target_update_interval": 50,
        "num_testing_episodes": 100,
        }

trainer.train(**training_parameters)

save_agent(trainer.agent, "DuelingDQNAgent")

generate_summary(trainer.scores,
                 trainer.losses,
                 trainer.winners,
                 trainer.test_winners,
                 training_parameters["num_testing_episodes"],
                 trainer.transitions,
                 "report/")

json.dump({k: str(v) for k, v in hyper_parameters.items()},
          open("report/hyperparameters.json", "w"),
          indent=4)

json.dump({k: str(v) for k, v in training_parameters.items()},
          open("report/training_parameters.json", "w"),
          indent=4)

trainer.save_raw_data("report/raw_data.json")
