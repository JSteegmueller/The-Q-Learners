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

easy_beater = load_agent("../Reproduce80EasyRun/DuelingDQNAgent")
easy_beater.act = lambda state:\
        TrainHockey.action_set[
                easy_beater(state / TrainHockey.state_scaling).item()
        ]

hard_beater = load_agent("../Reproduce80Run/DuelingDQNAgent")
hard_beater.act = lambda state:\
        TrainHockey.action_set[
                hard_beater(state / TrainHockey.state_scaling).item()
        ]

agent_beater = load_agent("../Reproduce80SelfplayRun2/DuelingDQNAgent")
agent_beater.act = lambda state:\
        TrainHockey.action_set[
                agent_beater(state / TrainHockey.state_scaling).item()
        ]

general_beater = load_agent("../Reproduce80SelfplayRun3/DuelingDQNAgent")
general_beater.act = lambda state:\
        TrainHockey.action_set[
                general_beater(state / TrainHockey.state_scaling).item()
        ]

opponent_pool = [h_env.BasicOpponent(weak=False),
                 h_env.BasicOpponent(weak=True),] * 3 \
              + [hard_beater, easy_beater] \
              + [agent_beater, general_beater] * 2
                

hyper_parameters = {
        "seed": 42,
        "max_transitions": 100,
        "mode" : 0,
        "gamma": 0.99,
        "feature_space_dim": 1024,
        "segment_width": 1024,
        "segment_depth": 2,
        "activation_function": torch.nn.ReLU(),
        "loss_function": torch.nn.SmoothL1Loss(),
        "optimizer": torch.optim.Adam,
        "learning_rate": 0.0002,
        "learning_rate_halving_episodes": [90_000, 95_000],
        "optimization_steps": 10,
        "batch_size": 128,
        "device": "cpu",
        "epsilon0": 0.2,
        "eps_decay": 0.99995,
        "weak_opponent": False,
        "add_self_episodes": [i for i in range(60_000,
                                               100_000,
                                               10_000)],
        "opponent_pool": opponent_pool,
        "save_at_checkpoint": None,
        "forward_view_n": 3,
        }

trainer = TrainHockey.BasicHockeyTrainer(**hyper_parameters)

training_parameters = {
        "num_episodes": 100_000,
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
