## Simple training run for the Dueling DQN agent

import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import sys
sys.path.append("../..")
import json
import os

import TrainHockey 
from DuelingDQNAgent import save_agent, load_agent
from GenerateReport import generate_summary

torch.set_num_threads(24)
torch.set_num_interop_threads(24)

hyper_parameters = {
        "seed": 42,
        "max_transitions": 100,
        "mode" : 0,
        "gamma": 0.99,
        "feature_space_dim": 256,
        "segment_width": 256,
        "segment_depth": 2,
        "activation_function": torch.nn.ReLU(),
        "loss_function": torch.nn.SmoothL1Loss(),
        "optimizer": torch.optim.Adam,
        "learning_rate": 0.0002,
        "learning_rate_halving_episodes": [15000, 25000],
        "optimization_steps": 10,
        "batch_size": 128,
        "device": "cpu",
        "epsilon0": 0.2,
        "eps_decay": 0.99995
        }

training_parameters = {
        "num_episodes": 30_000,
        "checkpoint_interval": 500,
        "q_target_update_interval": None,
        "num_testing_episodes": 100,
        }

for q_update in [10, 25, 50, 100, 500, 1000]:
    training_parameters["q_target_update_interval"] = q_update 
    trainer = TrainHockey.BasicHockeyTrainer(**hyper_parameters)
    
    os.makedirs(f"qupdate_{q_update}")
    os.makedirs(f"qupdate_{q_update}/report")
 
    trainer.train(**training_parameters)

    save_agent(trainer.agent, f"qupdate_{q_update}/DuelingDQNAgent")

    generate_summary(trainer.scores,
                     trainer.losses,
                     trainer.winners,
                     trainer.test_winners,
                     training_parameters["num_testing_episodes"],
                     trainer.transitions,
                     f"qupdate_{q_update}/report/")

    json.dump({k: str(v) for k, v in hyper_parameters.items()},
              open(f"qupdate_{q_update}/report/hyperparameters.json", "w"),
              indent=4)
    json.dump({k: str(v) for k, v in training_parameters.items()},
              open(f"qupdate_{q_update}/report/training_parameters.json", "w"),
              indent=4)

    trainer.save_raw_data(f"qupdate_{q_update}/report/raw_data.json")
