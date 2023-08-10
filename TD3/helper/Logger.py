import pickle
import time

import numpy as np
import torch

from TD3.HyperHyper import HyperParams


class Logger:
    def __init__(self, agent, hyper: HyperParams):
        self._rewards = []
        self._lengths = []
        self._losses = []
        self._win_percentages = []
        self._agent = agent
        self._hyper = hyper
        self._wins = 0
        self._last_time_logged = time.time()

    def log(self, loss, reward, length, episode, won):
        self._losses.extend(loss)
        self._rewards.append(reward)
        self._lengths.append(length)
        self._wins += won

        # logging every interval episodes
        if episode % self._hyper.gym.log_interval == 0:
            self._win_percentages.append(self._wins / self._hyper.gym.log_interval)
            self._print(episode)
            self._wins = 0

        # save every interval episodes
        if episode % self._hyper.gym.save_interval == 0:
            self.save(episode)

    def save(self, episode):
        print("########## Saving a checkpoint... ##########")
        torch.save(self._agent.state(), f'./results/TD3_{self._hyper.gym.env_name.value}_{self._hyper.hyper_id}.torch')

        with open(f'./results/TD3_{self._hyper.gym.env_name.value}_{self._hyper.hyper_id}.pickle', 'wb') as f:
            pickle.dump(RunInfo(self._rewards, self._lengths, episode, self._losses,
                                self._hyper.hyper_id, self._hyper, self._win_percentages), f)

    def _print(self, episode):
        avg_reward = np.mean(self._rewards[-self._hyper.gym.log_interval:])
        avg_length = int(np.mean(self._lengths[-self._hyper.gym.log_interval:]))
        now = time.time()
        print(f'Episode {episode} \t '
              f'Wins {self._wins}/{self._hyper.gym.log_interval} \t '
              f'avg length: {avg_length}  \t '
              f'length: {sum(self._lengths)} \t '
              f'avg reward: {np.round(avg_reward)} \t '
              f'time: {round(now - self._last_time_logged, 0)}')
        self._last_time_logged = now


class RunInfo:
    def __init__(self, rewards, lengths, episode, losses, run_id, hyper, win_percentages):
        self.rewards = rewards
        self.lengths = lengths
        self.t = episode
        self.losses = losses
        self.run_id = run_id
        self.hyper: HyperParams = hyper
        self.win_percentages = win_percentages
