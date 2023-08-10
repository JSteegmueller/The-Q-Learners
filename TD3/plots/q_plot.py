import pickle
import string

import laserhockey.hockey_env as h_env
import numpy as np
import torch
from matplotlib import pyplot as plt

from TD3.agents.TD3Agent import TD3Agent
from TD3.helper.Logger import RunInfo
from TD3.helper.Util import to_torch


def plot_q_function(q_function, observations, actions, wins):
    plt.rcParams.update({'font.size': 12})
    values, _ = q_function.forward(observations, actions)

    _, ax = plt.subplots()
    ax.scatter(np.clip(observations[:, -6], -5, 5), np.clip(observations[:, -5], -5, 5), c=values.detach())
    ax.set_xlabel("Puck x pos")
    ax.set_ylabel("Puck y pos")
    _, bx = plt.subplots()
    bx.scatter(np.hstack([observations[:, 0], observations[:, 6]]), np.hstack([observations[:, 1], observations[:, 7]]),
               c=np.hstack([values.detach(), values.detach()]))
    bx.set_xlabel("Player x pos")
    bx.set_ylabel("Player y pos")
    _, cx = plt.subplots()
    cx.scatter(-observations[:, -3], observations[:, -4],
               c=values.detach())
    cx.set_xlabel("Puck x velo")
    cx.set_ylabel("Puck y velo")
    _, dx = plt.subplots()
    dx.scatter(range(1000), wins)
    dx.set_xlabel("time")
    dx.set_ylabel("win percentage")
    plt.show()


def run(env, agent, n_episodes=1000, act=False, render=False):
    rewards = []
    observations = []
    actions = []
    won = 0
    env.metadata["render_fps"] = 60
    win_percentage = []
    for ep in range(1, n_episodes + 1):
        ep_reward = 0
        state, _info = env.reset()
        for t in range(2000):
            if render: env.render()
            action = agent.act(state, enable_noise=False)
            state, reward, done, _trunc, _info = env.step(action) if act else env.step(
                np.hstack([action, [0, 0, 0, 0]]))
            observations.append(state)
            actions.append(action)
            ep_reward += reward
            if done or _trunc:
                break
        rewards.append(ep_reward)
        won += 1 if _info['winner'] >= 1 else 0
        win_percentage.append(won / ep)
    print(f'Won percentage: {won / n_episodes}')
    print(f'Mean reward: {np.mean(rewards)}')
    observations = np.asarray(observations)
    actions = np.asarray(actions)
    return observations, actions, rewards, win_percentage


def q_plot(file: string, render: bool):
    with open(f'{file}.pickle', 'rb') as f:
        data: RunInfo | None = pickle.load(f)
    if data is None: return
    state = torch.load(f'{file}.torch', map_location=torch.device('cpu'))
    hyper = data.hyper
    # Initialize the Gym environment
    env = h_env.HockeyEnv_BasicOpponent(mode=h_env.HockeyEnv.NORMAL, weak_opponent=True)
    agent = TD3Agent(env.observation_space, env.action_space, hyper.agent)
    agent.restore_state(state)

    # Define the grid for states
    observations, actions, rewards, win_percentage = run(env, agent, act=True, render=render)
    plot_q_function(agent.dq, to_torch(observations), to_torch(actions), win_percentage)
