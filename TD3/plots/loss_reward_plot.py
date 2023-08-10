import json
import pickle
from numbers import Number

import numpy as np
import pylab as plt

from TD3.helper.Logger import RunInfo


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def plot_q1_loss(file, q_fig, act_fig, reward_fig, win_fig):
    run_infos: list[RunInfo] = []
    for ii in range(1, 1 + 1):
        with open(f'{file}_{ii}.pickle', 'rb') as f:
            run_infos.append(pickle.load(f))

    hyper = run_infos[0].hyper

    if hasattr(run_infos[0], 'win_percentages'):
        win_percentages = np.asarray([info.win_percentages for info in run_infos])
    losses = np.asarray([info.losses for info in run_infos])
    rewards = np.asarray([info.rewards for info in run_infos])

    # mean_q_loss = np.mean(losses[:, :, 0], axis=0)
    # std_q_loss = np.std(losses[:, :, 0], axis=0)
    # min_q_loss = mean_q_loss - std_q_loss
    # max_q_loss = mean_q_loss + std_q_loss
    #
    # mean_actor_loss = np.mean(losses[:, :, 1], axis=0)
    # std_actor_loss = np.std(losses[:, :, 1], axis=0)
    # min_actor_loss = mean_actor_loss - std_actor_loss
    # max_actor_loss = mean_actor_loss + std_actor_loss
    #
    # mean_reward = np.mean(rewards, axis=0)
    # std_reward = np.std(rewards, axis=0)
    # min_reward = mean_reward - std_reward
    # max_reward = mean_reward + std_reward

    if hasattr(run_infos[0], 'win_percentages'):
        mean_win = np.mean(win_percentages, axis=0)
        std_win = np.std(win_percentages, axis=0)
        min_win = np.clip(mean_win - std_win / 2, 0, 1)
        max_win = np.clip(mean_win + std_win / 2, 0, 1)

    changed_value = hyper.agent.discount
    # q_fig.fill_between(np.arange(len(running_mean(min_q_loss, 1000))), running_mean(min_q_loss, 1000),
    #                    running_mean(max_q_loss, 1000),
    #                    alpha=0.5)
    # q_fig.plot(running_mean(mean_q_loss, 1000), label=f"Q1 loss {changed_value}")
    #
    # act_fig.fill_between(np.arange(len(running_mean(min_actor_loss, 1000))), running_mean(min_actor_loss, 1000),
    #                      running_mean(max_actor_loss, 1000),
    #                      alpha=0.5)
    # act_fig.plot(running_mean(mean_actor_loss, 1000), label=f"Actor loss {changed_value}")
    #
    # reward_fig.fill_between(np.arange(len(running_mean(min_reward, 1000))), running_mean(min_reward, 1000),
    #                         running_mean(max_reward, 1000),
    #                         alpha=0.5)
    # reward_fig.plot(running_mean(mean_reward, 1000), label=f"Reward {changed_value}")

    if hasattr(run_infos[0], 'win_percentages'):
        win_fig.fill_between(np.arange(len(running_mean(min_win, 10))), running_mean(min_win, 10),
                             running_mean(max_win, 10),
                             alpha=0.1)
        win_fig.plot(running_mean(mean_win, 10), label=f"TD3")

    with open(f'../results/raw_data.json', 'rb') as f:
        leo_data = json.load(f)
        scores: list[Number] = leo_data['winners']
    scores = scores[:10000]
    avg_scores: list[Number] = list()
    for y in range(1, 201):
        yy = y * 50
        batch = scores[yy - 50:yy]
        avg_scores.append(sum([1 if ww > 0 else 0 for ww in batch]) / 50)
    win_fig.plot(running_mean(avg_scores, 10), label=f"DDQN")
    with open(f'../results/tim.pkl', 'rb') as f:
        tim_data = pickle.load(f)
        w_l = tim_data['rewards']
    avg_scores: list[Number] = list()
    for y in range(1, 201):
        yy = y * 50
        batch = w_l[yy - 50:yy]
        avg_scores.append(sum([1 if ww > 0 else 0 for ww in batch]) / 50)
    win_fig.plot(running_mean(avg_scores, 10), label=f"SAC")


gym_name = 'Hockey-v0'
test_name = 'normal1'
# _, q_loss = plt.subplots()
# _, actor_loss = plt.subplots()
# _, reward = plt.subplots()
_, win = plt.subplots()
win.set_xlabel('Episode / 50')
win.set_ylabel('Win rate %')
for i in range(1, 1 + 1):
    plot_q1_loss(f'../results/TD3_{gym_name}_{test_name}_{i}', _, _, _, win)
# q_loss.legend()
# actor_loss.legend()
# reward.legend()
win.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.show()
