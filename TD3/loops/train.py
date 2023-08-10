import numpy as np

from agents.TD3Agent import TD3Agent
from helper.Logger import Logger


def train(agent: TD3Agent, env, logger: Logger, episodes, random_timesteps, num_train):
    print("#########Training....###################")
    episode = 0
    t = 0
    for episode in range(1, 1 + episodes):
        observation, info = env.reset()
        episode_t = 0
        episode_reward = 0
        while True:
            episode_t += 1
            t += 1

            action = agent.act(observation, random=(t <= random_timesteps))
            new_observation, reward, terminated, truncated, info = env.step(
                action if env.mode is env.NORMAL else np.hstack([action, [0, 0., 0, 0]]))
            agent.store_transition((observation, action, reward, new_observation, terminated))
            episode_reward += reward
            observation = new_observation
            if terminated or truncated:
                break
        episode_loss = agent.train(num_train)
        logger.log(loss=episode_loss, reward=episode_reward, length=episode_t, episode=episode,
                   won=max(0, info['winner']))
    logger.save(episode)
    env.close()
