import numpy as np


def play(agent, env, episodes) -> list[int]:
    print("#########Playing....###################")
    rewards = []
    for _ in range(1, 1 + episodes):
        observation, info = env.reset()
        session_reward = 0
        while True:
            if env.envs[0].unwrapped.spec.id == "Hockey-One-v0": env.render()

            action = agent.act(observation, enable_noise=False)
            observation, reward, terminated, truncated, info = env.step(action)

            session_reward += reward

            if terminated or truncated:
                break

        rewards.append(session_reward)
    env.close()
    print(f"avg play reward: {np.average(rewards)}")
    return rewards
