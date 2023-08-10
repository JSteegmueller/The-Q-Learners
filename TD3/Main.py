from typing import Dict, Tuple

import laserhockey.hockey_env as h_env

from HyperHyper import HyperParams, GymParams
from agents.TD3Agent import TD3Agent
from helper.Logger import Logger
from helper.Memory import Memory
from loops.train import train
from experiments import hy_update_every, hy_learning_rate, hy_hidden_size, hy_tau, hy_discount, hy_noise, \
    hy_random_timesteps, hy_buffer_threshold


def main():
    tests: Dict[str, Tuple[GymParams, list[HyperParams]]] = dict[str, Tuple[GymParams, list[HyperParams]]]()
    tests['update_every'] = (HyperParams().gym, [hy_update_every(i) for i in range(1, 6)])
    tests['learning_rate'] = (HyperParams().gym, [hy_learning_rate(i) for i in range(1, 6)])
    tests['hidden_size'] = (HyperParams().gym, [hy_hidden_size(i) for i in range(1, 6)])
    tests['tau'] = (HyperParams().gym, [hy_tau(i) for i in range(1, 6)])
    tests['discount'] = (HyperParams().gym, [hy_discount(i) for i in range(1, 6)])
    tests['noise'] = (HyperParams().gym, [hy_noise(i) for i in range(1, 6)])
    tests['random_timesteps'] = (HyperParams().gym, [hy_random_timesteps(i) for i in range(1, 6)])
    tests['buffer_threshold'] = (HyperParams().gym, [hy_buffer_threshold(i) for i in range(1, 6)])
    run_tests(tests, runs_per_hyper=10)


def run_tests(tests: Dict[str, Tuple[GymParams, list[HyperParams]]],
              runs_per_hyper: int = 1):
    for test_name, params in tests.items():
        (gym_params, hypers) = params
        print(f'#########Testing: {test_name}###############')
        test_name: str
        gym_params: GymParams
        hypers: list[HyperParams]

        i = 0
        for hyper in hypers:
            i += 1

            for hyper_run in range(1, runs_per_hyper + 1):
                hyper.hyper_id = f'{test_name}_{i}_{hyper_run}'
                print(f'#########Run ID: {hyper.hyper_id}###############')

                envs, names, num_trains = get_hockey_iteration(defending=0, shooting=0, normal=1, hard=0)
                agent = TD3Agent(envs[0].observation_space, envs[0].action_space, hyper.agent)
                logger = Logger(agent, hyper)
                for env, name in zip(envs, names):
                    print(f'{name}')
                    train(agent=agent,
                          env=env,
                          episodes=gym_params.episodes,
                          random_timesteps=hyper.agent.random_timesteps,
                          logger=logger,
                          num_train=num_trains[0])
                    agent.buffer = Memory(max_size=agent.buffer.max_size)


def get_hockey_iteration(defending, shooting, normal, hard):
    env_shooting = h_env.HockeyEnv_BasicOpponent(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
    env_defending = h_env.HockeyEnv_BasicOpponent(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
    env_normal = h_env.HockeyEnv_BasicOpponent(mode=h_env.HockeyEnv.NORMAL, weak_opponent=True)
    env_hard = h_env.HockeyEnv_BasicOpponent(mode=h_env.HockeyEnv.NORMAL, weak_opponent=False)

    envs = [env_defending for _ in range(defending)] + \
           [env_shooting for _ in range(shooting)] + \
           [env_normal for _ in range(normal)] + \
           [env_hard for _ in range(hard)]
    names = ["defending" for _ in range(defending)] + \
            ["shoot" for _ in range(shooting)] + \
            ["normal" for _ in range(normal)] + \
            ["hard" for _ in range(hard)]
    num_trains = [100]
    return envs, names, num_trains


if __name__ == '__main__':
    main()
