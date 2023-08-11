from DuelingDQNAgent import load_agent
import argparse
import laserhockey.hockey_env as h_env
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path to model')
parser.add_argument('--max_steps', type=int, help='max steps per episode', default=1000)
parser.add_argument('--render_fps', type=int, help='fps for rendering', default=30)
parser.add_argument('--render', action='store_true', help='render environment')
parser.add_argument('--easy', action='store_true', help='play against easy opponent')
parser.add_argument('--human', action='store_true', help='play against human opponent')
parser.add_argument('--selfplay', action='store_true', help='play against self')
parser.add_argument('--agent_side', type=str, help='side of agent', default='left')
parser.add_argument('--agent_opponent', type=str, help='opponent of agent', default=None)

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

state_scaling = np.array([ 1.0,  1.0, 0.5, 4.0, 4.0, 4.0,  
                           1.0,  1.0, 0.5, 4.0, 4.0, 4.0,  
                           2.0,  2.0, 10.0, 10.0, 4.0, 4.0])

if __name__ == '__main__':
    args = parser.parse_args()

    # load agent
    agent = load_agent(args.path)
    agent.act = lambda state: action_set[agent(
        state / state_scaling
        ).item()]
    
    # prepare environment
    render_mode = "human" if args.render else None
    env = h_env.HockeyEnv(keep_mode=True)
    env.metadata["render_fps"] = args.render_fps

    if args.human:
        opponent = h_env.HumanOpponent(env, player=2)
    elif args.selfplay:
        opponent = load_agent(args.path)
        opponent.act = lambda state: action_set[opponent(
            state / state_scaling
            ).item()]
    elif args.agent_opponent is not None:
        opponent = load_agent(args.agent_opponent)
        opponent.act = lambda state: action_set[opponent(
            state / state_scaling
            ).item()]
    else:
        opponent = h_env.BasicOpponent(weak=args.easy)
    
    if args.agent_side == 'right':
        player_1 = opponent
        player_2 = agent
    else:
        player_1 = agent
        player_2 = opponent

    enemy_wins = 0
    agent_wins = 0
    draws = 0

    # run agent
    while True:
        state, info = env.reset()    

        done = False
        for i in range(args.max_steps):
            action_1 = player_1.act(state)
            action_2 = player_2.act(env.obs_agent_two())
            state, reward, done, trunc, info = env.step(np.hstack([action_1, action_2]))
            if args.render:
                env.render()

            if done or trunc or i == args.max_steps:
                    
                if info["winner"] == 1:
                    agent_wins += 1
                elif info["winner"] == -1:
                    enemy_wins += 1
                else:
                    draws += 1
        
                print("\033[A                             \033[A")
                print(f"wins: {agent_wins}, loses: {enemy_wins}, draws: {draws}, winrate: {agent_wins / (agent_wins + enemy_wins + draws)}")

                break
