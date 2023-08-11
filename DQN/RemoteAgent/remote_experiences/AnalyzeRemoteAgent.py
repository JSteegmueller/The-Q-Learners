import argparse
import laserhockey.hockey_env as h_env
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('experiences', type=str, help='path to experiences')
parser.add_argument('--render_fps', type=int, help='fps for rendering', default=60)
parser.add_argument('--agent_side', type=str, help='side of agent', default='left')

if __name__ == '__main__':
    args = parser.parse_args()
    
    # prepare environment
    render_mode = "human"
    env = h_env.HockeyEnv(keep_mode=True)
    env.metadata["render_fps"] = args.render_fps

    # run agent
    for filename in os.listdir(args.experiences):
        if not(".DS_Store" in filename):
            experience = os.path.join(args.experiences, filename)
            experience = np.load(experience, allow_pickle=True)["arr_0"]
            states = experience.item()["transitions"]
            player_one = experience.item()["player_one"] 
            player_two = experience.item()["player_two"]
            if "StrongBasicOpponent" in {player_one, player_two} \
                    or "WeakBasicOpponent" in {player_one, player_two}:
                continue
            print(player_two, "vs", player_one)
            print("file name", filename)
            for last_observation, last_action, next_obs, r, done, trunc, info in states:
                if info["winner"] != 0:
                    print(info["winner"])
                state = np.array(next_obs)
                CENTER_X = h_env.CENTER_X 
                CENTER_Y = h_env.CENTER_Y

                env.player1.position = (state[[0, 1]] + [CENTER_X, CENTER_Y]).tolist()
                env.player1.angle = state[2]
                env.player2.position = (state[[6, 7]] + [CENTER_X, CENTER_Y]).tolist()
                env.player2.angle = state[8]
                env.player2.linearVelocity = [state[9], state[10]]
                env.player2.angularVelocity = state[11]
                env.puck.position = (state[[12, 13]] + [CENTER_X, CENTER_Y]).tolist()
                
                env.render()
