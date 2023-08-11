"""
Trainer Classes for the Hockey Environment.
"""
import numpy as np
import torch
from time import time
import json

import laserhockey.hockey_env as h_env
from DuelingDQNAgent import DuelingDQNEnsembleAgent, \
        DuelingDQNAgent, save_agent

# constants
state_scaling = np.array([ 1.0,  1.0, 0.5, 4.0, 4.0, 4.0,  
                           1.0,  1.0, 0.5, 4.0, 4.0, 4.0,  
                           2.0,  2.0, 10.0, 10.0, 4.0, 4.0])

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

# Useful Functions
def epsilon_greedy_policy(agent,
                          state, 
                          num_actions,
                          epsilon):
    # test if random action else use actor
    if np.random.random() < epsilon:
        return np.random.randint(num_actions, size=(1, 1))
    else:
        return agent(state) 

def optimistic_policy(agent,
                      state,
                      num_actions,
                      epsilon):   
    return agent.optimistic_action(state)

def test_agent(agent, 
               env, 
               opponent_pool,
               wrapper,  
               num_episodes,
               max_transitions):
    """
    Test an agent on an environment with a given opponent pool.

    Args:
        agent: agent to be tested
        env: environment to be tested on
        opponent_pool: list of opponents to be tested against
                       Note: these must have an act() method
        wrapper: function to wrap agent action to environment action
        num_episodes: number of episodes to test on
        max_transitions: maximum number of transitions per episode
    """

    # run agent on environment
    scores = []
    wins = []
    transitions = []
    for _ in range(num_episodes):

        # reset environment
        state, _ = env.reset()
        state /= state_scaling

        # select opponent
        opponent = np.random.choice(opponent_pool)

        done = False
        score = 0
        transition_count = 0
        
        for _ in range(max_transitions):
            action = agent(state)
            trans_action = wrapper(action.item())
            opponent_action = opponent.act(env.obs_agent_two())
            state, reward, done, _, info = env.step(np.hstack(
                [trans_action, opponent_action]))
            
            state /= state_scaling
            
            score += reward
            transition_count += 1

            if done:
                break
        
        scores.append(score)
        transitions.append(transition_count)
        wins.append(info["winner"])
    
    return scores, wins, transitions


class BasicHockeyTrainer:
    """
    A hockey trainer that has the option to add coppies of 
    itself to the opponent pool.
    """
    def __init__(self,
                 # seed
                 seed,
                 # Environment
                 max_transitions,
                 mode,
                
                 # Agent
                 gamma,
                 feature_space_dim,
                 segment_width,
                 segment_depth,
                 activation_function,

                 # Optimization
                 loss_function,
                 optimizer,
                 learning_rate,
                 learning_rate_halving_episodes,
                 optimization_steps,
                 batch_size,
                 device,
                 
                 # Exploration
                 epsilon0,
                 eps_decay,

                 # Weak Opponent
                 weak_opponent=False,

                 # Selfplay
                 add_self_episodes=[],

                 # Opponent Pool
                 opponent_pool = [],
                 save_at_checkpoint=None, #path

                 # Forward view n
                 forward_view_n=1,

                 # Experience buffer size
                 experience_buffer_size = 100_000,

                 # Action selector
                 action_selector = epsilon_greedy_policy
                 ):

        # Environment and Opponent
        self.env = h_env.HockeyEnv(mode=mode)
        if opponent_pool == []:
            basic_opponent = h_env.BasicOpponent(weak=weak_opponent)
            self.opponent_pool = [basic_opponent]
        else:
            self.opponent_pool = opponent_pool
        
        self.max_transitions = max_transitions
        
        # Seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.env.seed(seed)
        
        # Agent
        if type(activation_function) == list:
            self.agent = DuelingDQNEnsembleAgent(gamma**forward_view_n ,
                self.env.observation_space.shape[0],
                feature_space_dim,
                segment_width,
                segment_depth,
                19,
                activation_function,
                loss_function,
                optimizer,
                learning_rate,
                optimization_steps,
                batch_size,
                device,
                experience_buffer_size
            )
        else:
            self.agent = DuelingDQNAgent(gamma**forward_view_n ,
                self.env.observation_space.shape[0],
                feature_space_dim,
                segment_width,
                segment_depth,
                19,
                activation_function,
                loss_function,
                optimizer,
                learning_rate,
                optimization_steps,
                batch_size,
                device,
                experience_buffer_size
            )

        # Optimization
        self.learning_rate = learning_rate
        self.learning_rate_halving_episodes = \
            learning_rate_halving_episodes
        self.gamma = gamma
        
        # Exploration
        self.eps_decay = eps_decay
        self.eps = epsilon0
        
        # Actions
        self.action_set = action_set

        # monitoring variables
        self.scores = []
        self.winners = []
        self.losses = []
        self.test_scores = []
        self.test_winners = []
        self.epsilons = []
        self.learning_rates = []
        self.transitions = []
        self.test_transitions = []

        self.add_self_episodes = add_self_episodes

        self.save_at_checkpoint = save_at_checkpoint

        self.forward_view_n = forward_view_n

        self.action_selector = action_selector

    
    def action_wrapper(self, action):
        return self.action_set[action]

    
    def run_episode(self):
        # reset environment
        state, _ = self.env.reset()
        state /= state_scaling
        
        # monitoring variables
        score = 0
        transitions = 0

        # select opponent
        opponent = np.random.choice(self.opponent_pool)
        
        # transition storage for forward view
        forward_view = []

        # run episode
        for t in range(self.max_transitions):
            # get action from agent with epsilon greedy policy
            action = self.action_selector(self.agent, 
                                          state, 
                                          19,
                                          self.eps)
            
            # transform action
            trans_action = self.action_wrapper(action.item())
            
            # get action from opponent
            action_opponent = opponent.act(self.env.obs_agent_two())

            # take step in environment
            next_state, reward, done, trunc, info = \
                    self.env.step(np.hstack([trans_action,
                                             action_opponent]))
            next_state /= state_scaling

            # store transition in forward view list
            forward_view.append((state,
                                action.item(),
                                reward,
                                next_state,
                                done))

            # calculate modified reward
            modified_reward = sum([self.gamma**i * t[2] 
                                   for i, t in 
                                   enumerate(forward_view)])
            
            # add forward view experience to experience buffer
            self.agent.experience_buffer.add_experience(
                    forward_view[0][0],
                    forward_view[0][1],
                    modified_reward,
                    next_state,
                    done)

            # remove oldest transition if list is too long
            if len(forward_view) == self.forward_view_n:
                forward_view.pop(0)

            state = next_state
            score += reward
            transitions += 1

            if done or trunc or t == self.max_transitions - 1:
                break
        
        # empty transition list
        while len(forward_view) > 0:
            modified_reward = sum([self.gamma**i * t[2] 
                                   for i, t in 
                                   enumerate(forward_view)])
            self.agent.experience_buffer.add_experience(
                    forward_view[0][0],
                    forward_view[0][1],
                    modified_reward,
                    forward_view[-1][3],
                    forward_view[-1][4])
            
            forward_view.pop(0)

        return score, info['winner'], transitions


    def train(self, num_episodes, 
                    checkpoint_interval,
                    q_target_update_interval,
                    num_testing_episodes,):

        print("~~~ Starting Training ~~~")

        # start timer
        start_time = time()
        last_checkpoint = start_time

        # training loop
        for i in range(num_episodes):
            score, winner, transitions = self.run_episode()

            # optimize agent
            losses = self.agent.train()

            # update monitoring variables
            self.scores.append(score)
            self.winners.append(winner)
            self.losses += losses
            self.transitions.append(transitions)

            # save and update epsilon
            self.epsilons.append(self.eps)
            self.eps = self.eps * self.eps_decay

            # add self to opponent pool
            if i in self.add_self_episodes:
                opponent = self.agent.copy()
                opponent.act = lambda state: self.action_wrapper(
                    opponent(state / state_scaling).item())
                self.opponent_pool.append(opponent)

            # update target network
            if i % q_target_update_interval == 0:
                self.agent.update_frozen_QFunction()
            
            # checkpointing
            if i % checkpoint_interval == 0:
                # test agent
                test_scores, test_winners, test_transitions = \
                    test_agent(self.agent,
                               self.env,
                               self.opponent_pool,
                               self.action_wrapper,
                               num_testing_episodes,
                               self.max_transitions)

                # update monitoring variables
                self.test_scores.append(test_scores)
                self.test_winners.append(test_winners)
                self.test_transitions.append(test_transitions)
                
                # statistics
                P_win = sum(np.array(test_winners) == 1) \
                        / num_testing_episodes
                dt = time() - last_checkpoint
                last_checkpoint = time()
                pred = (time() - start_time)/(i+1) \
                        *(num_episodes-i-1)/60 
                mean_trans = np.mean(test_transitions)
                score_train = np.mean(self.scores[-checkpoint_interval:])

                # log
                print(f'Ep.: {i:5.0f},',
                      f'Score: {score_train:6.2f},',
                      f'P_win: {P_win:.2f},',
                      f'Tran: {mean_trans:.2f},',
                      f'eps: {self.eps:.2f},',
                      f'lr: {self.learning_rate:.2e},',
                      f'pred: {pred:5.2f} min')

                # save model
                if self.save_at_checkpoint != None:
                    save_agent(self.agent,
                               self.save_at_checkpoint + f'_{i}')

            # update learning rate
            if i in self.learning_rate_halving_episodes:
                self.learning_rate /= 2
                self.agent.QFunction.update_learning_rate(
                    self.learning_rate)

        print("~~~ Training Complete ~~~")


    def save_raw_data(self, filename):
        data = {
            'scores': self.scores,
            'winners': self.winners,
            'losses': self.losses,
            'test_scores': self.test_scores,
            'test_winners': self.test_winners,
            'epsilons': self.epsilons,
            'learning_rates': self.learning_rates,
            'transitions': self.transitions,
            'test_transitions': self.test_transitions,
        }
        json.dump(data, open(filename, 'w'), indent=4)


