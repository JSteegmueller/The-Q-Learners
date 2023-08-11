import doctest
import torch
import torch.nn as nn   
import pickle

from QFunction import QFunction
from QFunction import QFunctionEnsemble
from ExperienceBuffer import ExperienceBuffer


class DuelingDQNAgent:
    """
    A Dueling Deep Q Network agent
    """
    def __init__(self, 
                 gamma, 
                 state_space_dim,
                 feature_space_dim, 
                 segment_width,
                 segment_depth,
                 action_space_dim, 
                 activation, 
                 loss, 
                 optimizer,
                 learning_rate,
                 optimization_steps,
                 batch_size,
                 device='cpu',
                 experience_buffer_size = 100_000,
                 ):
        """
        Initializes the agent

        Args:
            gamma: The discount factor
            state_space_dim: The dimension of the state space
            feature_space_dim: The dimension of the feature space
            segment_width: The width of the QFunction segments
            segment_depth: The depth of the QFunction segments
            action_space_dim: The dimension of the action space
            activation: The activation function
            loss: The loss function
            optimizer: The optimizer
            learning_rate: The learning rate
            optimization_steps: The number of optimization steps 
                                per train call
            batch_size: The batch size
            device: The device to run the agent on
        
        Usage:
            >>> import numpy as np
            >>> np.random.seed(0)
            >>> s = torch.manual_seed(0)
            >>> agent = DuelingDQNAgent(0.99, 4, 4, 4, 4, 2,
            ...                         nn.ReLU(),
            ...                         nn.MSELoss(),
            ...                         torch.optim.Adam,
            ...                         0.001,
            ...                         10,
            ...                         32,
            ...                         'cpu')
            >>> agent(np.random.rand(4))
            array([[0]])
        """
        
        # save activation, loss, optimizer, and learning rate
        self.activation = activation
        self.loss = loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.device = device

        # QFunctions
        self.QFunction = QFunction(state_space_dim,
                                   feature_space_dim,
                                   segment_width,
                                   segment_depth,
                                   action_space_dim,
                                   activation,
                                   loss,
                                   optimizer,
                                   learning_rate,
                                   device)

        self.frozen_QFunction = self.QFunction.copy()

        # Hyperparameters
        self.gamma = gamma
        self.optimization_steps = optimization_steps
        self.batch_size = batch_size

        # Spaces
        self.state_space_dim = state_space_dim
        self.action_space_dim = action_space_dim
        self.feature_space_dim = feature_space_dim
        self.segment_width = segment_width
        self.segment_depth = segment_depth

        # Experience
        self.experience_buffer = ExperienceBuffer(state_space_dim,
                                                  device,
                                                  experience_buffer_size)

        
    def __call__(self, state):
        """
        Returns the action that the agent takes
        """
        state = torch.from_numpy(state).float()
        state = torch.atleast_2d(state)
        return self.QFunction.argmax(state).numpy() 

    
    def update_frozen_QFunction(self):
        """
        Updates the frozen QFunction
        """
        self.frozen_QFunction = self.QFunction.copy()

    
    def train(self, 
              doubling=True,
              optimization_steps=None,
              batch_size=None):
        """
        Updates the QFunction based on the experience
        """
        if optimization_steps is None:
            optimization_steps = self.optimization_steps
        if batch_size is None:
            batch_size = self.batch_size

        # Train loop
        losses = []
        
        for states, actions, rewards, next_states, done in \
                self.experience_buffer(optimization_steps,
                                       batch_size):
            
            # Break if there is not enough experience
            if len(self.experience_buffer) < batch_size:
                break
            
            # Calculate the target
            if doubling:
                target = rewards + self.gamma * (1 - done) * \
                    self.frozen_QFunction(next_states,
                                          self.QFunction.argmax(
                                               next_states))
            else:
                target = rewards + self.gamma * (1 - done) * \
                    self.frozen_QFunction.argmax(next_states)
            losses.append(self.QFunction.update(states, actions, target))
        
        return losses

    # copy agent
    def copy(self):
        """
        Returns a copy of the agent
        """
        agent = DuelingDQNAgent(self.gamma,
                                self.state_space_dim,
                                self.feature_space_dim,
                                self.segment_width,
                                self.segment_depth,
                                self.action_space_dim,
                                self.activation,
                                self.loss,
                                self.optimizer,
                                self.learning_rate,
                                self.optimization_steps,
                                self.batch_size,
                                self.device)
        
        agent.QFunction = self.QFunction.copy()
        agent.frozen_QFunction = self.frozen_QFunction.copy()
        return agent

    
    # Loading and saving
    def save(self, path):
        """
        Saves the agent
        """
        self.QFunction.save(path)

    
    def load(self, path):
        """
        Loads the agent
        """
        self.QFunction.load(path)


class DuelingDQNEnsembleAgent(DuelingDQNAgent):
    """
    Dueling DQN Ensemble Agent
    """
    def __init__(self, 
                 gamma, 
                 state_space_dim,
                 feature_space_dim, 
                 segment_width,
                 segment_depth,
                 action_space_dim, 
                 activations, 
                 loss, 
                 optimizer,
                 learning_rate,
                 optimization_steps,
                 batch_size,
                 device='cpu',
                 experience_buffer_size = 100_000,
                 ):
        
        super().__init__(gamma, 
                         state_space_dim,
                         feature_space_dim, 
                         segment_width,
                         segment_depth,
                         action_space_dim, 
                         activations[0], 
                         loss, 
                         optimizer,
                         learning_rate,
                         optimization_steps,
                         batch_size,
                         device=device,
                         experience_buffer_size = 100_000)

        self.QFunction = QFunctionEnsemble(state_space_dim,
                                           feature_space_dim,
                                           segment_width,
                                           segment_depth,
                                           action_space_dim,
                                           activations,
                                           loss,
                                           optimizer,
                                           learning_rate,
                                           device)

        self.activations = activations

        self.frozen_QFunction = self.QFunction.copy()

    def optimistic_action(self, state):
        """
        Returns the optimistic action
        """
        state = torch.from_numpy(state).float()
        state = torch.atleast_2d(state)
        return self.QFunction.optimistic_action(state).numpy()


# Helper functions
def save_agent(agent, path):
    """
    Saves an agent

    Args:
        agent: The agent to save
        path: The path to save the agent to

    Usage:
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> s = torch.manual_seed(0)
        >>> agent = DuelingDQNAgent(0.99, 4, 4, 4, 4, 2,
        ...                         nn.ReLU(),
        ...                         nn.MSELoss(),
        ...                         torch.optim.Adam,
        ...                         0.001,
        ...                         10,
        ...                         32,
        ...                         'cpu')
        >>> input_ = np.random.rand(4)
        >>> agent(input_)
        array([[0]])
        >>> save_agent(agent, "agent")
        >>> agent.load("agent")
        >>> agent(input_)
        array([[0]])
        >>> import os # Cleanup
        >>> os.remove("agent.pt")
        >>> os.remove("agent_meta_data.pickle")
            
    """
    agent.save(path)
    if type(agent) == DuelingDQNAgent:
        meta_data = {"gamma": agent.gamma,
                     "state_space_dim": agent.state_space_dim,
                     "feature_space_dim": agent.feature_space_dim,
                     "segment_width": agent.segment_width,
                     "segment_depth": agent.segment_depth,
                     "action_space_dim": agent.action_space_dim,
                     "activation": agent.activation,
                     "loss": agent.loss,
                     "optimizer": agent.optimizer,
                     "learning_rate": agent.learning_rate,
                     "optimization_steps": agent.optimization_steps,
                     "batch_size": agent.batch_size,
                     "device": agent.device}
    elif type(agent) == DuelingDQNEnsembleAgent:
        meta_data = {"gamma": agent.gamma,
                     "state_space_dim": agent.state_space_dim,
                     "feature_space_dim": agent.feature_space_dim,
                     "segment_width": agent.segment_width,
                     "segment_depth": agent.segment_depth,
                     "action_space_dim": agent.action_space_dim,
                     "activations": agent.activations,
                     "loss": agent.loss,
                     "optimizer": agent.optimizer,
                     "learning_rate": agent.learning_rate,
                     "optimization_steps": agent.optimization_steps,
                     "batch_size": agent.batch_size,
                     "device": agent.device}

    with open(path + "_meta_data.pickle", "wb") as file:
        pickle.dump(meta_data, file)


def load_agent(path):
    """
    Loads an agent
    """
    with open(path + "_meta_data.pickle", "rb") as file:
        meta_data = pickle.load(file)

    if "activations" in meta_data.keys():
        agent = DuelingDQNEnsembleAgent(meta_data["gamma"],
                                        meta_data["state_space_dim"],
                                        meta_data["feature_space_dim"],
                                        meta_data["segment_width"],
                                        meta_data["segment_depth"],
                                        meta_data["action_space_dim"],
                                        meta_data["activations"],
                                        meta_data["loss"],
                                        meta_data["optimizer"],
                                        meta_data["learning_rate"],
                                        meta_data["optimization_steps"],
                                        meta_data["batch_size"],
                                        meta_data["device"])

    else:
        agent = DuelingDQNAgent(meta_data["gamma"],
                                meta_data["state_space_dim"],
                                meta_data["feature_space_dim"],
                                meta_data["segment_width"],
                                meta_data["segment_depth"],
                                meta_data["action_space_dim"],
                                meta_data["activation"],
                                meta_data["loss"],
                                meta_data["optimizer"],
                                meta_data["learning_rate"],
                                meta_data["optimization_steps"],
                                meta_data["batch_size"],
                                meta_data["device"])
    agent.load(path)
    return agent

doctest.testmod()
