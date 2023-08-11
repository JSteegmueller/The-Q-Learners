import torch
from torch import from_numpy as fnp
import doctest

class ExperienceBuffer:
    """
    This class stores the experiences of the agent in a buffer
    We use a tensor to store the experiences and have following format:
    [state, action, reward, next_state, done]


    Usage:
    >>> import numpy as np
    >>> import torch
    >>> seed = torch.manual_seed(42)
    >>> # initialize buffer for 2d state space
    >>> buffer = ExperienceBuffer(2) 
    >>> # Add two experiences to the buffer
    >>> state_1, next_state_1 = np.array([1, 2]), np.array([3, 4])
    >>> buffer.add_experience(state_1, 1, 1, next_state_1, 0)
    >>> state_2, next_state_2 = np.array([5, 6]), np.array([7, 8])
    >>> buffer.add_experience(state_2, 2, 2, next_state_2, 1)
    >>> # Sample a batch of size 2
    >>> states, actions, rewards, next_states, done = buffer.sample(2)
    >>> # Check if the batch is correct
    >>> states
    tensor([[1., 2.],
            [5., 6.]])
    >>> actions
    tensor([[1],
            [2]])
    >>> rewards
    tensor([[1.],
            [2.]])
    >>> next_states
    tensor([[3., 4.],
            [7., 8.]])
    >>> done
    tensor([[0.],
            [1.]])

    """
    def __init__(self, state_space_dim, device="cpu", size=100_000):
        """
        Initializes the experience buffer
        """

        # This experience buffer stores the experiences in a tensor
        self.size = size
        self.experiences = torch.zeros(size,
                                       2 * state_space_dim + 3,
                                       requires_grad=False).to(device)

        # scalars for write index and allready filled experiences
        self.filled_until = 0
        self.write_idx = 0
        

        self.state_space_dim = state_space_dim
        
        self.device = device
   

    def add_experience(self, state, action, reward, next_state, done):
        """
        Adds the experience to the experience buffer
        """
        
        # short hand for state space dimension
        sd = self.state_space_dim

        # Add the experience to the buffer
        self.experiences[self.write_idx, 0:sd] = fnp(state)
        self.experiences[self.write_idx, sd] = action
        self.experiences[self.write_idx, sd+1] = reward
        self.experiences[self.write_idx, sd+2:2*sd+2] = fnp(next_state)
        self.experiences[self.write_idx, 2*sd+2] = done
        
        # Update the write index
        self.write_idx = (self.write_idx + 1) % self.size

        # Update the filled until counter
        self.filled_until = min(self.filled_until + 1,
                                self.size)


    def sample(self, batch_size):
        """
        Samples a batch of experiences from the experience buffer.

        """
        # Make sure that the batch size is not larger than the buffer
        batch_size = min(batch_size, self.filled_until)

        # Sample indices with multinomial distribution
        probabilites = torch.ones(self.filled_until) \
                / self.filled_until
        indices = torch.multinomial(probabilites, batch_size)
        
        # Make the batch
        batch = self.experiences[indices]
        
        # Short hand for state space dimension
        sd = self.state_space_dim
        
        # Unpack the batch 
        states = batch[:, 0:sd]
        actions = batch[:, sd: sd + 1].long() # Long for indexing
        rewards = batch[:, sd + 1: sd + 2]
        next_states = batch[:, sd + 2: 2 * sd + 2]
        done = batch[:, 2 * sd + 2:2 * sd + 3]
        
        # Return the unpacked batch
        return states, actions, rewards, next_states, done

    
    def __call__(self,  num_steps, batch_size):
        """
        returns an iterator over experience batches
        """
        for i in range(num_steps):
            yield self.sample(batch_size)


    def __len__(self):
        """
        Returns the number of experiences in the buffer
        """
        return self.filled_until

doctest.testmod()
