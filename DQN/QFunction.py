import torch
import torch.nn as nn
import doctest

class QNetwork(torch.nn.Module):
    """
    Neural network based on the dueling architecture
    """
    def __init__(self,
                 input_dim,
                 f_space_dim,
                 f_seg_width,
                 f_seg_depth,
                 v_seg_width,
                 v_seg_depth,
                 a_seg_width,
                 a_seg_depth,
                 output_dim,
                 activation,
                 use_mean=True, #vs max
                 device="cpu"):
        """
        Args:
            input_dim: The dimension of the input
            f_seg_width: The width of the feature segment
            f_seg_depth: The depth of the feature segment
            f_space_dim: The dimension of the feature space
            v_seg_width: The width of the value segment
            v_seg_depth: The depth of the value segment
            a_seg_width: The width of the advantage segment
            a_seg_depth: The depth of the advantage segment
            output_dim: The dimension of the output
            activation: The activation function
            use_mean: Whether to use the mean or max in the 
                      dueling architecture
            device: The device to use

        Usage:
            >>> s = torch.manual_seed(0)
            >>> import torch.nn as nn
            >>> net = QNetwork(4, 32, 2, 32, 32, 2, 32, 2, 2,
            ...                nn.ReLU()) 
            >>> net(torch.randn(1, 4))
            tensor([[-0.0688,  0.0688]], grad_fn=<SubBackward0>)
        """
        
        # torch stuff
        super(QNetwork, self).__init__()
        self.device = torch.device(device)
        
        # save the parameters
        self.input_dim = input_dim
        self.feature_space_dim = f_space_dim
        self.feature_seg_width = f_seg_width
        self.feature_seg_depth = f_seg_depth
        self.value_seg_width = v_seg_width
        self.value_seg_depth = v_seg_depth
        self.advantage_seg_width = a_seg_width
        self.advantage_seg_depth = a_seg_depth
        self.output_dim = output_dim
        self.activation = activation
        self.use_mean = use_mean
        
        # create the segments

        # feature segment
        self.feature_seg = self._create_seg(input_dim,
                                            f_space_dim,
                                            f_seg_width,
                                            f_seg_depth,
                                            activation,
                                            True)
        
        self.advantage_seg = self._create_seg(f_space_dim,
                                              output_dim,
                                              a_seg_width,
                                              a_seg_depth,
                                              activation)

        self.value_seg = self._create_seg(f_space_dim,
                                          1,
                                          v_seg_width,
                                          v_seg_depth,
                                          activation)

    
    def _create_seg(self,
                    input_size,
                    output_size,
                    width,
                    depth,
                    activation,
                    last_layer_has_activation = False):
        """
        Create a segment of the network
        Args:
            input_size: The size of the input
            output_size: The size of the output
            width: The width of the segment
            depth: The depth of the segment
            activation: The activation function
        """
        # create the layers
        layer_widths = [input_size] + \
            [width for _ in range(depth - 1)] + \
            [output_size]

        return nn.Sequential(
            *[nn.Sequential(nn.Linear(layer_widths[i], 
                                      layer_widths[i + 1]),
                            activation if i < depth - 1 or 
                            last_layer_has_activation else nn.Identity())
              for i in range(depth)]
            )


    def forward(self, x):
        """
        Forward pass through the network
        Args:
            x: The input, shape (batch_size, input_size)
        """
        feature_space = self.feature_seg(x)
        value = self.value_seg(feature_space)
        advantage = self.advantage_seg(feature_space)

        # combine the advantage and value
        if self.use_mean:
            return value + advantage - \
                advantage.mean(dim=1, keepdim=True)
        
        # max
        return value + advantage - \
            torch.max(advantage, dim=1, keepdim=True)[0]


class SymmetricQNetwork(QNetwork):
    """
    QNetwork where all segments have the same length and depth
    """
    def __init__(self,
                 input_dim,
                 f_space_dim,
                 segment_width,
                 segment_depth,
                 output_dim,
                 activation,
                 use_mean=True, #vs max
                 device="cpu"):
        """
        Args:
            input_dim: The dimension of the input
            feature_space_dim: The dimension of the feature space
            segment_width: The width of the segments
            segment_depth: The depth of the segments
            output_dim: The dimension of the output
            activation: The activation function
            use_mean: Whether to use the mean or max in the 
                      dueling architecture
            device: The device to use

        Usage:
            >>> s = torch.manual_seed(0)
            >>> import torch.nn as nn
            >>> net = SymmetricQNetwork(4, 32, 2, 2, 2, nn.ReLU())
            >>> net(torch.randn(1, 4))
            tensor([[0.8541, 0.1668]], grad_fn=<SubBackward0>)
        """
        super(SymmetricQNetwork, self).__init__(input_dim,
                                                f_space_dim,
                                                segment_width,
                                                1,
                                                segment_width,
                                                segment_depth,
                                                segment_width,
                                                segment_depth,
                                                output_dim,
                                                activation,
                                                use_mean,
                                                device)

        self.segment_width = segment_width
        self.segment_depth = segment_depth


class QFunction:
    """
    Wrapper for a symmetric Q-Network that provides 
    additional functionality
    """
    def __init__(self,
                 input_dim,
                 feature_space_dim,
                 segment_width,
                 segment_depth,
                 output_dim, 
                 activation,
                 loss,
                 optimizer,
                 learning_rate,
                 device='cpu',
                 ):
        """
        Args:
            input_dim: The dimension of the input
            feature_space_dim: The dimension of the feature space
            segment_width: The width of the segments
            segment_depth: The depth of the segments
            output_dim: The dimension of the output
            activation: The activation function
            loss: The loss function
            optimizer: The optimizer
            learning_rate: The learning rate
            device: The device to use

        Usage:
            >>> s = torch.manual_seed(0)
            >>> import torch.nn as nn
            >>> q = QFunction(4, 32, 2, 2, 2, nn.ReLU(), nn.MSELoss(),
            ...               torch.optim.Adam, 1e-3)
            >>> q(torch.randn(1, 4), torch.tensor([[0]]))
            tensor([[0.8541]])
            >>> q.argmax(torch.randn(3, 4))
            tensor([[0],
                    [0],
                    [0]])
        """
        # save the parameters
        self.device = torch.device(device)
        self.optimizer_class = optimizer
        self.learning_rate = learning_rate
        self.loss = loss
         
        self.network = SymmetricQNetwork(input_dim,
                                         feature_space_dim,
                                         segment_width,
                                         segment_depth,
                                         output_dim,
                                         activation).to(self.device)
        
        self.optimizer = optimizer(self.network.parameters(),
                                   lr=learning_rate)


    def __call__(self,
                 states,
                 actions,
                 gradient=False):
        """
        Evaluate the Q-function for the provided state-action pairs
        Args:
            states: The states, shape (batch_size, state_size)
            actions: The actions, shape (batch_size, 1)
        """
        # for evaluation
        if not gradient:
            with torch.no_grad():
                return torch.gather(self.network(states), 1, actions)
        
        # for training
        return torch.gather(self.network(states), 1, actions)

    
    def argmax(self, states):
        return self.network(states).argmax(dim=1, keepdim=True)
    
    
    def update(self, states, actions, targets):
        """
        Update the network using the provided states, actions, and targets
        """
        self.optimizer.zero_grad()
        
        predictions = self(states, actions, gradient=True)
        loss = self.loss(predictions, targets)
        
        loss.backward()
        self.optimizer.step()

        return loss.item()


    def copy(self):
        """
        Return a copy of the Q-function
        """

        copy = QFunction(self.network.input_dim,  
                         self.network.feature_space_dim,
                         self.network.segment_width,
                         self.network.segment_depth,
                         self.network.output_dim,
                         self.network.activation,
                         self.loss,
                         self.optimizer_class,
                         self.learning_rate,
                         self.device)

        copy.network.load_state_dict(self.network.state_dict().copy())
        
        return copy

    
    # Loading and saving
    def save(self, path):
        """
        Save the network to the provided path
        """
        torch.save(self.network.state_dict(), path + '.pt')

    
    def load(self, path):
        """
        Load the network from the provided path
        """
        self.network.load_state_dict(torch.load(path + '.pt'))

    
    # Learning rate
    def update_learning_rate(self, learning_rate):
        """
        Update the learning rate
        """
        if self.learning_rate == learning_rate:
            return

        self.learning_rate = learning_rate
        self.optimizer = self.optimizer_class(
            self.network.parameters(),
            lr=learning_rate)


class QFunctionEnsemble:
    """
    Provides an interface to interact with a Q function ensemble
    """
    def __init__(self,
                 input_dim,
                 feature_space_dim,
                 segment_width,
                 segment_depth,
                 output_dim, 
                 activations, # Here a list is expected. Deterimines ensemble size
                 loss,
                 optimizer,
                 learning_rate,
                 device='cpu',
                 ):
        """
        Args:
            input_dim: The dimension of the input
            feature_space_dim: The dimension of the feature space
            segment_width: The width of the segments
            segment_depth: The depth of the segments
            output_dim: The dimension of the output
            activations: List of activation functions for each network
            loss: The loss function
            optimizer: The optimizer
            learning_rate: The learning rate
            ensemble_size: The size of the ensemble
            device: The device to use

        Usage:
            >>> s = torch.manual_seed(0)
            >>> import torch.nn as nn
            >>> q = QFunctionEnsemble(4, 32, 2, 2, 2, [nn.ReLU(), nn.ReLU()], nn.MSELoss(),
            ...                       torch.optim.Adam, 1e-3)
            >>> q(torch.randn(1, 4), torch.tensor([[0]]))
            [tensor([[-0.6312]]), tensor([[0.3810]])] 
            >>> q.argmax(torch.randn(3, 4))
            [tensor([[0],
                     [0],
                     [0]]), tensor([[0],
                                    [0],
                                    [0]])]
            >>> q.update(torch.randn(1, 4), torch.tensor([[0]]), torch.randn(1, 1))
            [0.000123, 0.000123]
            >>> q.majority_vote(torch.randn(3, 4))
            tensor([[0],
                    [0],
                    [0]])
            >>> q.optimistic_action(torch.randn(3, 4))
            tensor([[0],
                    [0],
                    [1]])
        """
        # save the parameters
        self.device = torch.device(device)
        self.optimizer_class = optimizer
        self.learning_rate = learning_rate
        self.loss = loss
        self.ensemble_size = len(activations)
        
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.networks = [SymmetricQNetwork(input_dim,
                                           feature_space_dim,
                                           segment_width,
                                           segment_depth,
                                           output_dim,
                                           activation).to(self.device)
                         for activation in activations]
                                           

        self.optimizers = [optimizer(network.parameters(),
                                     lr=learning_rate)
                           for network in self.networks]
    
    def __call__(self,
                 states,
                 actions,
                 gradient=False):
        """
        This returns the mean of the ensemble evaluation
        Args:
            states: The states, shape (batch_size, state_size)
            actions: The actions, shape (batch_size, 1)
        """
        ensemble_evaluation = self.evaluate_ensemble(states,
                                                    actions,
                                                    gradient=gradient)
        if not gradient:
            with torch.no_grad():
                return torch.mean(torch.stack(ensemble_evaluation), dim=0)
        torch.mean(torch.stack(ensemble_evaluation), dim=0)


    def evaluate_ensemble(self,
                          states,
                          actions,
                          gradient=False):
        """
        Evaluate the Q-function ensemble for the provided state-action pairs
        """
        # for evaluation
        if not gradient:
            with torch.no_grad():
                return [torch.gather(network(states), 1, actions)
                        for network in self.networks]
        
        # for training
        return [torch.gather(network(states), 1, actions)
                for network in self.networks]


    def update(self, states, actions, targets):
        """
        Update the ensemble using the provided states, actions, and targets
        """
        predictions = self.evaluate_ensemble(states, actions, gradient=True)
        for optimizer, prediction in zip(self.optimizers,
                                         predictions):
            optimizer.zero_grad()
            loss = self.loss(prediction, targets)
            loss.backward()
            optimizer.step()

        return loss.item()
    
    def argmax(self, states):
        """
        This returns the majority vote as we use this in place of the argmax
        """
        return self.majority_vote(states)

    
    def ensemble_argmax(self, states):
        """
        Return the argmax of the Q-function ensemble for the provided states
        """
        return [network(states).argmax(dim=1, keepdim=True) for 
                network in self.networks]
    
    def majority_vote(self, states):
        """
        Return the action with the majority vote
        """
        # voting
        batch_size = states.shape[0]
        votes = torch.zeros((batch_size, self.output_dim))
        for vote in self.ensemble_argmax(states):
            votes[torch.arange(batch_size), vote.flatten()] += 1
        
        ## randomize over winners
        # get the candidates
        max_votes = torch.max(votes, axis=1)[0][:, None]
        where = torch.argwhere(votes == max_votes)
        candidates = [[] for _ in range(batch_size)]
        for index in where:
            candidates[index[0].item()].append(index[1].item())
        
        # choose randomly
        choices = []
        for indices in candidates:
            choice = torch.randint(len(indices), (1, )).item()
            choices.append(indices[choice])

        return torch.tensor(choices)[:, None]


    def optimistic_action(self, states):
        """
        Get the optimistic action
        """
        batch_size = states.shape[0]
        predictions = torch.stack([network(states) for 
                                   network in self.networks])
        max_predictions = torch.max(predictions, dim=0)[0]
        return torch.argmax(max_predictions, dim=1).reshape(-1, 1)

    
    def copy(self):
        """
        Return a copy of the Q-function ensemble
        """
        copy = QFunctionEnsemble(self.networks[0].input_dim,  
                                 self.networks[0].feature_space_dim,
                                 self.networks[0].segment_width,
                                 self.networks[0].segment_depth,
                                 self.networks[0].output_dim,
                                 [network.activation for network in self.networks],
                                 self.loss,
                                 self.optimizer_class,
                                 self.learning_rate,
                                 self.device)

        for network, copy_network in zip(self.networks, copy.networks):
            copy_network.load_state_dict(network.state_dict().copy())

        return copy

    
    def save(self, path):
        """
        Save the Q-function ensemble to the provided path
        """
        for i, network in enumerate(self.networks):
            torch.save(network.state_dict(), path + str(i) + '.pt')

    def load(self, path):
        """
        Load the Q-function ensemble from the provided path
        """
        for i, network in enumerate(self.networks):
            network.load_state_dict(torch.load(path + str(i) + '.pt'))

    def update_learning_rate(self, learning_rate):
        """
        Update the learning rate of the Q-function ensemble
        """
        if self.learning_rate == learning_rate:
            return

        self.learning_rate = learning_rate
        self.optimizers = [self.optimizer_class(
            network.parameters(),
            lr=learning_rate) for network in self.networks]


# Helper functions
def load_QFunction(path,
                   input_dim,
                   feature_space_dim,
                   segment_width,
                   segment_depth,
                   output_dim,
                   activation,
                   loss,
                   optimizer,
                   learning_rate,
                   device='cpu'):

    """
    Load a Q-function from the provided path

    Usage:
        >>> s = torch.manual_seed(0)
        >>> import torch.nn as nn
        >>> q = QFunction(4, 32, 2, 2, 2, nn.ReLU(), nn.MSELoss(),
        ...               torch.optim.Adam, 1e-3)
        >>> test = torch.randn(1, 4)
        >>> q(test, torch.tensor([[0]]))
        tensor([[0.8541]])
        >>> q.save('test')
        >>> q_2 = load_QFunction('test', 4, 32, 2, 2, 2, nn.ReLU(),
        ...                      nn.MSELoss(), torch.optim.Adam, 1e-3)
        >>> q_2(test, torch.tensor([[0]]))
        tensor([[0.8541]])
        >>> import os # cleanup
        >>> os.remove('test.pt')
    """
    q_function = QFunction(input_dim,
                           feature_space_dim,
                           segment_width,
                           segment_depth,
                           output_dim,
                           activation,
                           loss,
                           optimizer,
                           learning_rate,
                           device)

    q_function.load(path)
    return q_function


def load_QFunctionEnsemble(path,
                           input_dim,
                           feature_space_dim,
                           segment_width,
                           segment_depth,
                           output_dim,
                           activations,
                           loss,
                           optimizer,
                           learning_rate,
                           device='cpu'):
    """
    Load a Q-function ensemble from the provided path

    Usage:
        >>> s = torch.manual_seed(0)
        >>> import torch.nn as nn
        >>> q = QFunctionEnsemble(4, 32, 2, 2, 2, [nn.ReLU(), nn.ReLU()],
        ...                       nn.MSELoss(), torch.optim.Adam, 1e-3)
        >>> test = torch.randn(1, 4)
        >>> q(test, torch.tensor([[0]]))
        tensor([[0]])
        >>> q.save('test')
        >>> q_2 = load_QFunctionEnsemble('test', 4, 32, 2, 2, 2,
        ...                              [nn.ReLU(), nn.ReLU()],
        ...                              nn.MSELoss(), torch.optim.Adam, 1e-3)
        >>> q_2(test, torch.tensor([[0]]))
        tensor([[0]])
        >>> import os # cleanup
        >>> os.remove('test0.pt')
        >>> os.remove('test1.pt')
    """
    q_function_ensemble = QFunctionEnsemble(input_dim,
                                            feature_space_dim,
                                            segment_width,
                                            segment_depth,
                                            output_dim,
                                            activations,
                                            loss,
                                            optimizer,
                                            learning_rate,
                                            device)

    q_function_ensemble.load(path)
    return q_function_ensemble

# Testing
doctest.testmod()
