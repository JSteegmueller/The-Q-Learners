### Network for critic / QFunction
import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_size=128,output_size=1,device="cpu",loss="L2",lr = 0.0002,rho=0.99):
        super(Critic, self).__init__()

        if loss == "L2":
            self.loss = nn.MSELoss()
        elif loss == "L1":
            self.loss = nn.SmoothL1Loss(reduction='mean')
        else:
            self.loss = nn.MSELoss()
        self.rho = rho
        self.Q1 = nn.Sequential(
            nn.Linear(observation_dim+action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,output_size)
            )
        #self.Q1_last = nn.Linear(hidden_size,output_size)

        self.Q2 = nn.Sequential(
            nn.Linear(observation_dim+action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,output_size)
            )
        #self.Q2_last = nn.Linear(hidden_size,output_size)

    def forward(self,observation,action):
        """
        args:
            observation: torch.Tensor
            action: torch.Tensor
    	return 
    		q1: torch.Tensor
    		q1: torch.Tensor
    	"""
        x = torch.cat((observation,action),dim=1)
        #y1 = self.Q1_last(F.relu(self.Q1(x)))
        #y2 = self.Q2_last(F.relu(self.Q2(x)))
        return self.Q1(x),self.Q2(x)
    	
    def update(self, Q):
        rho = self.rho
        target_dict = self.state_dict()
        net_dict    = Q.state_dict()
        for key in target_dict.keys():
            target_dict[key] = (
                rho * target_dict[key] + (1-rho) * net_dict[key])
        self.load_state_dict(target_dict)