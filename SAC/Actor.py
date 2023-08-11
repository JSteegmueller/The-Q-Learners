### This is the network for Actor / Policy
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import torch.nn as nn

class Actor(torch.nn.Module):
	def __init__(self, input_size, hidden_size , output_size,device="cpu", noise=1e-6, action_scale=1.,action_bias=0. ,lr = 0.0002):
		super(Actor, self).__init__()
		self.network = nn.Sequential(
			nn.Linear(input_size,hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size,hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size,hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size,hidden_size),
			nn.ReLU()
			)
		self.action_scale = action_scale
		self.action_bias  = action_bias
		self.mean = nn.Linear(hidden_size,output_size)
		self.log_std = nn.Linear(hidden_size,output_size)
		self.noise = noise


	def forward(self, state):
		""" forward
		args:
			state torch.Tensor, BxN
		return 
			mean torch.Tensor, BxA
			log_std torch.Tensor, BxA
		"""
		x = self.network(state)
		mean = self.mean(x)
		log_std = self.log_std(x)
		log_std = torch.clamp(log_std, min=-20, max=10)
		return mean, log_std

	def predict(self, state):
		""" forward
		args:
			state torch.Tensor, BxN
		return 
			action torch.Tensor, BxA
			log_prob torch.Tensor, Bx1
			mean torch.Tensor, BxA
		"""
		mean, log_std = self.forward(state)
		std = log_std.exp()
		normal = Normal(mean,std)
		
		# Reparametrization
		x = normal.rsample()
		y = torch.tanh(x)

		# re-scale
		action = y * self.action_scale + self.action_bias

		log_prob = normal.log_prob(x)
		log_prob -= torch.log(self.action_scale * (1 - y.pow(2)) + self.noise)
		
		try:
			log_prob = log_prob.sum(axis=1, keepdim=True)
			
		except:
			pass
			
		mean = torch.tanh(mean) * self.action_scale + self.action_bias
		#print("in actor prediction")
		#print("log prob size", log_prob.shape)
		#print("action size", action.shape)
		return action, log_prob, mean






