import torch
import torch.nn.functional as F
import numpy as np
from Actor import Actor
from Critic import Critic
import memory as mem

class SAC(object):
	def __init__(self, observation_space, action_space, **userconfig ):
		self._config = {
            "eps": 0.3,            # Epsilon: noise strength to add to policy
            "discount": 0.99,
            "buffer_size": int(1e6),
            "batch_size": 256,
            "learning_rate_actor": float(3e-4), ##It were : 0.00001
            "learning_rate_critic": float(1e-3), ##It were: 0.0001
            "hidden_sizes_actor": 256,
            "hidden_sizes_critic": 256,
            "update_target_every": 1,
            "alpha": 0.2,
            "use_target_net": True,
            "alpha_tuning":True,
            'DR3_term':0.00
        }

		self.observ_dim = observation_space.shape[0]
		if action_space.shape[0] == 4:
			self.action_dim = 4
		else:
			self.action_dim = int(action_space.shape[0] /2)
		self.alpha = 0.2
		if self._config["alpha_tuning"]:
			#milestones = [int(x) for x in (self._config['alpha_milestones'][0]).split(' ')]
			self.target_entropy = -torch.Tensor(self.action_dim)#.to(self.device)
			#self.target_entropy = -torch.tensor(4)#.to(self.device)
			self.log_alpha = torch.zeros(1, requires_grad=True)#, device=self.device)
			self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=0.0001)
			#self.alpha_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            #    self.alpha_optim, milestones=milestones, gamma=0.5
            #)
		try:
			self.low = action_space.low
			self.high = action_space.high
		except:
			n = action_space.n
			self.low = 0
			self.high = n

		self.buffer = mem.Memory(max_size=self._config["buffer_size"])
		try:
			self.scale = torch.FloatTensor( (self.high - self.low) / 2.)[:4]
			self.bias  = torch.FloatTensor((self.high + self.low) / 2.)[:4]
		except:
			self.scale = 1 
			self.bias  = 0

		### Q network
		self.Critic = Critic(self.observ_dim, self.action_dim, 
			hidden_size=self._config['hidden_sizes_critic'],output_size=1,device="cpu",loss="L2",
			lr = 0.0002,rho=0.995)

		self.Critic_target = Critic(self.observ_dim, self.action_dim, 
			hidden_size=self._config['hidden_sizes_critic'],output_size=1,device="cpu",loss="L2",
			lr = 0.0002,rho=0.995)

		self.Critic_optim = torch.optim.Adam(self.Critic.parameters(), 
			lr=self._config['learning_rate_critic'],eps=1e-6)

		### policy
		self.Actor = Actor(input_size=self.observ_dim, 
			hidden_size=self._config['hidden_sizes_actor'], output_size=self.action_dim,
			device="cpu", noise=1e-6, 
			action_scale=self.scale, action_bias=self.bias, lr = self._config["learning_rate_actor"])
		
		self.Actor_optim = torch.optim.Adam(self.Actor.parameters(), 
			lr=self._config['learning_rate_actor'],eps=1e-6)


		self.train_iter = 0
		self.copy()

	def act(self,state,evaluate=False):
		"""
		args:
			state, numpy
		return:
			action, numpy
		"""
		#state = torch.from_numpy(state).float()
		state = torch.Tensor(state)

		action, log_prob , mean = self.Actor.predict(state)

		if evaluate:
			a = mean
		else:
			a = action 

		return a.detach().numpy(), log_prob.sum().detach().numpy().item() * self._config["alpha"]

		

	def store_transition(self, transition):
		self.buffer.add_transition(transition)
		### add reward for exploration.

	def copy(self):
		self.Critic_target.load_state_dict(self.Critic.state_dict())

	def updateQtarget(self):
		self.Critic_target.update(self.Critic)
	

	def train(self,iter_fit=32):
		to_torch = lambda x: torch.from_numpy(x.astype(np.float32))
		losses = []
		Q_value = []
		self.train_iter+=1

		for i in range(iter_fit):
			data=self.buffer.sample(batch=self._config['batch_size'])
			state = to_torch(np.stack(data[:,0])) # s_t
			action = to_torch(np.stack(data[:,1])) # a_t
			reward = to_torch(np.stack(data[:,2])[:,None]) # rew  (batchsize,1)
			next_state = to_torch(np.stack(data[:,3])) # s_t+1
			done = to_torch(np.stack(data[:,4])[:,None]) # done signal  (batchsize,1)

			with torch.no_grad():
				next_action, next_logprob, _ = self.Actor.predict(next_state)
				q1, q2 = self.Critic_target(next_state,next_action)
				q = torch.minimum(q1,q2)  - self.alpha * next_logprob
				Q_value.append(q)
				#store
				gamma = self._config['discount']
				td_target = reward + gamma * (1.0-done) * q
				td_target = td_target.squeeze()

			### Critic update
			Q1, Q2 = self.Critic(state,action)
			Q1_loss = self.Critic.loss(Q1.squeeze(), td_target)
			Q2_loss = self.Critic.loss(Q2.squeeze(), td_target)
			Q_loss = Q1_loss + Q2_loss 

			self.Critic_optim.zero_grad()
			Q_loss.backward()
			self.Critic_optim.step()

			### Critic target update
			if i % self._config["update_target_every"] == 0:
				self.updateQtarget()

			### Actor update
			new_action, log_prob, _ = self.Actor.predict(state)
			Q_A1, Q_A2 = self.Critic(state, new_action)
			Q_A = torch.min(Q_A1,Q_A2)
			policy_loss = (self.alpha * log_prob - Q_A).mean(axis=0)

			self.Actor_optim.zero_grad()
			policy_loss.backward()
			self.Actor_optim.step()
			losses.append((Q_loss.item(), policy_loss.item()))

			### Alpha update
			if self._config["alpha_tuning"]:
				alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
				self.alpha_optim.zero_grad()
				alpha_loss.backward()
				self.alpha_optim.step()
				#self.alpha_scheduler.step()
				self.alpha = self.log_alpha.exp()

		return losses






            
