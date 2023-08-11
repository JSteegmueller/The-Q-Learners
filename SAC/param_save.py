import torch
def save_param(agent,folder,itr):
	Critic = agent.Critic
	Critic_target = agent.Critic_target
	Actor = agent.Actor
	Alpha = agent.alpha

	#np.save("Critic Param",agent.Actor.state_dict())
	f = "./" + folder + "/"
	torch.save( Actor , f+"Actor_Param"+itr)
	torch.save( Critic, f+"Critic_Param"+itr)
	torch.save( Critic_target, f+"Critic_target_Param"+itr)
	torch.save( Alpha, f+"alpha"+itr)

	return

def load_param(agent,folder,itr):
	f = "./" + folder + "/"
	A = torch.load(f+"Actor_Param"+itr)
	C = torch.load(f+"Critic_Param"+itr)
	C_t = torch.load(f+"Critic_target_Param"+itr)
	alpha = torch.load(f+"alpha"+itr)

	agent.Actor=A
	agent.Critic=C
	agent.Critic_target=C_t
	agent.alpha = alpha