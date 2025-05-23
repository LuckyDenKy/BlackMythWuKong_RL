import torch
from Model.BlackMythWuKongEnv import BlackMythWuKongEnv
from Model.Agent import PPO
from Model import model_utils

torch.manual_seed(42)

actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 10
hidden_dim = 128
gamma = 0.98
lmbda = 0.95
epochs = 10
eps = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = BlackMythWuKongEnv(device=device)
state_dim = env.state_dim
action_dim = env.action_dim
agent = PPO(state_dim,hidden_dim,action_dim,actor_lr,critic_lr,lmbda,epochs, eps, gamma, device)
return_list = model_utils.train_on_policy_agent(env,agent,num_episodes)
print(return_list)