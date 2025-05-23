import os
import torch
from torch import nn
import torch.nn.functional as F

try:
    import model_utils
except:
    from Model import model_utils

class PolicyNet(nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,action_dim)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x),dim=1)

class ValueNet(nn.Module):
    def __init__(self,state_dim,hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,1)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class PPO:
    def __init__(self,state_dim,hidden_dim,action_dim,actor_lr,critic_lr,
                 lmbda,epochs,eps,gamma,device):
        self.actor = PolicyNet(state_dim,hidden_dim,action_dim).to(device)
        self.critic = ValueNet(state_dim,hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=critic_lr)
        self.load_ckpt()

        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO截断范围的参数
        self.device = device

    def load_ckpt(self):
        if os.path.exists('ckpt/last_actor_ckpt.pth'):
            ckpt = torch.load('ckpt/last_actor_ckpt.pth')
            self.actor.load_state_dict(ckpt)
            # self.actor_optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if os.path.exists('ckpt/last_critic_ckpt.pth'):
            ckpt = torch.load('ckpt/last_critic_ckpt.pth')
            self.critic.load_state_dict(ckpt)
            # self.critic_optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    def save_ckpt(self):
        # torch.save(
        #     {'model_state_dict': self.actor.state_dict(), 'optimizer_state_dict': self.actor_optimizer.state_dict()},
        #     'ckpt/last_actor_ckpt.pth')
        # torch.save({'model_state_dict': self.critic.state_dict(), 'optimizer_state_dict': self.critic.state_dict()},
        #            'ckpt/last_critic_ckpt.pth')
        torch.save(self.actor.state_dict(),'ckpt/last_actor_ckpt.pth')
        torch.save(self.critic.state_dict(),'ckpt/last_critic_ckpt.pth')

    def take_action(self,state):
        assert type(state) == torch.Tensor
        probs = self.actor(state.unsqueeze(0))
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self,transition_dict):
        states = torch.stack(transition_dict['states'],dim=0).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1,1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1,1).to(self.device)
        next_states = torch.stack(transition_dict['next_states'],dim=0).to(self.device)
        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1,1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1-dones)
        td_delta = td_target - self.critic(states)
        advantage = model_utils.compute_advantage(self.gamma,self.lmbda,td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,actions)).detach()

        # 创建两套 states（分别用于 actor 和 critic）
        states_actor = states.detach().clone()  # 用于 actor
        states_critic = states.detach().clone()  # 用于 critic

        # for _ in range(self.epochs):
        #     log_probs = torch.log(self.actor(states).gather(1,actions))
        #     ratio = torch.exp(log_probs-old_log_probs)
        #     surr1 = ratio * advantage
        #     surr2 = torch.clamp(ratio,1-self.eps,1+self.eps)*advantage # 截断
        #     actor_loss = torch.mean(-torch.min(surr1,surr2))  # PPO损失
        #     critic_loss = torch.mean(F.mse_loss(self.critic(states),td_target.detach()))
        #     self.actor_optimizer.zero_grad()
        #     self.critic_optimizer.zero_grad()
        #     actor_loss.backward()
        #     critic_loss.backward()
        #     self.actor_optimizer.step()
        #     self.critic_optimizer.step()

        for _ in range(self.epochs):
            # 计算 actor_loss（仅依赖 states_actor）
            log_probs = torch.log(self.actor(states_actor).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失

            # 计算 critic_loss（仅依赖 states_critic）
            critic_loss = torch.mean(F.mse_loss(self.critic(states_critic), td_target.detach()))

            # 分别优化 actor 和 critic
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()  # 仅计算 actor 的梯度
            critic_loss.backward()  # 仅计算 critic 的梯度
            self.actor_optimizer.step()
            self.critic_optimizer.step()

