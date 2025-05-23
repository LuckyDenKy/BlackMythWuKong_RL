import torch
from Model.Environment import BlackMythWuKongEnv
from Model.Agent import PPO
from Model import model_utils
import time
from tqdm import tqdm
import numpy as np

torch.manual_seed(42)

actor_lr = 1e-3
critic_lr = 1e-2
ed_lr = 1e-2
num_episodes = 10
max_sequences = 100
hidden_dim = 128
gamma = 0.98
lmbda = 0.95
epochs = 20
eps = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = BlackMythWuKongEnv(device=device,ed_lr=ed_lr,epochs=epochs)
state_dim = env.state_dim
action_dim = env.action_dim
agent = PPO(state_dim,hidden_dim,action_dim,actor_lr,critic_lr,lmbda,epochs, eps, gamma, device)
# return_list = model_utils.train_on_policy_agent(env,agent,num_episodes,max_sequences)

return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes/10)):
            print("开始战斗")
            # env.activate_window()
            episode_return = 0
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
            state = env.get_state()
            done = False
            squences = 0
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state = next_state
                episode_return += reward
                squences += 1
                if squences == max_sequences:
                    break
            return_list.append(episode_return)
            agent.update(transition_dict)
            env.update(transition_dict)
            agent.save_ckpt()
            env.save_ckpt()
            if (i_episode+1) % 10 == 0:
                pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
            pbar.update(1)
            input("回车进行下一次战斗")
            time.sleep(1)

print(return_list)