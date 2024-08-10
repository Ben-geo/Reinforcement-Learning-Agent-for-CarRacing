import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from preprocess import preProcess
from torch.optim import Adam
from torch.distributions import MultivariateNormal

class Network(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_dim[0], out_channels=16, kernel_size=5, stride=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=3)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=3)
        
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(288,64)  
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, out_dim)
    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
class PPOAgent:
    def __init__(self,
                 network_class,
                 env
                ):
        
        
        self.obs_dim = (1,96,96)
        self.act_dim = 3
        self.env = env
        self.timesteps_per_batch = 4800                
        self.max_timesteps_per_episode = 1600          
        self.n_updates_per_iteration = 5              
        self.lr = 0.005                                
        self.gamma = 0.95                              
        self.clip = 0.2                               


        self.render = False                             
        self.save_freq = 10                            
        self.deterministic = False                      
        self.seed = None
        
        self.actor = network_class(self.obs_dim, self.act_dim)                                                   
        self.critic = network_class(self.obs_dim, 1)
        
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)
        self.logger = {
            't_so_far': 0,         
            'i_so_far': 0,         
            'batch_lens': [],       
            'batch_rews': [],      
            'actor_losses': [],     
        }
    def learn(self, total_timesteps):

        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        t_so_far = 0 # Timesteps simulated so far
        i_so_far = 0 # Iterations ran so far
        while t_so_far < total_timesteps:                                                                       # ALG STEP 2
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()                     # ALG STEP 3
            

            t_so_far += np.sum(batch_lens)

            i_so_far += 1

            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far


            V, _ = self.evaluate(batch_obs, batch_acts, batch_rtgs)
            A_k = batch_rtgs - V.detach()                                                                       
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):                                                       
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts, batch_rtgs)

                
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

 
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                self.logger['actor_losses'].append(actor_loss.detach())

        

            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './ppo_actor.pth')
                torch.save(self.critic.state_dict(), './ppo_critic.pth')

    def rollout(self):

        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []


        ep_rews = []

        t = 0 
        while t < self.timesteps_per_batch:
            ep_rews = []
            obs = self.env.reset()[0]
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                if self.render:
                    self.env.render()

                t += 1 #
                batch_obs.append(preProcess.process(obs).reshape((1,96,96)))


                action, log_prob = self.get_action(preProcess.process(obs).reshape((1,96,96)))
        
                obs, rew, done, _,_ = self.env.step(action)

                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).flatten()
        batch_rtgs = self.compute_rtgs(batch_rews)                                                              # ALG STEP 4

        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def compute_rtgs(self, batch_rews):

        batch_rtgs = []
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0 
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)


        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def get_action(self, obs):

        
        obs = np.float32(obs)
        obs = torch.tensor(obs)
        if len(obs.shape)<4:
            obs=torch.unsqueeze(obs, 0)
        mean = self.actor(obs)[0]

        
        dist = MultivariateNormal(mean, self.cov_mat)


   
        action = dist.rsample()

        log_prob = dist.log_prob(action)

        
        if self.deterministic:
            return mean.detach().numpy(), 1

        return action.detach().numpy(), log_prob.detach()

    def evaluate(self, batch_obs, batch_acts, batch_rtgs):

        batch_obs= batch_obs.reshape((batch_obs.shape[0],1,96,96))
        V = self.critic(batch_obs).squeeze()

        
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        return V, log_probs


"""
UnComment to train the PPO Agent
env =  env = gym.make("CarRacing-v2")      
agent = PPOAgent(Network,env)
total_timesteps = 28000000
agent.learn(total_timesteps)
"""