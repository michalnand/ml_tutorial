import numpy
import torch
import time
import gymnasium as gym

import cv2

from envs.robot_navigation_env  import *
from model_transformer          import *

class AgentPPO():
    def __init__(self, envs, Model):
        self.envs = envs

        # select device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # agent hyperparameters
        self.gamma              = 0.99
        self.entropy_beta       = 0.001
        self.eps_clip           = 0.1 
        self.adv_coeff          = 1.0
        self.val_coeff          = 0.5   

        self.trajectory_steps   = 128
        self.batch_size         = 512
        
        self.training_epochs    = 4
        self.envs_count         = len(envs)
        self.learning_rate      = 0.0001


        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.n

        # policy buffer for storing trajectory
        self._buffer_init()
        
        # create model
        self.model = Model(self.state_shape, self.actions_count)
        self.model.to(self.device)
        print(self.model)

        # initialise optimizer and trajectory buffer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

   
    # agent main step
    def step(self, states, training_enabled = False):        
        states_t = torch.tensor(states, dtype=torch.float).to(self.device)

        # obtain model output, logits and values for all agents in one stap
        logits_t, values_t, attn  = self.model.forward(states_t)

        # attention matrix visualisaiton
        if training_enabled != True:
            attn_img = attn[2][0].detach().cpu().numpy()
            attn_img = attn_img/(numpy.max(attn_img, axis=0, keepdims=True) + 10**-8)
            attn_img = cv2.resize(attn_img, (256, 256))
            cv2.imshow("attention", attn_img)

        
        n_agents     = states_t.shape[1]
        actions_list = numpy.zeros((states_t.shape[0], n_agents), dtype=int)

        # iterate truh all agents to obtain actions
        for n in range(n_agents):
            # sample action, probs computed from logits
            action_probs_t        = torch.nn.functional.softmax(logits_t[:, n], dim = 1)
            action_distribution_t = torch.distributions.Categorical(action_probs_t)
            action_t              = action_distribution_t.sample()
            actions               = action_t.detach().to("cpu").numpy()

            actions_list[:, n]    = actions

       
        # environment step
        states_new, rewards, dones, infos = self.envs.step(actions_list)

        #put into trajectory buffer
        if training_enabled:    
            # store results from agent on first index
            logits_t     = logits_t[:, 0]
            values_t     = values_t[:, 0]
            actions      = actions_list[:, 0]

            self._buffer_add(states_t, logits_t, values_t, actions, rewards, dones)

            # if buffer is full, run training loop and clear buffer after
            if self.buffer_ptr >= self.trajectory_steps:
                self._compute_returns(self.gamma)
                self._train()
                self._buffer_init()
  

        return states_new, rewards, dones, infos
    
    def save(self, result_path):
        torch.save(self.model.state_dict(), result_path + "/model.pt")

    def load(self, result_path):
        self.model.load_state_dict(torch.load(result_path + "/model.pt", map_location = self.device, weights_only=True))

    def _train(self): 
        samples_count = self.trajectory_steps*self.envs_count
        batch_count = samples_count//self.batch_size

        # epoch training
        for e in range(self.training_epochs):
            for batch_idx in range(batch_count):
                # sample batch
                states, logits, actions, returns, advantages = self._sample_batch(self.batch_size)
                
                # compute main PPO loss
                loss_ppo = self.loss_ppo(states, logits, actions, returns, advantages)

                self.optimizer.zero_grad()        
                loss_ppo.backward()

                # gradient clip for stabilising training
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step() 

         

    '''
        main PPO loss
    '''
    def loss_ppo(self, states, logits, actions, returns, advantages):
        logits_new, values_new, _  = self.model.forward(states)
        
        # we use only agent on first index
        logits_new = logits_new[:, 0]
        values_new = values_new[:, 0]

        log_probs_old = torch.nn.functional.log_softmax(logits, dim = 1).detach()

        probs_new     = torch.nn.functional.softmax(logits_new,     dim = 1)
        log_probs_new = torch.nn.functional.log_softmax(logits_new, dim = 1)

        '''
            compute critic loss, as MSE
            L = (T - V(s))^2
        '''
        values_new = values_new.squeeze(1)
        loss_value = (returns.detach() - values_new)**2
        loss_value = loss_value.mean()

        ''' 
            compute actor loss, surrogate loss
        '''
        advantages  = self.adv_coeff*advantages.detach() 
        advantages  = (advantages - torch.mean(advantages))/(torch.std(advantages) + 1e-10)

        log_probs_new_  = log_probs_new[range(len(log_probs_new)), actions]
        log_probs_old_  = log_probs_old[range(len(log_probs_old)), actions]
                        
        ratio       = torch.exp(log_probs_new_ - log_probs_old_)
        p1          = ratio*advantages
        p2          = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantages
        loss_policy = -torch.min(p1, p2)  
        loss_policy = loss_policy.mean()  
    
        '''
            compute entropy loss, to avoid greedy strategy
            L = beta*H(pi(s)) = beta*pi(s)*log(pi(s))
        '''
        loss_entropy = (probs_new*log_probs_new).sum(dim = 1)
        loss_entropy = self.entropy_beta*loss_entropy.mean()

        loss = self.val_coeff*loss_value + loss_policy + loss_entropy

        return loss
    
    '''
        trajectory buffer methods
        this is mostly held in separated class
    '''
    # trajectory buffer init
    def _buffer_init(self):
        self.states     = torch.zeros((self.trajectory_steps, self.envs_count, ) + self.state_shape, dtype=torch.float32, device=self.device)
        self.logits     = torch.zeros((self.trajectory_steps, self.envs_count,  self.actions_count), dtype=torch.float32, device=self.device)
        self.values     = torch.zeros((self.trajectory_steps, self.envs_count, ), dtype=torch.float32, device=self.device)        
        self.actions    = torch.zeros((self.trajectory_steps, self.envs_count, ), dtype=int, device=self.device)
        self.reward     = torch.zeros((self.trajectory_steps, self.envs_count, ), dtype=torch.float32, device=self.device)
        self.dones      = torch.zeros((self.trajectory_steps, self.envs_count, ), dtype=torch.float32, device=self.device)

        self.buffer_ptr = 0  

    # add new items into buffer
    def _buffer_add(self, states, logits, values, actions, rewards, dones):
        self.states[self.buffer_ptr]    = states.detach().to("cpu").clone() 
        self.logits[self.buffer_ptr]    = logits.detach().to("cpu").clone() 
        self.values[self.buffer_ptr]    = values.squeeze(1).detach().to("cpu").clone() 
        self.actions[self.buffer_ptr]   = torch.from_numpy(actions)
        
        self.reward[self.buffer_ptr]    = torch.from_numpy(rewards)
        self.dones[self.buffer_ptr]     = torch.from_numpy(dones).float()
        
        self.buffer_ptr = self.buffer_ptr + 1 


    def _compute_returns(self, gamma, lam = 0.95):
        self.returns, self.advantages   = self._gae(self.reward, self.values, self.dones, gamma, lam)
        
        #reshape buffer for faster batch sampling
        self.states     = self.states.reshape((self.trajectory_steps*self.envs_count, ) + self.state_shape)
        self.logits     = self.logits.reshape((self.trajectory_steps*self.envs_count, self.actions_count))

        self.values     = self.values.reshape((self.trajectory_steps*self.envs_count, ))        
     
        self.actions    = self.actions.reshape((self.trajectory_steps*self.envs_count, ))
        
        self.reward     = self.reward.reshape((self.trajectory_steps*self.envs_count, ))
      
        self.dones      = self.dones.reshape((self.trajectory_steps*self.envs_count, ))

        self.returns    = self.returns.reshape((self.trajectory_steps*self.envs_count, ))
        self.advantages = self.advantages.reshape((self.trajectory_steps*self.envs_count, ))
   
    # sampel random batch from buffer
    def _sample_batch(self, batch_size):
        indices         = torch.randint(0, self.envs_count*self.trajectory_steps, size=(batch_size, ))

        states          = self.states[indices]
        logits          = self.logits[indices]
        
        actions         = self.actions[indices]
        
        returns         = self.returns[indices]
        advantages      = self.advantages[indices]
       
        return states, logits, actions, returns, advantages
    
    # gae returns computing - more stable than basic returns computatiom
    def _gae(self, rewards, values, dones, gamma, lam):
        buffer_size = rewards.shape[0]
        envs_count  = rewards.shape[1]
        
        returns     = torch.zeros((buffer_size, envs_count), dtype=torch.float32, device=self.device)
        advantages  = torch.zeros((buffer_size, envs_count), dtype=torch.float32, device=self.device)

        last_gae    = torch.zeros((envs_count), dtype=torch.float32, device=self.device)
        
        for n in reversed(range(buffer_size-1)):
            delta           = rewards[n] + gamma*values[n+1]*(1.0 - dones[n]) - values[n]
            last_gae        = delta + gamma*lam*last_gae*(1.0 - dones[n])
            
            returns[n]      = last_gae + values[n]
            advantages[n]   = last_gae
 
        return returns, advantages

    
  




def training():
    # create multiple environments
    print("creating envs")
    n_envs = 128
    
    envs = RobotNavigationEnv(n_envs)
    states, _ = envs.reset_all()

    print("states shape ", states.shape)


    agent = AgentPPO(envs, ModelTransformer)

        
    episodes_count  = numpy.zeros(len(envs))
    rewards_sum     = numpy.zeros(len(envs))
    rewards_episode = numpy.zeros(len(envs))

    n_steps = 250000


    states, _ = envs.reset_all()
    for n in range(n_steps):
        # agent main step
        states_new, rewards, dones, infos = agent.step(states, True)

        # accumulate rewards for stats
        rewards_sum+= rewards

        # reset environments which finished episode 
        dones_idx = numpy.where(dones)[0]
        for i in dones_idx:
            states_new[i], _ = envs.reset(i)

            episodes_count[i]+= 1
            rewards_episode[i] = rewards_sum[i]
            rewards_sum[i] = 0

        states = states_new.copy()

        if n%128 == 0:    
            episodes_mean = round(episodes_count.mean(), 2)
            rewards_mean  = round(rewards_episode.mean(), 3)
            rewards_std   = round(rewards_episode.std(), 3)
            print(n, episodes_mean, rewards_mean, rewards_std)

        if n%10000 == 0:
            agent.save("trained/RobotNavigationTransformer/")

    agent.save("trained/RobotNavigationTransformer/")

    print("\n\n")
    print("training done")


def inference():
    # create multiple environments
    print("creating envs")
    n_envs = 1
    
    envs = RobotNavigationEnv(n_envs, render_mode="human")
    states, _ = envs.reset_all()

    print("states shape ", states.shape)

    agent = AgentPPO(envs, ModelTransformer)
    
    agent.load("trained/RobotNavigationTransformer/")

    
    rewards_sum     = numpy.zeros(len(envs))
    rewards_episode = numpy.zeros(len(envs))

    states, _ = envs.reset_all()
    while True:
        # agent main step
        states_new, rewards, dones, infos = agent.step(states, False)

        rewards_sum+= rewards

        # reset environments which finished episode 
        dones_idx = numpy.where(dones)[0]
        
        for i in dones_idx:
            states_new[i], _ = envs.reset(i)
            rewards_episode[i] = rewards_sum[i]
            rewards_sum[i] = 0
            print("rewards_episode = ", round(rewards_episode.mean(), 3))

        states = states_new.copy()

        time.sleep(0.05)


if __name__ == "__main__":
    #training()
    inference()