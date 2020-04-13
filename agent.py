import numpy as np
import random
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256         # minibatch size
GAMMA = 0.9            # Discount factor
TAU = 2e-3              # For soft update of target parameters
LR = 1e-3               # Learning rate for actor and critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.action_limits = [-1,1]     # Min, Max of all action values

        # Actor networks
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR)

        # Critic networks
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR)
    
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        
    def step(self, state, action, reward, next_state, done):
        """ Save experience in replay memory, and learn new target weights

        Params
        ======
            state:      current state
            action:     taken action
            reward:     earned reward
            next_state: next state
            done:       Whether episode has finished
        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        This will happen for critic and actor.

        According to the lessons:
            actor_target(state)             gives   action
            critic_target (state, action)   gives   Q-value

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ------------------- update critic ------------------- #
        next_actions = self.actor_target(next_states)
        # Get Q targets (for next states) from target model (on CPU)
        Q_targets_next = self.critic_target(next_states, next_actions).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.critic_local(states, actions)

        # Compute critic loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the critic loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1) # As mentioned on project page
        self.critic_optimizer.step()

        # ------------------- update actor ------------------- #
        actions_expected = self.actor_local(states)
        # Compute actor loss based on expectation from actions_expected
        actor_loss = -self.critic_local(states, actions_expected).mean()
        # Minimize the actor loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


        # ------------------- update target network ------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)                     
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def act(self, state):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.actor_local.eval()

        # Get actions for current state, transformed from probabilities
        #state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            probs = self.actor_local(state)#.cpu().detach().numpy()
        self.actor_local.train()

        #  Transform probability into valid action ranges
        act_min, act_max = self.action_limits
        action = (act_max - act_min) * (probs - 0.5) + (act_max + act_min)/2
        return action

    def save(self, filename):
        """Saves the agent to the local workplace

        Params
        ======
            filename (string): where to save the weights
        """

        checkpoint = {'input_size': self.state_size,
              'output_size': self.action_size,
              'actor_hidden_layers': [each.out_features for each in self.actor_local.hidden_layers if each._get_name()!='BatchNorm1d'],
              'actor_state_dict': self.actor_local.state_dict(),
              'critic_hidden_layers': [each.out_features for each in self.critic_local.hidden_layers if each._get_name()!='BatchNorm1d'],
              'critic_state_dict': self.critic_local.state_dict()}

        torch.save(checkpoint, filename)


    def load_weights(self, filename):
        """ Load weights to update agent's actor and critic networks.
        Expected is a format like the one produced by self.save()

        Params
        ======
            filename (string): where to load data from. 
        """
        checkpoint = torch.load(filename)
        if not checkpoint['input_size'] == self.state_size:
            print(f"Error when loading weights from checkpoint {filename}: input size {checkpoint['input_size']} doesn't match state size of agent {self.state_size}")
            return None
        if not checkpoint['output_size'] == self.action_size:
            print(f"Error when loading weights from checkpoint {filename}: output size {checkpoint['output_size']} doesn't match action space size of agent {self.action_size}")
            return None
        my_actor_hidden_layers = [each.out_features for each in self.actor_local.hidden_layers if each._get_name()!='BatchNorm1d']
        if not checkpoint['actor_hidden_layers'] == my_actor_hidden_layers:
            print(f"Error when loading weights from checkpoint {filename}: actor hidden layers {checkpoint['actor_hidden_layers']} don't match agent's actor hidden layers {my_actor_hidden_layers}")
            return None
        my_critic_hidden_layers = [each.out_features for each in self.critic_local.hidden_layers if each._get_name()!='BatchNorm1d']
        if not checkpoint['critic_hidden_layers'] == my_critic_hidden_layers:
            print(f"Error when loading weights from checkpoint {filename}: critic hidden layers {checkpoint['critic_hidden_layers']} don't match agent's critic hidden layers {my_critic_hidden_layers}")
            return None
        self.actor_local.load_state_dict(checkpoint['actor_state_dict'])
        self.critic_local.load_state_dict(checkpoint['critic_state_dict'])

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
