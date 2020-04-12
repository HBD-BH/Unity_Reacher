import numpy as np
import random
#from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 

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

        # Policy and Optimizer
        self.policy = Policy(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)

    
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """


    def save(self, filename):
        """Saves the agent to the local workplace

        Params
        ======
            filename (string): where to save the weights
        """

        checkpoint = {'input_size': self.state_size,
              'output_size': self.action_size,
              'hidden_layers': [each.out_features for each in self.qnetwork_local.hidden_layers],
              'state_dict': self.policy.state_dict()}

        torch.save(checkpoint, filename)


    def load_weights(self, filename):
        """ Load weights to update agent's Q-Network.
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
        my_hidden_layers = [each.out_features for each in self.qnetwork_local.hidden_layers]
        if not checkpoint['hidden_layers'] == my_hidden_layers:
            print(f"Error when loading weights from checkpoint {filename}: hidden layers {checkpoint['hidden_layers']} don't match agent's hidden layers {my_hidden_layers}")
            return None
        self.policy.load_state_dict(checkpoint['state_dict'])


class Policy(nn.Module):
    """The policy agents will follow."""

    def __init__(self, input_size, output_size, seed, hidden_layers=[64,64]):
        """ Builds a feedforward network with arbitrary hidden layers.

        Params: 
        =======
            input_size (int): Size of state space (inputs to NN)
            output_size (int): Size of action space (output of NN)
            seed (int): Random seed
            hidden_layers ([int*]): sizes of hidden layers used.
        """ 
        super().__init__()
        self.seed = torch.manual_seed(seed)
        # state space size = 33 to action space size =4

        # First hidden layer from input (=state) space
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        # Add all hidden layers
        layer_sizes = zip(hidden_layers[:-1],hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        # Transform to output space
        self.output = nn.Linear(hidden_layers[-1], output_size)

        # Sigmoid to get probabilities
        self.sig = nn.Sigmoid()

    def forward(self, x):
        """ Forward pass through the network, returns the output probabilities """

        for hl in self.hidden_layers:
            x = F.relu(hl(x))

        x = self.output(x)

        return self.sig(x)

