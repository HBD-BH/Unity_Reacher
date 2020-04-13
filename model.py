import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import chain

# Similar to Deep Q-Network lecture exercise and the PyTorch extracurricular Content
class Actor(nn.Module):
    def __init__(self, input_size, output_size, seed, hidden_layers=[64,64]):
        ''' Builds a feedforward network with arbitrary hidden layers.
        Actor: state --> action
        
            Arguments
            ---------
            input_size: integer, size of the input (e.g., state space)
            output_size: integer, size of the output layer (e.g., action space)
            seed (int): Random seed
            hidden_layers: list of integers, the sizes of the hidden layers
        '''
        super().__init__()
        self.seed = torch.manual_seed(seed)
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        self.hidden_layers.extend([nn.BatchNorm1d(hidden_layers[0])])
        
        # Add a variable number of more hidden layers, including normalization for each layer
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend(list(chain.from_iterable((nn.Linear(h1, h2), nn.BatchNorm1d(h2)) for h1, h2 in layer_sizes)))
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
    def forward(self, state):
        ''' Forward pass through the network, returns the action '''
        
        x = state
        # Forward through each layer in `hidden_layers`, with ReLU activation
        for linear in self.hidden_layers:
            x = F.selu(linear(x))
        
        x = self.output(x)
        
        return F.tanh(x)
    
# Similar to Deep Q-Network lecture exercise and the PyTorch extracurricular Content
class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden_layers=[64,64]):
        ''' Builds a feedforward network with arbitrary hidden layers.
        Critic: (state, action) --> Q-values
        
            Arguments
            ---------
            state_size: integer, size of the state space
            action_size: integer, size of the action space
            seed (int): Random seed
            hidden_layers: list of integers, the sizes of the hidden layers
                if single int x: using [x,x]
                otherwise, at least two elements are needed.
        '''
        super().__init__()
        self.seed = torch.manual_seed(seed)
        # Transform single int to two-element list of hidden layers
        if len(hidden_layers)==1:
            hidden_layers = [hidden_layers, hidden_layers]
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
        self.hidden_layers.extend([nn.Linear(hidden_layers[0] + action_size, hidden_layers[1])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[1:-1], hidden_layers[2:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        # We want to estimate max Q value, so only 1 single output node.
        self.output = nn.Linear(hidden_layers[-1], 1)
        
    def forward(self, state, action):
        ''' Forward pass through the network, returns the max Q value'''
        
        # State is input into first layer, convert everything to float tensors
        x = F.selu(self.hidden_layers[0](state)).float()
        action = action.float()
        
        # Action comes as additional input for second layer
        x = torch.cat((x, action), dim=1)
        x = F.selu(self.hidden_layers[1](x))

        # Forward through each other layer in `hidden_layers`, with ReLU activation
        if len(self.hidden_layers)>2:
            for linear in self.hidden_layers[2:]:
                x = F.selu(linear(x))
        
        x = self.output(x)
        
        return x
    
