import torch
from torch import nn
import numpy as np
from Quantum import Hybrid


class NN(nn.Module):
    """ Generic NN model.
        This is not used anymore.
    """
    def __init__(self, input_dim, output_dim, n_hidden_layers, neurons_per_layer):
        super(NN, self).__init__()
        
        # Create hidden layers
        hidden_layers = []
        for i in range(n_hidden_layers):
            if i == 0:
                neurons_per_layer = input_dim
            hidden_layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            hidden_layers.append(nn.SELU())
            hidden_layers.append(nn.Dropout(0.2))
        self.hidden_layers = nn.Sequential(*hidden_layers)

        # Create output layer
        if len(hidden_layers) == 0:
            neurons_per_layer = input_dim
        self.output_layer = nn.Sequential(
            nn.Linear(neurons_per_layer, output_dim)
        )

    def forward(self, x):
        """ Forward pass through network
        """
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x


class MLP(NN):
    """ Simple multi-layer perceptron
    """
    def __init__(self, input_dim, output_dim):
        super(NN, self).__init__()
        
        self.hidden_layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.policy_layer = nn.Sequential(
            nn.Linear(64, output_dim),
            nn.Softmax(dim=1)
        ) 
        self.value_layer = nn.Sequential(
            nn.Linear(64, output_dim)
        )

    def forward(self, x, v=False):
        x = torch.tensor(x).unsqueeze(0)
        x = self.hidden_layers(x)
        if v : 
            return self.value_layer(x)
        return self.policy_layer(x)

class MLPHybrid(NN):
    def __init__(self, input_dim, output_dim, shots=50, Ansatz_mode=False):
        super(NN, self).__init__()

        self.Ansatz_mode = Ansatz_mode # if true use just the quantum circit otherwise use the hybrid net 
        self.shots = shots

        self.hidden_layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.fully_connected_layer = nn.Sequential(
            nn.Linear(64, output_dim)
        )

        self.output = nn.Sequential(
            Hybrid(int(np.log2(output_dim)), self.shots), 
            nn.Softmax(dim=0)
        )
        
    def forward(self, x):
        x = torch.tensor(x).unsqueeze(0)
        if self.Ansatz_mode:
            return self.output(x)
        x = self.hidden_layers(x)
        x = self.fully_connected_layer(x)
        x = self.output(x)
        return x
        
    

