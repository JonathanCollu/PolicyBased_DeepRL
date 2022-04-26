import torch
from torch import nn
import numpy as np
from Quantum import Hybrid


class MLP(nn.Module):
    """ Simple multi-layer perceptron
    """
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()

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
            nn.Linear(64, 1),
            nn.ReLU()
        )

    def forward(self, x, v=False):
        x = torch.tensor(x).unsqueeze(0)
        x = self.hidden_layers(x)
        if v : 
            return self.value_layer(x)
        return self.policy_layer(x)

class MLPHybrid(nn.Module):
    def __init__(self, input_dim, output_dim, shots=50, Ansatz_mode=False):
        super(MLPHybrid, self).__init__()

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
        
    

