import torch
from torch import nn
import numpy as np
from Quantum import Hybrid


class MLP(nn.Module):
    """ Simple multi-layer perceptron
    """
    def __init__(self, input_dim, output_dim, quantum=False, shots=50):
        super(MLP, self).__init__()
        self.quantum = quantum

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

        self.quantum_layer = nn.Sequential(
            nn.Linear(64, output_dim),
            Hybrid(int(np.log2(output_dim)), shots), 
            nn.Softmax(dim=0)
        )

        self.value_layer = nn.Sequential(
            nn.Linear(64, 1),
            nn.ReLU()
        )

    def forward(self, x, v=False):
        x = torch.tensor(x).unsqueeze(0)
        x = self.hidden_layers(x)
        if v : return self.value_layer(x)
        return self.policy_layer(x) if not self.quantum else self.quantum_layer(x)
        
    

