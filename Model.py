import torch
from torch import nn
import numpy as np
from Quantum import Hybrid

def argmax(x):
    ''' Own variant of np.argmax with random tie breaking '''
    try:
        return torch.tensor(np.random.choice(torch.where(x == torch.max(x))[0]))
    except:
        return torch.argmax(x)

class ActorCritic(nn.Module):
    """ Simple multi-layer perceptron
    """
    def __init__(self, input_dim, output_dim, value=False, quantum=False, shots=50):
        super(ActorCritic, self).__init__()
        self.quantum = quantum
        self.value = value

        self.hidden_layers = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU()
            )

        if self.value:
            self.value_layer = nn.Sequential(
                nn.Linear(64, 1),
                nn.ReLU()
            )
        else:
            if self.quantum:
                self.quantum_layer = nn.Sequential(
                    nn.Linear(64, output_dim),
                    Hybrid(int(np.log2(output_dim)), shots), 
                    nn.Softmax(dim=0)
                )
            else:
                self.policy_layer = nn.Sequential(
                    nn.Linear(64, output_dim),
                    nn.Softmax(dim=1)
                )

    def forward(self, x, device):
        x = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
        x = self.hidden_layers(x)
        if self.value : return self.value_layer(x)[0]
        elif self.quantum: return self.quantum_layer(x)
        else: return self.policy_layer(x)