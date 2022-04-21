import torch
from torch import nn


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

        self.output_layer = nn.Sequential(
            nn.Linear(64, output_dim),
            nn.Softmax(dim=1)
        )