import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()



        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        
        # Initialize weights and biases for linear layers
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, -0.5, 0.5)
                nn.init.uniform_(layer.bias, -0.5, 0.5)

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, -0.5, 0.5)
                nn.init.uniform_(layer.bias, -0.5, 0.5)


    def forward(self, x):
        """Forward pass"""
        return self.layers(x)

# end
