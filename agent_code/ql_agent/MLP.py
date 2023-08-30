import torch
import torch.nn as nn

class CustomMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomMLP, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Define layers
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Initialize weights and biases within the specified range
        self.hidden_layer.weight.data.uniform_(-0.5, 0.5)
        self.hidden_layer.bias.data.uniform_(-0.5, 0.5)
        self.output_layer.weight.data.uniform_(-0.5, 0.5)
        self.output_layer.bias.data.uniform_(-0.5, 0.5)
    
    def forward(self, x):
        # Forward pass through hidden layer
        hidden_output = self.hidden_layer(x)
        hidden_output = torch.relu(hidden_output)
        
        # Forward pass through output layer
        output = self.output_layer(hidden_output)
        return output

# Define input size, hidden size, and output size
input_size = ...
hidden_size = 100
output_size = ...

# Create an instance of the custom MLP model
model = CustomMLP(input_size, hidden_size, output_size)
print(model)  # Print the model architecture
