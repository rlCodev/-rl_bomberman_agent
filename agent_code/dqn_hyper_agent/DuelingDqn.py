import torch
import torch.nn as nn
import agent_code.dqn_hyper_agent.static_params as c

class DuelingDeepQNetwork(nn.Module):
    def __init__(self):
        super(DuelingDeepQNetwork, self).__init__()
        self.dense1 = nn.Linear(c.STATE_SHAPE, c.DENSE_LAYER_DIMS)
        self.dense2_val = nn.Linear(c.DENSE_LAYER_DIMS, 1)
        self.dense2_adv = nn.Linear(c.DENSE_LAYER_DIMS, len(c.ACTIONS))
        self.flatten = nn.Flatten()

        # Initialize weights using He initialization
        nn.init.kaiming_uniform_(self.dense1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.dense2_val.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.dense2_adv.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = self.dense1(x)

        # Split into val and adv streams
        val, adv = torch.split(x, 1, dim=1)
        # Flatten before input into dense layers
        val = self.flatten(val.unsqueeze(1))
        adv = self.flatten(adv.unsqueeze(1))

        val = torch.relu(self.dense2_val(val))
        adv = torch.relu(self.dense2_adv(adv))
        
        # Combine streams into Q-values
        qvals = val + (adv - adv.mean(dim=1, keepdim=True))
        
        return qvals
