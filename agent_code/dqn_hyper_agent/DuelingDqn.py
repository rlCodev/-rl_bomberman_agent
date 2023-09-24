# import torch
# import torch.nn as nn
import agent_code.dqn_hyper_agent.static_params as c

# class DuelingDeepQNetwork(nn.Module):
#     def __init__(self):
#         super(DuelingDeepQNetwork, self).__init__()
#         self.dense1 = nn.Linear(c.STATE_SHAPE, c.DENSE_LAYER_DIMS)
#         self.dense2_val = nn.Linear(int(c.DENSE_LAYER_DIMS/2), 1)
#         self.dense2_adv = nn.Linear(int(c.DENSE_LAYER_DIMS/2), len(c.ACTIONS))
#         self.flatten = nn.Flatten()

#         # Initialize weights using He initialization
#         nn.init.kaiming_uniform_(self.dense1.weight, mode='fan_in', nonlinearity='relu')
#         nn.init.kaiming_uniform_(self.dense2_val.weight, mode='fan_in', nonlinearity='relu')
#         nn.init.kaiming_uniform_(self.dense2_adv.weight, mode='fan_in', nonlinearity='relu')

#     def forward(self, x):
#         x = self.flatten(x)
#         x = c.ACTIVATION_FUNCTION(self.dense1(x))

#         # Split into val and adv streams
#         # val, adv = torch.split(x, 2, dim=1)
#         val, adv = torch.split(x, x.size(1) // 2, dim=1)

#         # Flatten before input into dense layers
#         val = self.flatten(val.unsqueeze(1))
#         adv = self.flatten(adv.unsqueeze(1))

#         val = torch.relu(self.dense2_val(val))
#         adv = torch.relu(self.dense2_adv(adv))
        
#         # Combine streams into Q-values
#         qvals = val + (adv - adv.mean(dim=1, keepdim=True))
        
#         return qvals


# import torch
# import torch.nn as nn
# import agent_code.dqn_hyper_agent.static_params as c

# class DuelingDeepQNetwork(nn.Module):
#     def __init__(self):
#         super(DuelingDeepQNetwork, self).__init__()
#         self.conv1 = nn.Conv2d(c.STATE_SHAPE[0], 32, kernel_size=8, stride=4)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
#         # Calculate the output size after convolution layers
#         conv_out_size = self._get_conv_out(c.STATE_SHAPE)
        
#         self.dense1_val = nn.Linear(conv_out_size, 256)
#         self.dense1_adv = nn.Linear(conv_out_size, 256)
#         self.dense2_val = nn.Linear(256, 1)
#         self.dense2_adv = nn.Linear(256, len(c.ACTIONS))

#         # Initialize weights using He initialization
#         nn.init.kaiming_uniform_(self.dense1_val.weight, mode='fan_in', nonlinearity='relu')
#         nn.init.kaiming_uniform_(self.dense1_adv.weight, mode='fan_in', nonlinearity='relu')
#         nn.init.kaiming_uniform_(self.dense2_val.weight, mode='fan_in', nonlinearity='relu')
#         nn.init.kaiming_uniform_(self.dense2_adv.weight, mode='fan_in', nonlinearity='relu')

#     def _get_conv_out(self, shape):
#         o = self.conv1(torch.zeros(1, *shape))
#         o = self.conv2(o)
#         o = self.conv3(o)
#         return int(o.view(1, -1).size(1))

#     def forward(self, x):
#         x = torch.relu(self.conv1(x))
#         x = torch.relu(self.conv2(x))
#         x = torch.relu(self.conv3(x))

#         # Flatten before input into dense layers
#         x = x.view(x.size(0), -1)

#         val = torch.relu(self.dense1_val(x))
#         adv = torch.relu(self.dense1_adv(x))
        
#         val = self.dense2_val(val)
#         adv = self.dense2_adv(adv)
        
#         # Combine streams into Q-values
#         adv_mean = adv.mean(dim=1, keepdim=True)
#         qvals = val + (adv - adv_mean)
        
#         return qvals


import torch
import torch.nn as nn


class DuelingDeepQNetwork(nn.Module):
    def __init__(self):
        super().__init__()



        self.layers = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(c.STATE_SHAPE, c.DENSE_LAYER_DIMS),
            nn.ReLU(),
            nn.Linear(c.DENSE_LAYER_DIMS, c.DENSE_LAYER_DIMS_2),
            nn.ReLU(),
             nn.Linear(c.DENSE_LAYER_DIMS_2, c.DENSE_LAYER_DIMS_3),
            nn.ReLU(),
            nn.Linear(c.DENSE_LAYER_DIMS_3, len(c.ACTIONS)),
        )
        
        # Initialize weights and biases for linear layers
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.uniform_(layer.bias, -0.5, 0.5)


    def forward(self, x):
        """Forward pass"""
        return self.layers(x)
