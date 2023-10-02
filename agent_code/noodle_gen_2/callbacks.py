import os
import random

from agent_code.noodle_gen_2.feature import state_to_features_matrix
from .MLP import MLP
import numpy as np
import torch
from gymnasium.spaces import Discrete
from agent_code.noodle_gen_2.utils import action_index_to_string, action_string_to_index
from ..rule_based_agent import callbacks as rule_based_agent

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Create a custom Discrete action space
action_space = Discrete(len(ACTIONS))


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # if self.train or not os.path.isfile("my-saved-model.pt"):
    #     self.logger.info("Setting up model from scratch.")
    #     weights = np.random.rand(len(ACTIONS))
    #     self.model = weights / weights.sum()
    # else:
    #     self.logger.info("Loading model from saved state.")
    #     with open("my-saved-model.pt", "rb") as file:
    #         self.model = pickle.load(file)
    input_size = 23
    hidden_size = 128
    output_size = len(ACTIONS)
    self.bomb_buffer = 0
    self.current_round = 0
    self.tiles_visited = {}

    if not os.path.isfile("custom_mlp_policy_model.pth") and not self.train:
        # Size of feature representation below
        self.policy_net = MLP(input_size, hidden_size, output_size)
    elif os.path.isfile("custom_mlp_policy_model.pth") and not self.train:
        self.logger.info("Loading MLP from saved state.")
        # Create an instance of the custom MLP model
        self.policy_net = MLP(input_size, hidden_size, output_size)

        # Load the saved model state dictionary
        # self.policy_net = torch.load('custom_mlp_policy_model.pth')
        # Load the saved model state dictionary
        self.policy_net.load_state_dict(torch.load('custom_mlp_policy_model.pth'))

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    # State object structure:
    # state = {
    #     'round': self.round,
    #     'step': self.step,
    #     'field': np.array(self.arena),
    #     'self': agent.get_state(),
    #     'others': [other.get_state() for other in self.active_agents if other is not agent],
    #     'bombs': [bomb.get_state() for bomb in self.bombs],
    #     'coins': [coin.get_state() for coin in self.coins if coin.collectable],
    #     'user_input': self.user_input,
    # }

    if self.train and random.random() < self.eps_threshold:
        self.logger.debug("Random action.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        # choice = np.random.choice([1,2])
        # if choice == 1:
        rule_based_action = rule_based_agent.act(self, game_state)
        if rule_based_action is not None: #and random.random() < self.eps_threshold:
            return rule_based_action
        else:
            return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1]) 
        # else:
        #     print("Random")
        #     return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    else:
        with torch.no_grad():
            state = torch.tensor(state_to_features_matrix(self, game_state), dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            prediction = self.policy_net(state).argmax(dim=2).item()
            chosen_action = action_index_to_string(prediction)
            
            # action_index = self.policy_net(state).argmax().item()
            # print(action_index)
            # chosen_action = ACTIONS[action_index]
            
            # q_values = self.model(state)
            # action_index = torch.argmax(q_values).item()
            # chosen_action = ACTIONS[action_index]
            self.logger.info(f'Predicted action: {chosen_action}')
            return chosen_action