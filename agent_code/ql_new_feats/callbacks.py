import os
import random
from collections import deque
from agent_code.ql_new_feats.features import state_to_features
from agent_code.ql_new_feats.parameters import INPUT_SIZE, HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE, ACTIONS
import agent_code.ql_agent.helper as helper
from .MLP import MLP
import numpy as np
import torch
from .utils import action_index_to_string, action_string_to_index
from ..rule_based_agent import callbacks as rule_based_agent

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
    # For rule based inference learnign:
    self.bomb_buffer = 0
    self.current_round = 0

    self.tiles_visited = set()
    self.coordinate_history = deque([], 20)
    self.position_history = deque([], 5)

    if not os.path.isfile("model/custom_mlp_policy_model.pth") and not self.train:
        # Size of feature representation below
        self.model = MLP(INPUT_SIZE, HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE, 6)
    elif os.path.isfile("model/custom_mlp_policy_model.pth") and not self.train:
        self.logger.info("Loading MLP from saved state.")
        # Create an instance of the custom MLP model
        self.model = MLP(INPUT_SIZE, HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE, 6)
        # Load the saved model state dictionary
        self.model.load_state_dict(torch.load('model/custom_mlp_policy_model.pth'))


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """


    if self.train and random.random() < self.eps_threshold:
        p = 0.0
        rule_based_action = rule_based_agent.act(self, game_state)
        if rule_based_action is not None and np.random.rand() < p:
            self.logger.debug("Random action from rule based Agent.")
            action_chosen = rule_based_action
        else:
            self.logger.debug("Random action.")
            action_chosen = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    else:
        with torch.no_grad():
            state = torch.tensor(state_to_features(game_state), dtype=torch.float32) # Add batch dimension
            prediction = self.model(state).max(-1)[1].view(1, 1).item()
            action_chosen = action_index_to_string(prediction)
            self.logger.info(f'Predicted action: {action_chosen}')
    x, y = game_state['self'][3]
    self.position_history.append((x, y))  
    self.tiles_visited.add((x,y))
    return action_chosen
