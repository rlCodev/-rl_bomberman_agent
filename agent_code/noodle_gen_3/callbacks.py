import os
import random
from collections import deque
from .MLP import MLP
import pickle
import numpy as np
import torch
from .utils import action_index_to_string, action_string_to_index
from ..rule_based_agent import callbacks as rule_based_agent
from agent_code.Noodle.feature import state_to_features
import agent_code.Noodle.static_props as hp

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

    self.model = MLP(hp.INPUT_SIZE, hp.HIDDEN_LAYER_1_SIZE, hp.HIDDEN_LAYER_2_SIZE, 6)
    if os.path.isfile("Noodle_agent.pth"):
        self.logger.info("Loading MLP from saved state.")
        # Load the saved model state dictionary
        self.model.load_state_dict(torch.load('Noodle_agent.pth'))

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """


    if self.train and random.random() < self.eps_threshold:
        p = 0.8
        rule_based_action = rule_based_agent.act(self, game_state)
        if rule_based_action is not None and np.random.rand() < p:
            self.logger.debug("Random action from rule based Agent.")
            action_chosen = rule_based_action
        else:
            self.logger.debug("Random action.")
            action_chosen = np.random.choice(hp.ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    else:
        with torch.no_grad():
            state = state_to_features(game_state)
            prediction = self.model(state).max(-1)[1].view(1, 1).item()
            action_chosen = action_index_to_string(prediction)
            self.logger.info(f'Predicted action: {action_chosen}')

    return action_chosen