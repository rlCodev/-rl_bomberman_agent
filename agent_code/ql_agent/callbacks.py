import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


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
    if self.train or not os.path.isfile("coin-collector-qtable.pt"):
        self.logger.info("Setting up Q-table from scratch.")
        self.q_table = None
    else:
        self.logger.info("Loading Q-table from saved state.")
        with open("coin-collector-qtable.pt", "rb") as file:
            self.q_table = pickle.load(file)


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
    if self.q_table is None:
        state_space_size = state_to_features(game_state).shape[0]
        self.q_table = np.zeros([state_space_size, len(ACTIONS)])  # Initialize Q-table with zeros

    if self.train and np.random.rand() < self.exploration_rate:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    else:
        state = state_to_features(game_state)
        action_index = np.argmax(self.q_table[state, :])
        self.logger.info(f'Taking action: {ACTIONS[action_index]}')
        return ACTIONS[action_index]
    


    # # todo Exploration vs exploitation
    # random_prob = .1
    # if self.train and random.random() < random_prob:
    #     self.logger.debug("Choosing action purely at random.")
    #     # 80%: walk in any direction. 10% wait. 10% bomb.
    #     return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    # self.logger.debug("Querying model for action.")
    # return np.random.choice(ACTIONS, p=self.model)


def state_to_features(game_state: dict) -> np.array:
    """
    Converts the game state to the input of your model, i.e. a feature vector.

    :param game_state: A dictionary describing the current game board.
    :return: np.array
    """
    if game_state is None:
        return None
    
    # Extract relevant information from the game state
    field = game_state['field']
    player_state = game_state['self']
    coins = game_state['coins']
    bombs = game_state['bombs']
    
    # Create channels for different features
    channels = []
    
    # Channel for walls and crates
    wall_crate_channel = np.where(field == -1, 1, 0)  # Walls and crates are set to 1, rest to 0
    channels.append(wall_crate_channel)
    
    # Channel for player's position
    player_channel = np.zeros_like(field)
    player_channel[player_state[3][1], player_state[3][0]] = 1  # Set player's position to 1
    channels.append(player_channel)
    
    # Channel for coin positions
    coin_channel = np.zeros_like(field)
    for coin in coins:
        coin_channel[coin] = 1  # Set coin positions to 1
    channels.append(coin_channel)
    
    # Channel for bomb positions and timers
    bomb_channel = np.zeros_like(field)
    for bomb in bombs:
        bomb_pos = bomb[0]
        bomb_timer = bomb[1]
        bomb_channel[bomb_pos] = bomb_timer  # Set bomb positions to their timers
    channels.append(bomb_channel)
    
    # Stack all channels and reshape into a vector
    stacked_channels = np.stack(channels)
    return stacked_channels.reshape(-1)
