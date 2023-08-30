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
        self.logger.info(f'Taking action: {ACTIONS[0]}')
        return ACTIONS[0]
    


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
    
    # celltype, self_position, other_position, coin, danger
    gamestate_2d = [[[value,0,0,0,0] for value in row] for row in game_state['field']]

    # Extract relevant information from the game state
    self_position = game_state['self'][3]
    others_positions = [o[3] for o in game_state['others']]
    coins = game_state['coins']
    bombs = game_state['bombs']

    # Add self to gamestate
    gamestate_2d[self_position[0]][self_position[1]][1] = 1
    # Add others to gamestate
    if (type(others_positions) == list):
        for other in others_positions:
            gamestate_2d[other[0]][other[1]][2] = 1
    else:
        gamestate_2d[others_positions[0]][others_positions[1]][2] = 1
    
    # Add coins to gamestate
    if (type(coins) == list):
        for coin in coins:
            gamestate_2d[coin[0]][coin[1]][3] = 1
    else:
        gamestate_2d[coins[0]][coins[1]][3] = 1
    # Add danger level of position
    if (type(bombs) == list):
        for bomb in bombs:
            # Relative time to detnation remaining. After dropping a bomb it takes 4 time steps t to detonate.
            # Negative for own and positive for others bombs is not possible from this feature space.
            danger_level = bomb[1]/4
            gamestate_2d[bomb[0]][bomb[1]][3] = danger_level
    else:
        danger_level = bombs[1]/4
        gamestate_2d[bombs[0]][bombs[1]][3] = danger_level
    
    # Stack all  and reshape into a vector
    stacked_channels = np.stack(gamestate_2d)
    gamestate_one_hot = stacked_channels.reshape(-1)
    return gamestate_one_hot
