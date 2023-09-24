from collections import deque
import os
import random

from agent_code.ql_agent.helper import state_to_features_matrix
import agent_code.ql_agent.helper as helper
from .MLP import MLP
import numpy as np
import torch
from .utils import action_index_to_string, action_string_to_index
from ..rule_based_agent import callbacks as rule_based_agent
import agent_code.ql_agent.hyperparameter as c

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

    # For rule based inference learnign:
    self.bomb_buffer = 0
    self.current_round = 0

    self.tiles_visited = set()
    self.coor_hist = deque([], 10)

    if not os.path.isfile("custom_mlp_policy_model.pth") and not self.train:
        # Size of feature representation below
        self.policy_net = MLP(c.INPUT_SIZE, c.HIDDEN_SIZE, c.HIDDEN_SIZE_2, c.HIDDEN_SIZE_3, c.OUTPUT_SIZE)
    elif os.path.isfile("custom_mlp_policy_model.pth") and not self.train:
        # Create an instance of the custom MLP model
        self.policy_net = MLP(c.INPUT_SIZE, c.HIDDEN_SIZE, c.HIDDEN_SIZE_2, c.HIDDEN_SIZE_3, c.OUTPUT_SIZE)

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

    # if self.train and random.random() < self.eps_threshold:
    #     self.logger.debug("Random action.")
    #     # 80%: walk in any direction. 10% wait. 10% bomb.
    #     # choice = np.random.choice([1,2])
    #     # if choice == 1:
    #     rule_based_action = rule_based_agent.act(self, game_state)
    #     if rule_based_action is not None: #and random.random() < self.eps_threshold:
    #         return rule_based_action
    #     else:
    #         return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1]) 
    action_chosen = None
    # if self.train and random.random() < self.eps_threshold:
    if self.train and random.random() < 0.7:
        # 80%: walk in any direction. 10% wait. 10% bomb.
        # return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        # rule_based_action = rule_based_agent.act(self, game_state)
        # rule_based_choice = np.random.choice([1,2])
        # if rule_based_action is not None: #and rule_based_choice == 1:
        #     action_chosen = rule_based_action
        # else:
        action_chosen = np.random.choice(c.ACTIONS, p=[.2, .2, .2, .2, .1, .1]) 
    else:
        with torch.no_grad():
            state = torch.tensor(state_to_features(self, game_state), dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            prediction = self.policy_net(state).max(1)[1].view(1, 1).item()
            action_chosen = action_index_to_string(prediction)

            # If agent has been in the same location three times recently, it's a loop
            if action_chosen != 'BOMB':
                try:
                    x, y = game_state['self'][3] + helper.get_step(action_chosen)
                except:
                    x, y = game_state['self'][3]
                if self.coor_hist.count((x, y)) > 4:
                    self.logger.debug("LOOP DETECTED")
                    try:
                        valid_actions = helper.get_valid_action_strings(game_state)
                        valid_actions.remove(action_chosen)
                        action_chosen = np.random.choice(valid_actions)
                    except:
                        action_chosen = 'DOWN'
            
                new_pos = tuple(helper.get_step(action_chosen) + game_state['self'][3])
                self.coor_hist.append(new_pos)
            # action_index = self.policy_net(state).argmax().item()
            # print(action_index)
            # chosen_action = ACTIONS[action_index]
            
            # q_values = self.model(state)
            # action_index = torch.argmax(q_values).item()
            # chosen_action = ACTIONS[action_index]
    self.logger.info(f'Predicted action: {action_chosen}')
    if action_chosen != 'BOMB':
        new_pos = tuple(helper.get_step(action_chosen) + game_state['self'][3])
        self.tiles_visited.add(new_pos)

    if self.train:    
        self.action_history.append(action_chosen)
        
    self.logger.info(f'Took action: {action_chosen}')
    
    return action_chosen


    # # todo Exploration vs exploitation
    # random_prob = .1
    # if self.train and random.random() < random_prob:
    #     self.logger.debug("Choosing action purely at random.")
    #     # 80%: walk in any direction. 10% wait. 10% bomb.
    #     return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    # self.logger.debug("Querying model for action.")
    # return np.random.choice(ACTIONS, p=self.model)


def state_to_features(self, game_state: dict) -> np.array:
    """
    Converts the game state to the input of your model, i.e. a feature vector.

    :param game_state: A dictionary describing the current game board.
    :return: np.array
    """
    if game_state is None:
        return None
    
    feature = helper.state_to_features_matrix(self, game_state).ravel()

    return feature
    
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
            danger_level = 4-bomb[1]/4
            # TODO: Add danger level to all cells in blast radius
            width = len(gamestate_2d)
            height = len(gamestate_2d[0])
            pos_x = bomb[0][0]
            pos_y = bomb[0][1]
            top, down, left, right = True, True, True, True
            gamestate_2d[pos_x][pos_y][3] = danger_level
            for i in range(1,3):    
                if  right and pos_x+i <width and gamestate_2d[pos_x+i][pos_y][0] == 0:
                    gamestate_2d[pos_x+i][pos_y][3] = danger_level
                else:
                    right = False    
                if  left and pos_x-i >= 0 and gamestate_2d[pos_x-i][pos_y][0] == 0:
                    gamestate_2d[pos_x-i][pos_y][3] = danger_level
                else:
                    left = False
                if  down and pos_y + 1 < height and gamestate_2d[pos_x][pos_y+i][0] == 0:
                    gamestate_2d[pos_x][pos_y+i][3] = danger_level
                else:
                    down = False
                if  top and pos_y - 1 >= 0 and gamestate_2d[pos_x][pos_y-i][0] == 0:
                    gamestate_2d[pos_x][pos_y-i][3] = danger_level
                else:
                    top = False
                
            #     if pos_x <= width and gamestate_2d[pos_x][bomb[0][1]][0] == 0:
            # gamestate_2d[bomb[0][0]-3:bomb[0][0]+3][bomb[0][1]][3] = danger_level
            # gamestate_2d[bomb[0][0]][bomb[0][1]-3:bomb[0][1]+3][3] = danger_level
    else:
        danger_level = 4 - bombs[1]/4
        gamestate_2d[bombs[0]][bombs[1]][3] = danger_level

    # TODO: add explosion map

    
    # Stack all  and reshape into a vector
    stacked_channels = np.stack(gamestate_2d)
    gamestate_one_hot = stacked_channels.reshape(-1)
    return gamestate_one_hot