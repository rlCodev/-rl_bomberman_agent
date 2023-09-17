import os
import random
from .MLP import MLP
import numpy as np
from ..rule_based_agent import callbacks as rule_based_agent
import torch
from .utils import action_index_to_string, action_string_to_index

INPUT_SIZE = 867
HIDDEN_LAYER_1_SIZE = 620
HIDDEN_LAYER_2_SIZE = 310
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

    if not os.path.isfile("custom_mlp_policy_model.pth") and not self.train:
        # Size of feature representation below
        self.model = MLP(INPUT_SIZE, HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_1_SIZE, 6)
    elif os.path.isfile("custom_mlp_policy_model.pth") and not self.train:
        self.logger.info("Loading MLP from saved state.")
        # Create an instance of the custom MLP model
        self.model = MLP(INPUT_SIZE, HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_1_SIZE, 6)
        # Load the saved model state dictionary
        self.model.load_state_dict(torch.load('custom_mlp_policy_model.pth'))

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    if self.train and random.random() < self.eps_threshold:
        self.logger.debug("Random action.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        # posX, posY = game_state['self'][3]
        # legal_actions = ['WAIT']
        # # game_state2d = gamestate_to_2dMap(game_state)
# 
        # if game_state['field'][posX+1][posY] == 0:
        #     legal_actions.append('RIGHT')
        # if game_state['field'][posX-1][posY] == 0:
        #     legal_actions.append('LEFT')
        # if game_state['field'][posX][posY+1] == 0:
        #     legal_actions.append('DOWN')
        # if game_state['field'][posX][posY-1] == 0:
        #     legal_actions.append('UP')
        # if game_state2d[posX][posY][2] == 0:
        #     legal_actions.append('BOMB')

        # p = [1/len(legal_actions) for i in range(len(legal_actions))]
        #choice = np.random.choice([0,1])
        #if(choice == 0):
        #    return rule_based_agent.act(self, game_state)
        #else:
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    else:
        with torch.no_grad():
            state = torch.tensor(state_to_features(game_state), dtype=torch.float32).unsqueeze(0)
            prediction = self.model(state).max(1)[1].view(1, 1).item()
            chosen_action = action_index_to_string(prediction)
            self.logger.info(f'Predicted action: {chosen_action}')
            return chosen_action

def state_to_features(game_state: dict) -> np.array:
    """
    Converts the game state to the input of your model, i.e. a feature vector.

    :param game_state: A dictionary describing the current game board.
    :return: np.array
    """
    gamestate_2d = gamestate_to_2dMap(game_state)
    stacked_channels = np.stack(gamestate_2d)
    gamestate_one_hot = stacked_channels.reshape(-1)
    return gamestate_one_hot

def gamestate_to_2dMap(game_state: dict) -> np.array:
    if game_state is None:
        return None
    
    # We change our vector as follows:
    # Celltype: -1 = wall, 0 = crate, 1 = free, 2 = coin
    # positions: -1 = other player, 0 = free, 1 = self
    # danger: danger level 1-4 (bomb about to explode), 5 = explosion
    # celltype, positions, danger
    gamestate_2d = [[[value,0,0] for value in row] for row in game_state['field']]

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
            gamestate_2d[other[0]][other[1]][1] = -1
    else:
        gamestate_2d[others_positions[0]][others_positions[1]][1] = 1
    
    # Add coins to gamestate
    if (type(coins) == list):
        for coin in coins:
            gamestate_2d[coin[0]][coin[1]][0] = 2
    else:
        gamestate_2d[coins[0]][coins[1]][0] = 2
    # Add danger level of position
    if (type(bombs) == list):
        for bomb in bombs:
            danger_level = 4-bomb[1]
            width = len(gamestate_2d)
            height = len(gamestate_2d[0])
            pos_x = bomb[0][0]
            pos_y = bomb[0][1]
            top, down, left, right = True, True, True, True
            gamestate_2d[pos_x][pos_y][2] = danger_level
            for i in range(1,4):    
                if  right and pos_x+i <width and gamestate_2d[pos_x+i][pos_y][0] == 0:
                    gamestate_2d[pos_x+i][pos_y][2] = danger_level
                else:
                    right = False    
                if  left and pos_x-i >= 0 and gamestate_2d[pos_x-i][pos_y][0] == 0:
                    gamestate_2d[pos_x-i][pos_y][2] = danger_level
                else:
                    left = False
                if  down and pos_y + 1 < height and gamestate_2d[pos_x][pos_y+i][0] == 0:
                    gamestate_2d[pos_x][pos_y+i][2] = danger_level
                else:
                    down = False
                if  top and pos_y - 1 >= 0 and gamestate_2d[pos_x][pos_y-i][0] == 0:
                    gamestate_2d[pos_x][pos_y-i][2] = danger_level
                else:
                    top = False
    else:
        danger_level = 4 - bombs[1]
        gamestate_2d[bomb[0][0]][bomb[0][1]][2] = danger_level

    # Add explosion map to gamestate
    explosion_map = game_state['explosion_map']
    for i in range(len(explosion_map)):
        for j in range(len(explosion_map[0])):
            if(explosion_map[i][j] > 0):
                gamestate_2d[i][j][2] = 5
    
    return gamestate_2d

def gamestate2D_to_outputMap(gamestate_2d):
    print()
    gamestate_mock_ui = [['.' for value in row] for row in gamestate_2d]
    for i in range(len(gamestate_2d)):
        for j in range(len(gamestate_2d[0])):
            if(gamestate_2d[i][j][0] == -1):
                gamestate_mock_ui[i][j] = "#"
            elif(gamestate_2d[i][j][0] == 1):
                gamestate_mock_ui[i][j] = "X"
            if(gamestate_2d[i][j][3] == 1):
                gamestate_mock_ui[i][j] = "O"
            if(gamestate_2d[i][j][4] != 0):
                gamestate_mock_ui[i][j] = str(gamestate_2d[i][j][4])
            if(gamestate_2d[i][j][1] == 1):
                gamestate_mock_ui[i][j] = "S"
        print(gamestate_mock_ui[i])
    print("-----------------")