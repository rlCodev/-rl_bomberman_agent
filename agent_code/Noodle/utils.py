import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
# Function to convert action space index to string action
def action_index_to_string(action_index):
    return ACTIONS[action_index]

# Function to convert string action to action space index
def action_string_to_index(action_string):
    if action_string in ACTIONS:
        return ACTIONS.index(action_string)
    else:
        raise ValueError(f"Invalid action: {action_string}")
    
def get_own_position(game_state):
    agent = game_state['self']
    return agent[3]

def get_bombs_position(game_state):
    return [bomb[3] for bomb in game_state['bombs']]

def get_others_positions(game_state):
    # Get index 3 for all others (all coordinates)
    return [other[3] for other in game_state['others']]

def get_distance_list(pos, pos_list):
    # Input pos -> [x,y]
    # Input pos_list -> [[x1,y1], [x2,y2], ...]
    # Output -> [distance_1, distance_2, ...]
    return [np.linalg.norm(np.subtract(pos, pos_two)) for pos_two in pos_list]

def get_min_distance(pos, pos_list):
    # Input pos -> [x,y]
    # Input pos_list -> [[x1,y1], [x2,y2], ...]
    # Output -> min_distance
    return min(get_distance_list(pos, pos_list))

def get_crate_positions(game_state):
    return np.argwhere(game_state['field'] == 1)

def is_bomb_available(game_state):
    return game_state['self'][2]