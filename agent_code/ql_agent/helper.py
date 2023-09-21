import numpy as np
from collections import deque
import settings as s

STEP = np.array([[1,0], [-1,0], [0,1], [0,-1]])

def state_to_features_matrix(game_state, tiles_explored_set):
    position = game_state['self'][3]
    feature_matrix = []

    for step in STEP:
        move_coords = position + step
        move_feature_vector = []

        # check invalid actions
        move_feature_vector.append(invalid_action(move_coords, game_state))

        # get distance to nearest coin
        move_feature_vector.append(distance_to_coin(move_coords, game_state))

        # get distance to nearest opponent
        move_feature_vector.append(distance_to_opponent(move_coords, game_state))

        # get number of tiles explored
        move_feature_vector.append(tiles_explored(move_coords, game_state, tiles_explored_set))

        # get danger
        # TODO

        # get certain death
        # TODO

        # get bomb effect
        move_feature_vector.append(bomb_effect(move_coords, game_state))

        # check for chained invalid actions 

        # check for backtracked moves
        

def invalid_action(position, game_state):
    if game_state['field'][position] == 0:
        return 1
    else:
        return -1

def distance_to_coin(position, game_state):
    # Returns distance of nearest coin for each step and for the current position respectively
    delta_coins = {}
    for coin in game_state['coins']:
        delta_coins[coin] = manhattan_distance(position, coin)
    nearest_coins = sorted(delta_coins.items(), key=lambda x:x[1])
    return nearest_coins[0][1]

def distance_to_opponent(position, game_state):
    # Returns distance of nearest opponents for each step and for the current position respectively
    opponents = [player[1] for player in game_state['others']]
    return nearest_distance(position, opponents)

# helper function to calculate the nearest distance to a list of coordinates
def nearest_distance(position, coordinates):
    distance = 100
    for coord in coordinates:
        if manhattan_distance(position, coord) < distance:
            distance = manhattan_distance(position, coord)
    return distance

def tiles_explored(position, game_state, tiles_explored_set):
    if game_state['field'][position] == 0:
        if position not in tiles_explored_set:
            return 1
        else:
            return 0
    else:
        return -1

def bomb_effect(position, game_state):
        '''
        calculate the effectivenes of a bomb at position pos

        :param pos: position of bomb (x,y)
        '''
        destroyed_crates = 0
        for direction in STEP:
            for radius in range(1, 4):
                explosion = position + direction*radius
                tile = game_state['field'][explosion[0], explosion[1]]
                if tile == -1:
                    break
                if (tile == 1): # we will ge the crate destroyed
                    destroyed_crates += 1
        return destroyed_crates

def calculate_bomb_effectiveness(game_state, bomb_position):
    # Define constants for tile types
    FREE_TILE = 0
    CRATE = 1
    STONE_WALL = -1

    # Extract relevant information from the game state
    field = game_state['field'].copy()
    explosion_map = game_state['explosion_map'].copy()
    self_position = game_state['self'][3]
    others_positions = [agent[3] for agent in game_state['others']]

    # Simulate the bomb explosion at the given position
    explosion_radius = game_state['bombs'][0][1]  # Assuming all bombs have the same explosion radius
    bomb_x, bomb_y = bomb_position
    explosion_map[bomb_x, bomb_y] = explosion_radius

    # Calculate the number of crates destroyed by the bomb
    crates_destroyed = 0
    for dx in range(-explosion_radius, explosion_radius + 1):
        for dy in range(-explosion_radius, explosion_radius + 1):
            x, y = bomb_x + dx, bomb_y + dy
            if 0 <= x < field.shape[0] and 0 <= y < field.shape[1]:
                if explosion_map[x, y] > 0:
                    if field[x, y] == CRATE:
                        crates_destroyed += 1

    # Calculate the proximity to other agents
    proximity_to_others = 0
    for agent_position in others_positions:
        distance = np.linalg.norm(np.array(agent_position) - np.array(bomb_position))
        if distance <= explosion_radius:
            proximity_to_others += 1

    # Calculate the potential effectiveness score
    effectiveness_score = crates_destroyed - proximity_to_others

    return effectiveness_score


def manhattan_distance(point1, point2):
    """
    Calculate the Manhattan distance between two points.

    Args:
    point1 (tuple): The coordinates of the first point as (x, y).
    point2 (tuple): The coordinates of the second point as (x, y).

    Returns:
    int: The Manhattan distance between the two points.
    """
    x1, y1 = point1
    x2, y2 = point2
    return abs(x1 - x2) + abs(y1 - y2)

def get_extended_explosion_map(game_state):
    field = game_state['field']
    bombs = game_state['bombs']
    extended_explosion_map = np.full_like(field, -1)
    
    for bomb_position, countdown in bombs:
        extended_explosion_map[bomb_position[0], bomb_position[1]] = countdown

        for length in range(1, s.BOMB_POWER + 1):
            # Calculate the possible explosion positions
            beams = bomb_position + STEP * length
            
            # Clip the positions to stay within the field
            beams = np.clip(beams, 0, np.array(field.shape) - 1)
            
            # Get objects at the possible explosion positions
            objects = field[beams[:, 0], beams[:, 1]]
            
            # Update the explosion map where necessary
            update_mask = (objects != -1) & (extended_explosion_map[beams[:, 0], beams[:, 1]] < countdown)
            extended_explosion_map[beams[update_mask][:, 0], beams[update_mask][:, 1]] = countdown

    return extended_explosion_map

# def get_extended_explosion_map(game_state):
#     field = game_state['field']
#     bombs = game_state['bombs']
#     extended_explosion_map = np.full_like(field, -1)
    
#     for bomb_position, countdown in bombs:
#         extended_explosion_map[bomb_position[0], bomb_position[1]] = countdown

#         for length in range(1, s.BOMB_POWER + 1):
#             # Calculate the possible explosion positions
#             beams = bomb_position + STEP * length
            
#             # Clip the positions to stay within the field
#             beams = np.clip(beams, 0, np.array(field.shape) - 1)
            
#             # Get objects at the possible explosion positions
#             objects = field[beams[:, 0], beams[:, 1]]
            
#             # Update the explosion map where necessary
#             update_mask = (objects != -1) & (extended_explosion_map[beams[:, 0], beams[:, 1]] < countdown)
#             extended_explosion_map[beams[update_mask][:, 0], beams[update_mask][:, 1]] = countdown

#     return extended_explosion_map

def get_extended_explosion_map(game_state):
    field = game_state['field']
    bombs = game_state['bombs'].copy()
    extended_explosion_map = np.full_like(field, -1)
    
    for bomb_position, countdown in bombs:
        extended_explosion_map[bomb_position[0], bomb_position[1]] = countdown
        
        for direction in STEP:
            for length in range(1, s.BOMB_POWER + 1):
                beam = direction * length + np.array(bomb_position)
                obj = field[beam[0], beam[1]]
                
                if obj == -1:
                    break
                if extended_explosion_map[beam[0], beam[1]] < countdown:
                    extended_explosion_map[beam[0], beam[1]] = countdown
                else:
                    break  # No need to continue updating if countdown is not greater
    return extended_explosion_map

def get_danger(field, bombs, position, direction):
    extended_explosion_map = np.zeros_like(field)
    max_danger = s.BOMB_POWER + 1
    for bomb_position, countdown in bombs:
        extended_explosion_map[bomb_position[0], bomb_position[1]] = max_danger
        
        for direction in STEP:
            for length in range(1, s.BOMB_POWER + 1):
                beam = direction * length + np.array(bomb_position)
                obj = field[beam[0], beam[1]]
                
                if obj == -1:
                    break
                if extended_explosion_map[beam[0], beam[1]] < max_danger - length:
                    extended_explosion_map[beam[0], beam[1]] = max_danger - length
                else:
                    break  # No need to continue updating if countdown is not greater
    new_agent_position = position + direction
    return extended_explosion_map[new_agent_position[0], new_agent_position[1]]


def certain_death(game_state, direction):
    # Get the extended explosion map
    extended_explosion_map = get_extended_explosion_map(game_state)
    
    # Extract relevant information from the game state
    self_x, self_y = game_state['self'][3] + direction
    time_to_explosion = extended_explosion_map[self_x, self_y]

    # Get closest safe tile without explosion
    if time_to_explosion == -1:
        return False
    else:
        # Get coordinates of the closest safe tile
        safe_tiles = np.argwhere(extended_explosion_map == -1)
        closest_safe_tile = closest_safe_tile[np.argmin(np.sum(np.abs(safe_tiles - [self_x, self_y]), axis=1))]

        # Calculate steps to take to reach the closest safe tile considering walls and crates
        return (not tile_reachable(game_state, closest_safe_tile, extended_explosion_map))
        

def tile_reachable(game_state, tile, extended_explosion_map):
    # Extract relevant information from the game state
    field = game_state['field']
    self_x, self_y = game_state['self'][3]


    found_tile = False
    steps = 0

    while not found_tile:
        valid_steps = get_valid_actions((self_x, self_y), game_state)
        # Get step with minimum distance to the tile
        min_step = valid_steps[np.argmin(np.sum(np.abs(np.array(valid_steps) - np.array(tile)), axis=1))]
        self_x, self_y = self_x + min_step[0], self_y + min_step[1]
        # Update explosion map -1 for each tile where not -1
        if extended_explosion_map[self_x, self_y] == 0:
            return False
        if (self_x, self_y) == tile and extended_explosion_map[self_x, self_y] != 0:
            return True

        extended_explosion_map[extended_explosion_map != -1] -= 1
        steps += 1
        

def get_valid_actions(position, game_state):
    valid_actions = []

    for (dx, dy) in STEP.items():
        new_position = (position[0] + dx, position[1] + dy)

        # Check if the action is valid using the invalid_action method
        if invalid_action(new_position, game_state) == 1:
            valid_actions.append((dx, dy))

    return valid_actions