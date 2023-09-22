import numpy as np
from collections import deque
import settings as s

STEP = np.array([(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)])
ACTION_NAME = ['RIGHT', 'LEFT', 'UP', 'DOWN', 'WAIT']


def state_to_features_matrix(self, game_state, previous_feature_matrix=None):
    position = game_state['self'][3]
    feature_matrix = []
    print("step feature")
    danger_map, extended_explosion_map = get_danger_map(game_state['field'], game_state['bombs'])
    for step in STEP:
        move_coords = position + step
        if (valid_action(move_coords, game_state)):
            move_feature_vector = []

            # get distance to nearest coin
            move_feature_vector.append(distance_to_coin(move_coords, game_state))

            # get distance to nearest opponent
            move_feature_vector.append(distance_to_opponent(move_coords, game_state))

            # get number of tiles explored
            if (tuple(step) == (0, 0)):
                move_feature_vector.append(len(self.tiles_visited))
            else:
                move_feature_vector.append(tiles_explored(move_coords, game_state, self.tiles_visited))

            # get danger
            move_feature_vector.append(get_danger(danger_map, move_coords))

            # get certain death
            move_feature_vector.append(certain_death(game_state, move_coords, danger_map, extended_explosion_map))

            # get bomb effect
            move_feature_vector.append(bomb_effect(move_coords, game_state))

            move_feature_vector.append(backtracking(step), previous_feature_matrix)

        else:
            move_feature_vector = [-1] * 6
        # # check for chained invalid actions 
        # move_feature_vector.append(chain_inv_a(self, move_coords, game_state))

        # # check for backtracked moves
        # move_feature_vector.append(backtracked_move(self, move_coords, game_state))

        feature_matrix.append(move_feature_vector)
    return np.array(feature_matrix)


def invalid_action(position, game_state):
    if game_state['field'][position[0]][position[1]] == 0:
        return 1
    else:
        return -1


def distance_to_coin(position, game_state):
    # Returns distance of nearest coin for each step and for the current position respectively
    if game_state['coins'] == []:
        return 0
    delta_coins = {}
    for coin in game_state['coins']:
        delta_coins[coin] = manhattan_distance(position, coin)
    nearest_coins = sorted(delta_coins.items(), key=lambda x: x[1])
    return nearest_coins[0][1]


def distance_to_opponent(position, game_state):
    # Returns distance of nearest opponents for each step and for the current position respectively
    opponents = [player[3] for player in game_state['others']]
    return nearest_distance(position, opponents)


# helper function to calculate the nearest distance to a list of coordinates
def nearest_distance(position, coordinates):
    distance = 100
    for coord in coordinates:
        if manhattan_distance(position, coord) < distance:
            distance = manhattan_distance(position, coord)
    return distance


def tiles_explored(position, game_state, tiles_visited):
    x_pos = position[0]
    y_pos = position[1]
    if game_state['field'][x_pos][y_pos] == 0:
        position = tuple(position)
        if tuple(position) not in tiles_visited:
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
            explosion = position + direction * radius
            # Clip the positions to stay within the field
            explosion = np.clip(explosion, 0, np.array(game_state['field'].shape) - 1)
            tile = game_state['field'][explosion[0], explosion[1]]
            if tile == -1:
                break
            if tile == 1:  # we will ge the crate destroyed
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


def backtracking(step, prev_feature_matrix):
    # Note: implement logic in rewards to punish LEFT_AND_Right==1 or Up_AND_Down==1
    backtracking_vector = []

    if prev_feature_matrix is not None:
        backtracking_vector = [direction[8] for direction in prev_feature_matrix]

    if backtracking_vector.count(1) == 2 or prev_feature_matrix is None:
        backtracking_vector = [0 for step in STEP]

    for i, step_i in enumerate(STEP):
        if step_i == step:
            backtracking_vector[i] = 1 if backtracking_vector[i] != 1 else 0

    return backtracking_vector


def manhattan_distance(point1, point2):
    """
    Calculate the Manhattan distance between two points.

    Args:
    point1 (tuple): The coordinates of the first point as (x, y).
    point2 (tuple): The coordinates of the second point as (x, y).

    Returns:
    int: The Manhattan distance between the two points.
    """
    try:
        x1, y1 = point1
        x2, y2 = point2

        return abs(x1 - x2) + abs(y1 - y2)
    except TypeError:
        print(f'point1: {point1}, point2: {point2}')
        raise TypeError


# def get_extended_explosion_map(game_state):
#     field = game_state['field']
#     bombs = game_state['bombs']
#     extended_explosion_map = np.full_like(field, -1)
#     # Add walls and crates to the explosion map
#     extended_explosion_map[field == -1] = -2
#     extended_explosion_map[field == 1] = -3

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

# def get_extended_explosion_map(game_state):
#     field = game_state['field']
#     bombs = game_state['bombs']
#     extended_explosion_map = np.full_like(field, -1)

#     # Add walls and crates to the explosion map
#     extended_explosion_map[field == -1] = -2
#     extended_explosion_map[field == 1] = -3

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
    extended_explosion_map[field == -1] = -2
    extended_explosion_map[field == 1] = -3

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


def get_danger_map(field, bombs):
    danger_map = np.zeros_like(field)
    extended_explosion_map = np.full_like(field, 10)
    max_danger = s.BOMB_POWER + 1
    for bomb_position, countdown in bombs:
        danger_map[bomb_position[0], bomb_position[1]] = max_danger
        extended_explosion_map[bomb_position[0], bomb_position[1]] = countdown
        for direction in STEP:
            for length in range(1, s.BOMB_POWER + 1):
                beam = direction * length + np.array(bomb_position)
                obj = field[beam[0], beam[1]]

                if obj == -1:
                    break
                if danger_map[beam[0], beam[1]] < max_danger - length:
                    danger_map[beam[0], beam[1]] = max_danger - length
                if extended_explosion_map[beam[0], beam[1]] > countdown:
                    extended_explosion_map[beam[0], beam[1]] = countdown
    return danger_map, extended_explosion_map


def get_danger(danger_map, position):
    return danger_map[position[0], position[1]]


def certain_death(game_state, move_coords, danger_map, extended_explosion_map):
    # Extract relevant information from the game state
    self_x, self_y = move_coords
    danger = danger_map[self_x, self_y]
    time_to_death = extended_explosion_map[self_x, self_y]
    # Get closest safe tile without explosion
    if danger == 0:
        return 0
    else:
        # Calculate steps to take to reach the closest safe tile considering walls and crates
        # if invalid_action((self_x, self_y), game_state) == 1 and time_to_death == 1:
        #     return 1
        # else:
        return 1 if (not move_to_save_space(game_state, (self_x, self_y), danger_map, time_to_death)) else 0


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
        extended_explosion_map[extended_explosion_map > 0] -= 1
        # Update explosion map -1 for each tile where not -1
        if extended_explosion_map[self_x, self_y] == 0:
            return False
        if (self_x, self_y) == tuple(tile) and extended_explosion_map[self_x, self_y] != 0:
            return True

        steps += 1


def move_to_save_space(game_state, position, danger_map, expl_timer):
    for i in range(expl_timer):
        valid_steps = get_valid_actions(position, game_state)
        min_step = valid_steps[
            np.argmin([danger_map[position[0] + step[0], position[1] + step[1]] for step in valid_steps])]
        position = position[0] + min_step[0], position[1] + min_step[1]
        if danger_map[position[0], position[1]] == 0:
            return True
    return False


def get_valid_actions(position, game_state):
    valid_actions = []

    for step in STEP:
        new_position = position + step

        # Check if the action is valid using the invalid_action method
        if invalid_action(new_position, game_state) == 1:
            valid_actions.append(step)

    return valid_actions


def backtracked_move(self, move_coords, game_state):
    # check in memory if move_coords is in the last 3 moves
    # Get the last 5 actions
    last_5_actions = [item.action for item in self.memory.get_last_n_items(5)]

    # Count occurrences of a specific action name (e.g., 'action_2')
    action_name = get_action_name(move_coords)
    count = last_5_actions.count(action_name)
    return count


def chain_inv_a(self, move_coords, game_state):
    pass


def get_action_name(coord_change):
    matching_indices = np.where(np.all(STEP == coord_change, axis=1))[0]
    if matching_indices.size > 0:
        return ACTION_NAME[matching_indices[0]]
    return None


def get_step(step_name):
    print(ACTION_NAME, step_name)
    actions = np.array(ACTION_NAME)
    matching_indices = np.where(actions == step_name)[0]
    if matching_indices.size > 0:
        return STEP[matching_indices[0]]
    return None


def valid_action(position, game_state):
    if game_state['field'][position[0]][position[1]] == 0:
        return True
    else:
        return False


def get_valid_action_strings(game_state):
    position = game_state['self'][3]
    valid_actions = []
    for idx, step in enumerate(STEP):
        new_position = position + step
        if valid_action(new_position, game_state):
            valid_actions.append(ACTION_NAME[idx])
