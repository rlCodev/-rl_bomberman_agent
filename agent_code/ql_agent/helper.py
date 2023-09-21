import numpy as np

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

def calculate_danger_level(game_state):
    """
    Calculate the danger level based on the distance to the closest tile that would be hit by an exploding bomb
    considering that bomb rays are stopped by walls.

    Args:
    game_state (dict): The current game state.

    Returns:
    float: A danger level between 0 and 1, where 0 means no danger, and 1 means imminent danger.
    """
    player_position = game_state['self'][3]
    field = game_state['field']
    bombs = game_state['bombs']

    if not bombs:
        return 0.0  # No bombs on the field, no danger.

    min_distance_to_bomb = float('inf')

    # Iterate through all active bombs to find the nearest tile hit by an exploding bomb.
    for bomb_position, countdown in bombs:
        distance = manhattan_distance(player_position, bomb_position)

        # Calculate the actual bomb ray radius in each direction
        up_radius, down_radius, left_radius, right_radius = 0, 0, 0, 0

        for r in range(1, countdown + 1):
            if bomb_position[1] - r >= 0 and field[bomb_position[0]][bomb_position[1] - r] != -1:
                up_radius = r
            else:
                break

        for r in range(1, countdown + 1):
            if bomb_position[1] + r < field.shape[1] and field[bomb_position[0]][bomb_position[1] + r] != -1:
                down_radius = r
            else:
                break

        for r in range(1, countdown + 1):
            if bomb_position[0] - r >= 0 and field[bomb_position[0] - r][bomb_position[1]] != -1:
                left_radius = r
            else:
                break

        for r in range(1, countdown + 1):
            if bomb_position[0] + r < field.shape[0] and field[bomb_position[0] + r][bomb_position[1]] != -1:
                right_radius = r
            else:
                break

        # Find the minimum distance to the nearest tile hit by any bomb in all directions
        min_radius = min(up_radius, down_radius, left_radius, right_radius)
        if distance <= min_radius:
            return 1.0  # Imminent danger, as player is within blast radius.

        if distance < min_distance_to_bomb:
            min_distance_to_bomb = distance

    # Scale the danger level between 0 and 1 based on the distance to the nearest tile hit by an exploding bomb.
    danger_level = min_distance_to_bomb / field.shape[0]  # Assuming field width == height
    return danger_level
