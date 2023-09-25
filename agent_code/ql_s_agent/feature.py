import numpy as np
from collections import deque
import settings as s
import torch

STEP = np.array([(0,-1), (0,1), (-1,0), (1,0)])
ACTION_NAME = ['UP', 'DOWN', 'LEFT', 'RIGHT']
CRATE_WALL_BOMB_PLAYER = 1 
FREE_TILE = 0
DANGER_TILE = 5

def state_to_features(game_state: dict) -> np.array:
    """

    """
    if game_state is None:
        return None

    # This will be our feature matrix. Each column contains the information vector for one potential move.
    feature_matrix = []

    position = game_state['self'][3]

    # Danger map: [0,1], the higher the value, the more dangerous (1 = explosion) | for each tile
    danger_map = get_danger_map(game_state['field'], game_state['bombs'], game_state['explosion_map'])
    adjusted_danger_map = get_adjusted_danger_map(game_state, danger_map)
    coin_map = get_coin_map(game_state)
    adjusted_coin_map = get_adjusted_coin_map(game_state, coin_map)
    crate_map = get_crate_map(game_state['field'])
    crates_positions = np.argwhere(crate_map == 1)

    # For each possible move, find the closest safe tile -> dictionary with step as key
    if danger_map[position[0], position[1]] > 0:
        safety_potentials = get_safety_potentials(adjusted_danger_map, position)
    else:
        safety_potentials = {tuple(step): 0 for step in STEP.tolist()}

    # Get segments for defining potentials to nearest opponent and crate:
    segments = get_segments(position)

    # Get distance potentials to nearest opponent
    delta_opponents_potentials = get_delta_opponents_potentials(game_state, segments, position)

    # Get distance potentials to nearest crate
    delta_crates_potentials = get_delta_crates_potentials(game_state, segments, position, crates_positions)

    # Get distance potentials to nearest coin
    if(game_state['coins'] != []):
        delta_coins_potentials = get_delta_coins_potentials(position, adjusted_coin_map)
    else:
        delta_coins_potentials = {tuple(step): 0 for step in STEP.tolist()}

    for step in STEP:
        # For each step we define a vector and append it to our matrix at the end.
        move_feature_vector = np.zeros(8)

        pos_after_move = position + step

        # Check for wall
        if game_state['field'][pos_after_move[0], pos_after_move[1]] == -1:
            move_feature_vector[0] = 1
             
        # Check for coin
        if coin_map[pos_after_move[0], pos_after_move[1]] == 1:
            move_feature_vector[1] = 1

        # Check for crate
        if game_state['field'][pos_after_move[0], pos_after_move[1]] == 1:
            move_feature_vector[2] = 1

        # Check for danger
        move_feature_vector[3] = danger_map[pos_after_move[0], pos_after_move[1]]

        # Check for nearest safe tile
        move_feature_vector[4] = safety_potentials[tuple(step)]

        # Check for distance to coin
        move_feature_vector[5] = delta_coins_potentials[tuple(step)]

        # Check for distance to crate
        move_feature_vector[6] = delta_crates_potentials[tuple(step)]

        # Check for distance to enemy
        move_feature_vector[7] = delta_opponents_potentials[tuple(step)]

        feature_matrix.append(move_feature_vector)

    #combining current channels:
    stacked_channels = np.stack(feature_matrix).reshape(-1)
    
    #our agent needs to now whether bomb action is possible
    own_bomb = []
    if game_state['self'][2]:
        own_bomb.append(1)
    else:
        own_bomb.append(0)
    
    stacked_channels = np.concatenate((stacked_channels, own_bomb))


    return torch.tensor(stacked_channels, dtype=torch.float32)


def get_danger_map(field, bombs, explosion_map):

    danger_map = np.zeros_like(field, dtype = np.float32)
    bomb_timer = s.BOMB_TIMER

    for bomb_position, countdown in bombs:
        danger_map[bomb_position[0], bomb_position[1]] = (bomb_timer - countdown) / bomb_timer

        for direction in STEP:
            for length in range(1, s.BOMB_POWER + 1):
                beam = direction * length + np.array(bomb_position)
                danger = (bomb_timer - countdown) / bomb_timer
                
                obj = field[beam[0], beam[1]]
                if obj == -1:
                    break

                if danger_map[beam[0], beam[1]] < danger:
                    danger_map[beam[0], beam[1]] = danger
    
    for (x,y) in np.argwhere(explosion_map > 0):
        danger_map[x,y] = 1

    return danger_map

def get_coin_map(field):
    '''
    Returns a replica of the field, with 1 where coins are and 0 otherwise
    '''
    coin_map = np.zeros_like(field['field'], dtype = np.float32)
    if field['coins'] == []:
        return coin_map
    else:
        for coin in field['coins']:
            coin_map[coin[0], coin[1]] = 1
    return coin_map

def get_crate_map(field):
    crate_map = np.zeros_like(field, dtype = np.float32)
    crate_map[field == 1] = 1
    return crate_map
    

def get_adjusted_danger_map(game_state, danger_map):

    # Define new field, where all obstacles are set to 1
    field_with_obstacles = game_state['field'].copy()                                         
    field_with_obstacles = np.where(field_with_obstacles == -1 , 1, field_with_obstacles)
    if game_state['others'] != []:
        for enemy in game_state['others']:
            field_with_obstacles[enemy[3][0],enemy[3][1]] = 1
    if game_state['bombs'] != []:
        for bomb in game_state['bombs']:
            field_with_obstacles[bomb[0][0],bomb[0][1]] = 1

    # Set all tiles, where danger is > 0 to 5
    danger_map = np.where(danger_map > 0, 5, danger_map)
    
    # Combine field_with_obstacles and danger_map
    combined_map = np.where(field_with_obstacles == 0, danger_map, field_with_obstacles)
    return combined_map

def get_adjusted_coin_map(game_state, coin_map):
    '''
    Returns adjusted coin map, where all obstacles are set to 1 and all free tiles with coin = 1 are set to 3
    '''
    # Define new field, where all obstacles are set to 1
    field_with_obstacles = game_state['field'].copy()                                         
    field_with_obstacles = np.where(field_with_obstacles == -1 , 1, field_with_obstacles)
    if game_state['others'] != []:
        for enemy in game_state['others']:
            field_with_obstacles[enemy[3][0],enemy[3][1]] = 1
    if game_state['bombs'] != []:
        for bomb in game_state['bombs']:
            field_with_obstacles[bomb[0][0],bomb[0][1]] = 1

    # Set all tiles, where coin = 1 to 3
    coin_map = np.where(coin_map == 1, 3, coin_map)
    
    # Combine field_with_obstacles and coin_map
    combined_map = np.where(field_with_obstacles == 0, coin_map, field_with_obstacles)
    return combined_map

def get_safety_potentials(adjusted_danger_map, player_pos):
    '''
    Returns the distance to the closest safe tile
    '''
    distances = []

    for step in STEP:
        distances.append(danger_search(adjusted_danger_map, player_pos + step, player_pos))      
    
    if np.sum(distances) == 0: return {tuple(step): 0 for step in STEP.tolist()}
    tile_count_ratio = distances * 1/max(distances)

    potentials = {tuple(step): ratio for step, ratio in zip(STEP, tile_count_ratio)}          
    return potentials

def get_delta_coins_potentials(player, adj_coin_map):
    steps_to_coins = []
    for step in STEP:
        steps_to_coins.append(bfs_position_to_object(player, player + step, adj_coin_map)) 
    steps_to_coins = np.array(steps_to_coins)
    if max(steps_to_coins) != 0:
        steps_to_coins = steps_to_coins * 1/max(steps_to_coins)
    
    return {tuple(step): ratio for step, ratio in zip(STEP, steps_to_coins)}

def get_delta_opponents_potentials(game_state, segments, player):
    '''
    For each segment, find the nearest opponent and calculate distance.
    Return map, mapping each step to the distance to the nearest opponent.
    '''
    
    #getting position of other players from state:
    opponents_present = game_state['others'] != []
    enemies_pos = []
    distances = []

    # Check that we have opponents, otherwise we will always return zeros
    if opponents_present:
        for opponent in game_state['others']:
            enemies_pos.append(opponent[3]) 
        enemies_pos = np.array(enemies_pos)
        
        for segment in segments:
            x_min, x_max = segment[0]
            y_min, y_max = segment[1]
            # Find all the enemies within a segment
            enemies_in_segment = np.where((enemies_pos[:,0] > x_min) & 
                                          (enemies_pos[:, 0] < x_max) & 
                                          (enemies_pos[:, 1] > y_min) & 
                                          (enemies_pos[:,0] < y_max))

            # If there are no enemies in segment, append 0
            if len(enemies_in_segment[0]) == 0:
                distances.append(0)
            else:
                distance_to_enemies = np.subtract(enemies_pos[enemies_in_segment[0]], player)
                normalized_distance = np.linalg.norm(distance_to_enemies, axis = 1)
                min_distance = 2/ (1 + min(normalized_distance))
                distances.append(min_distance)
    else:
        distances = np.zeros(4)

    potentials = {tuple(step): ratio for step, ratio in zip(STEP, distances)}
    return potentials


def get_delta_crates_potentials(game_state, segments, player, crates_present):
    '''
    For each segment, find the nearest crate and calculate distance.
    Return map, mapping each step to the distance to the nearest crate.
    '''

    distances = []

    if len(crates_present) > 0:
        
        for segment in segments:
            x_min, x_max = segment[0]
            y_min, y_max = segment[1]
            crates_in_segment = np.where((crates_present[:,0] > x_min) & 
                                          (crates_present[:, 0] < x_max) & 
                                          (crates_present[:, 1] > y_min) & 
                                          (crates_present[:,0] < y_max))

            if len(crates_in_segment[0]) == 0:
                distances.append(0)
            else:
                distance_to_crates = np.subtract(crates_present[crates_in_segment[0]], player)
                normalized_distance = np.linalg.norm(distance_to_crates, axis = 1)
                min_distance = len(normalized_distance)/len(crates_present)
                distances.append(min_distance)
    else:
        distances = np.zeros(4)

    potentials = {tuple(step): ratio for step, ratio in zip(STEP, distances)}
    return potentials

def danger_search(field, neighbor_pos, player_pos):

    if field[tuple(neighbor_pos)] == CRATE_WALL_BOMB_PLAYER:                         
        return 0                                                              

    tiles = []
    tiles_visited = [tuple(player_pos)]                                                  
    queue = deque()                                                               
    queue.append(tuple(neighbor_pos))

    while queue:                                                                         
        pos = queue.popleft()
        tiles_visited.append(pos)

        for step in STEP:
            neighbor = pos + step
            if field[neighbor[0], neighbor[1]] == FREE_TILE:
                tiles.append(neighbor)                                          
                
            elif field[neighbor[0], neighbor[1]] == DANGER_TILE:
                if not np.any(np.sum(np.abs(tiles_visited - neighbor), axis=1) == 0):
                    queue.append(neighbor)
                    tiles_visited.append(neighbor)


    tiles = np.array(tiles)
    if len(tiles) == 0:
        return 0
    if len(tiles) == 1:
        return 1/np.linalg.norm(tiles - player_pos, ord=1)
    
    
    dist = np.linalg.norm(tiles - player_pos, axis=1, ord=1) 
    return 1/min(dist)

def get_segments(player):
    '''
    Define segments relative to position of player.
    Order of segments: [UP, DOWN, LEFT, RIGHT].
    '''
    right_wall = s.COLS
    lower_wall = s.ROWS

    upper_segment = np.array([[0, right_wall], [0, player[1]]])
    lower_segment = np.array([[0, right_wall], [player[1], lower_wall]])
    left_segment =  np.array([[0, player[0]], [0, lower_wall]])
    right_segment = np.array([[player[0], right_wall], [0, lower_wall]])
    
    segments = np.array([upper_segment, lower_segment, left_segment, right_segment])

    return segments

def bfs_position_to_object(player_pos, neighbouring_tile, field):
    
    if field[neighbouring_tile[0],neighbouring_tile[1]] != 0 :            
        if field[neighbouring_tile[0],neighbouring_tile[1]] == 3: 
            return 1  
        return 0
    
    rows, cols = field.shape
    neighbor_flattened = cols * neighbouring_tile[0] + neighbouring_tile[1]
    player_flattened = cols * player_pos[0] + player_pos[1]
    flat_obj = None

    parents = [None] * (rows * cols) 

    parents[neighbor_flattened] = neighbor_flattened
    parents[player_flattened] = player_flattened

    q = deque()
    q.append(neighbouring_tile)              
 
    while len(q) > 0:
        node = q.popleft()     
        if field[node[0],node[1]] == 3:
            flat_obj = 17 *node[0] + node[1]
            break   

        for step in STEP:
            new_pos = node + step
            if 0 <= new_pos[0] < rows and 0 <= new_pos[1] < cols and \
               (field[new_pos[0], new_pos[1]] == 0 or field[new_pos[0], new_pos[1]] == 3):
                if parents[cols * new_pos[0] + new_pos[1]] is None:
                    parents[cols * new_pos[0] + new_pos[1]] = cols * node[0] + node[1]
                    q.append(new_pos)

    if flat_obj == None:
        return 0                  
    
    path = [flat_obj]
    while path[-1] != neighbor_flattened:
        path.append(parents[path[-1]])
    
    return 1/len(path)