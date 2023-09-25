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
    test_stack = np.stack(feature_matrix)
    
    # TODO: Maybe add backtracking?
    #our agent needs to now whether bomb action is possible
    own_bomb = []
    if game_state['self'][2]:
        own_bomb.append(1)                      #if the player has a bomb, append 1 to our feature vector
    else:
        own_bomb.append(0)                      #if not, apppend 0
    
    stacked_channels = np.concatenate((stacked_channels, own_bomb)) #flatten feature vector


    return torch.tensor(stacked_channels, dtype=torch.float32), test_stack


def get_danger_map(field, bombs, explosion_map):
    '''
    returns a 2d array with danger values for each tile. 
    The danger is within [0,1], with 1 being certain death and 0 being no danger at all.
    '''
    danger_map = np.zeros_like(field, dtype = np.float32)
    bomb_timer = s.BOMB_TIMER

    for bomb_position, countdown in bombs:
        danger_map[bomb_position[0], bomb_position[1]] = (bomb_timer - countdown) / bomb_timer

        for direction in STEP:
            for length in range(1, s.BOMB_POWER + 1):
                # Set danger along beam of each bomb
                beam = direction * length + np.array(bomb_position)
                danger = (bomb_timer - countdown) / bomb_timer
                
                # If we hit a wall, break
                obj = field[beam[0], beam[1]]
                if obj == -1:
                    break

                # Else, set danger to highest danger value
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
    '''
    Returns adjusted danger map, where all obstacles are set to 1 and all free tiles with danger > 0 are set to 5
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
        distances.append(width_search_danger(adjusted_danger_map, player_pos + step, player_pos))      
    
    if np.sum(distances) == 0: return {tuple(step): 0 for step in STEP.tolist()}
    tile_count_ratio = distances * 1/max(distances)

    potentials = {tuple(step): ratio for step, ratio in zip(STEP, tile_count_ratio)}          
    return potentials

def get_delta_coins_potentials(player, adj_coin_map):
    steps_to_coins = []
    for step in STEP:
        steps_to_coins.append(bfs_position_to_object(player, player + step, adj_coin_map)) 
    steps_to_coins = np.array(steps_to_coins)
    if max(steps_to_coins) != 0:                                            #return for each neighbor: #steps to coin / #minimal steps to coin
        steps_to_coins = steps_to_coins * 1/max(steps_to_coins)
    
    return {tuple(step): ratio for step, ratio in zip(STEP, steps_to_coins)}

def get_neighbor_pos(player):
    '''
     returns positions of neigboring tiles of :param player: in the order (UP, DOWN, LEFT, RIGHT)
    '''
    neighbor_pos = []
    neighbor_pos.append((player[0], player[1] - 1))
    neighbor_pos.append((player[0], player[1] + 1))
    neighbor_pos.append((player[0] - 1, player[1]))
    neighbor_pos.append((player[0] + 1, player[1]))
    
    return np.array(neighbor_pos)

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

    # Check that we have opponents, otherwise we will always return zeros
    if len(crates_present) > 0:
        
        for segment in segments:
            x_min, x_max = segment[0]
            y_min, y_max = segment[1]
            # Find all the enemies within a segment
            crates_in_segment = np.where((crates_present[:,0] > x_min) & 
                                          (crates_present[:, 0] < x_max) & 
                                          (crates_present[:, 1] > y_min) & 
                                          (crates_present[:,0] < y_max))

            # If there are no enemies in segment, append 0
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


def get_bomb_position(game_state):
    '''
    :returns: array containing bomb positions
    '''
    bomb_position = []
    for i in range(len(game_state['bombs'])):
        bomb_position.append([game_state['bombs'][i][0][0], game_state['bombs'][i][0][1]] )
    return np.array(bomb_position)

def get_neighbor_value(game_state, channels, neighbor_pos, position_coins, i):
    '''
    Sets crate and wall values in channels for each neighbor 

    :param game_state: game state
    :param channels: feature vector, not flattened yet
    :param neighbor_pos: position of neighbor in form [x,y] (np.array)
    :param position_coins: coin position in form [[x,y],..] (np.array)

    :returns: channels. If neighbors are crates or walls, their according values in channels have been altered to 1
    '''
    
    #importing field values in neighbor position
    field_value = game_state['field'][neighbor_pos[i][0]][neighbor_pos[i][1]] 
    
    #finding a wall:
    if field_value == -1:
        channels[i][0] = 1
    
    #finding crate
    if field_value == 1:
        channels[i][1] = 1

    return channels

def get_neighbor_danger(game_state, channels, neighbor_pos, close_bomb_indices, bomb_position, player, explosion_map, i):
    '''
    Determines the danger values for each neighbor. Danger value is set if neighbor is ini reach of an explosion.

    :param game_state: game state
    :param channels: feature vector, not flattened yet
    :param neighbor_pos: position of neighbor in form [x,y] (np.array)
    :close_bomb_indices: indices for bombs in bomb_position that are located near the player
    :param bomb_position: position of all current bomb in form [[x,y],...] (np.array)
    :param player: player position [x,y] (np.array)
    :param i: index of neighbor in neighbors array. Needed for iteration

    :returns:   channels: danger value for each neighbor added in channels if neighbor is located on dangerous tile
                player_on_bomb: boolean indicating whether the player is located on a bomb (True) or not (False)
    '''
    #TODO define ->
    player_on_bomb = False

    #we need tuples as keys for the dictionary explosion_map to determine which tiles are dangerous
    bomb_tuples = [tuple(x) for x in bomb_position]
    
    for j in close_bomb_indices:                                                     #only look at close bombs             
        dangerous_tiles = np.array(exploding_tiles_map[bomb_tuples[j]])              #get all tiles exploding with close bombs
        if np.any(np.sum(np.abs(dangerous_tiles-neighbor_pos[i]), axis=1) == 0):     #if neighbor is on dangerous tile -> set danger value
            channels[i,6] = (4 - game_state['bombs'][j][1]) /4                       #danger value depends on the bomb timer, more dangerous if shortly before explosion
            

        if i == 3 and np.any(np.sum(np.abs(dangerous_tiles-player), axis=1) == 0):
            player_on_bomb = True                                                    #if player is on a dangerous tile set boolean

    #are there already exploding tiles in the neighbors (remember: explosions last for 1 step even after bomb exploded)
    if len(np.where(explosion_map != 0)[0]):                                        #check if there are current explosions
        if explosion_map[neighbor_pos[i,0],neighbor_pos[i,1]] != 0:
            channels[i,6] = 1                                                       #set highest danger value of 1 

    return channels, player_on_bomb

def width_search_danger(field, neighbor_pos, player_pos):
    '''
    short width searach algorithm to search for free tiles if player is located on a dangerous tile.

    :param field: game_state['field'] where crates = walls = 1, dangerous tiles = 5 and free tiles = 0
    :param neighbor_pos: position of neighbor [x,y] (np.array)
    :param player_pos: position of player [x,y] (np.array)

    :returns: inverted distance to closest free tile for neighbor in neighbor_pos. If no free tile is found returns 0 
    '''

    if field[tuple(neighbor_pos)] == CRATE_WALL_BOMB_PLAYER:                            #neighbor is bomb,crate,player or wall
        return 0                                                                # -> return 0

    tiles = []
    tiles_visited = [tuple(player_pos)]                                                      #history array that notes which tiles have been visited
    q = deque()                                                                 #deque used as queue
    q.append(tuple(neighbor_pos))

    while q:                                                                         #while there are elements in the queue
        pos = q.popleft()
        tiles_visited.append(pos)                                                      #get position of neighbor and add it to history

        for neighbor in get_neighbor_pos(pos):
            if field[neighbor[0], neighbor[1]] == FREE_TILE:                            #neighbor is on not exploding tile
                tiles.append(neighbor)                                          
                
            elif field[neighbor[0], neighbor[1]] == DANGER_TILE:                                        #check if neighbor is wall or crate, if not...
                if not np.any(np.sum(np.abs(tiles_visited - neighbor), axis=1) == 0):
                    q.append(neighbor)
                    tiles_visited.append(neighbor)

    tiles = np.array(tiles)
    if len(tiles) == 0:
        return 0
    if len(tiles) == 1:
        return 1/np.linalg.norm(tiles - player_pos, ord=1)
    
    
    dist = np.linalg.norm(tiles - player_pos, axis=1, ord=1) 
    return 1/min(dist)


def get_crate_dist(field, segments, player):
    '''
    Determine crate priority for each neighbor

    :param field: game_state['field']
    :param segments: segments of th field determined in get_segments 
    :param player: player position [x,y] (np.array)

    :returns:   densities: array with proportion of crates for each segment (segements are corresponding to each neighbor)
                crates_position: postion of all crates in form [[x,y],...]

    '''

    crates_position = np.array([np.where(field == 1)[0], np.where(field == 1)[1]]).T
    
    densities = []
    
    if crates_position.size > 0:
        for segment in segments:

            
            crates_in_segment = np.where((crates_position[:,0] > segment[0,0]) & (crates_position[:, 0] < segment[0,1]) & (crates_position[:, 1] > segment[1,0]) & (crates_position[:,0] < segment[1,1]))
            
            if len(crates_in_segment[0]) == 0:
                densities.append(0)
                continue
            
            d_crates = np.subtract(crates_position[crates_in_segment[0]], player)   
        
            dist_norm = np.linalg.norm(d_crates, axis = 1)
        
            density = len(dist_norm)/len(crates_position)
            
            densities.append(density)
        
        return densities, crates_position
    
    return densities, crates_position

def get_segments(player):
    '''
    Define segments relative to position of player.
    Order of segments: [UP, DOWN, LEFT, RIGHT].
    '''
    # each segment saves the upper and lower bound for x and y coordinates
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