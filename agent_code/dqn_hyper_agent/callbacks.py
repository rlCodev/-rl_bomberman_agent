import calendar
from collections import deque
import os
import pickle
import agent_code.ql_agent.helper as helper
from .DuelingDqn import DuelingDeepQNetwork
import numpy as np
import torch
import agent_code.dqn_hyper_agent.static_params as hp
from collections import deque
from random import shuffle
import numpy as np
from settings import COLS, ROWS


def setup(self):
    """Called once before a set of games to initialize data structures ethp.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    self.logger.debug('Successfully entered setup code')
    np.random.seed()

    global exploding_tiles_map
    with open('explosion_map.pt', 'rb') as file:
        exploding_tiles_map = pickle.load(file)
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    self.current_round = 0
    if self.train:
        self.logger.info("Setting up model from scratch...")

    if not os.path.isfile(f"models/{hp.CURRENT_MODEL}.pth") and not self.train:
        # Size of feature representation below
        self.policy_net = DuelingDeepQNetwork()
        self.target_net = DuelingDeepQNetwork()
        self.target_net.load_state_dict(self.policy_net.state_dict())
    elif os.path.isfile(f"models/{hp.CURRENT_MODEL}.pth") and not self.train:
        # Create an instance of the custom DuelingDeepQNetwork model
        self.policy_net = DuelingDeepQNetwork()
        self.policy_net.load_state_dict(
            torch.load(f"models/{hp.CURRENT_MODEL}.pth"))
    else:
        self.policy_net = DuelingDeepQNetwork()
        self.target_net = DuelingDeepQNetwork()


def act(self, game_state: dict) -> str:
    rand = np.random.random()
    if self.train and rand < self.epsilon:
        # Exploration function
        # or
        # action = np.argmax(np.random.multinomial(1, actions[0]))
        # or:
        action = np.random.randint(len(hp.ACTIONS))
        self.logger.info(
            "epsilon {} random number {}".format(self.epsilon, rand))
        self.logger.info("Randomly picked {}".format(hp.ACTIONS[action]))
    else:
        # state = np.array([state_to_features(game_state)])
        # actions = self.online_model.predict(state)
        # action = np.argmax(actions)
        # self.logger.info("Model picked {}".format(hp.ACTIONS[action]))

        # Assuming state_to_features returns a NumPy array, convert it to a PyTorch tensor
        state = state_to_features(game_state)

        # Replace self.online_model.predict with forward pass of the model
        # Assuming self.online_model is a PyTorch model
        with torch.no_grad():
            actions = self.policy_net(state)

        # Get the action index with the highest value
        action = int(torch.argmax(actions))

        self.logger.info("Model picked {}".format(hp.ACTIONS[action]))

    return hp.ACTIONS[action]


def state_to_features(game_state: dict):
    """
    Converts the game state to the input of your model, i.e.
    a feature vector.print(new_state_vector[0])

    Our model considers only the nearest neighbors of the player, assigning 7 Different values to each of them:
    [Wall, Crate, Coin_prio, Opponent_prio, Crate_prio, free_tile, Danger]

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """

    if game_state is None:
        return None

    # creating channels for each neighboring tile of our player
    channels = np.zeros((4, 7))

    # get player position:
    player = np.array(game_state['self'][3])

    # get field values
    field = np.array(game_state['field'])

    # get explosion map (remeber: explosions last for 2 steps)
    explosion_map = game_state['explosion_map']

    # getting bomb position from state
    bomb_position = get_bomb_position(game_state)

    # positions of neighboring tiles in the order (UP, DOWN, LEFT, RIGHT)
    neighbor_pos = get_neighbor_pos(player)

    # getting segments
    segments = get_segments(player)

    # getting position of other players and distance
    dist_others, other_position = get_player_dist(game_state, segments, player)

    # getting position of crates and distance
    dist_crates, crates_position = get_crate_dist(field, segments, player)

    # searching for near bombs:
    player_on_bomb = False  # needed later to determine if player is in explosion range

    close_bomb_indices = []  # consider only bombs that can be dangerous to our agent
    if len(bomb_position) != 0:
        bomb_distances = np.linalg.norm(
            np.subtract(bomb_position, player), axis=1)
        close_bomb_indices = np.where(bomb_distances <= 4)[0]

    # determine the minimal number of steps needed to get to a coin from each neighbor on
    position_coins = np.array(game_state['coins'])
    if position_coins.size > 0:

        coin_field = field.copy()  # parameter for our search_for_obj function.
        for coin in position_coins:
            coin_field[coin[0], coin[1]] = 3

        steps_to_coins = np.zeros(4)
        # get the number of steps from every neighbor to closest coin
        for j, neighbor in enumerate(neighbor_pos):
            steps_to_coins[j] = search_for_obj(player, neighbor, coin_field)

        if max(steps_to_coins) != 0:  # return for each neighbor: #steps to coin / #minimal steps to coin
            steps_to_coins = steps_to_coins * 1/max(steps_to_coins)

    # each direction is encoded by [Wall, Crate, Coin_prio, Opponent_prio, Crate_prio, free_tile, Danger]
    for i in range(np.shape(neighbor_pos)[0]):

        # is the neighbor a wall or crate?
        channels = get_neighbor_value(
            game_state, channels, neighbor_pos, position_coins, i)

        # finding bomb:
        if len(close_bomb_indices) != 0:
            # neighbor danger is described in get_neighbor_danger. give every neighbor its danger-value, player_on_bomb is a boolean (True if player on bomb)
            channels, player_on_bomb = get_neighbor_danger(
                game_state, channels, neighbor_pos, close_bomb_indices, bomb_position, player, explosion_map, i)

        # describing coin prio:
        if position_coins.size > 0:
            # for each neighbor note number of steps to closest coin
            channels[i, 2] = steps_to_coins[i]

    # describing pritority for next free tile if a bomb has been placed
    if player_on_bomb:  # parameter set in get_neighbor_danger
        tile_count = find_closest_free_tile(
            game_state, player, close_bomb_indices, bomb_position, neighbor_pos)
        for j in range(4):
            channels[j, 5] = tile_count[j]

    # describing distance to other players
    if other_position.size > 0:
        for i in range(len(dist_others)):  # set opponent_prio for each neighbor
            channels[i, 3] = dist_others[i]

    # describing distance to crates
    if crates_position.size > 0:
        for i in range(len(dist_crates)):  # set crate_prio for each neighbor
            channels[i, 4] = dist_crates[i]

    # combining current channels:
    stacked_channels = np.stack(channels).reshape(-1)

    # our agent needs to now whether bomb action is possible
    own_bomb = []
    if game_state['self'][2]:
        # if the player has a bomb, append 1 to our feature vector
        own_bomb.append(1)
    else:
        own_bomb.append(0)  # if not, apppend 0

    stacked_channels = np.concatenate(
        (stacked_channels, own_bomb))  # flatten feature vector

    return torch.tensor(stacked_channels, dtype=torch.float32)


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


def get_player_dist(game_state, segments, player):
    '''
    Determine the distance to the closest opponent for each neighbor

    :param game_state:The same object that is passed to all of your callbacks
    :param segments: segments of the game field determined in 'get_segments'
    :param player: player position in form [x,y]

    :returns: array with distances to closest opponent for each neighbor, array with the opponents positions
    '''

    # getting position of other players from state:
    other_position = []
    for i in range(len(game_state['others'])):
        other_position.append(
            [game_state['others'][i][3][0], game_state['others'][i][3][1]])
    other_position = np.array(other_position)

    distances = []

    # distance from others to player
    if other_position.size > 0:
        for segment in segments:

            others_in_segment = np.where((other_position[:, 0] > segment[0, 0]) & (other_position[:, 0] < segment[0, 1]) & (
                other_position[:, 1] > segment[1, 0]) & (other_position[:, 0] < segment[1, 1]))

            if len(others_in_segment[0]) == 0:
                distances.append(0)
                continue

            # determine distance to our player for all opponents in the segment
            d_others = np.subtract(
                other_position[others_in_segment[0]], player)

            dist_norm = np.linalg.norm(d_others, axis=1)

            # use distance of closest neighbor
            dist_closest = 2 / (1 + min(dist_norm))
            distances.append(dist_closest)

        return distances, other_position

    return distances, other_position


def get_bomb_position(game_state):
    '''
    :returns: array containing bomb positions
    '''
    bomb_position = []
    for i in range(len(game_state['bombs'])):
        bomb_position.append(
            [game_state['bombs'][i][0][0], game_state['bombs'][i][0][1]])
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

    # importing field values in neighbor position
    field_value = game_state['field'][neighbor_pos[i][0]][neighbor_pos[i][1]]

    # finding a wall:
    if field_value == -1:
        channels[i][0] = 1

    # finding crate
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

    player_on_bomb = False

    # we need tuples as keys for the dictionary explosion_map to determine which tiles are dangerous
    bomb_tuples = [tuple(x) for x in bomb_position]

    for j in close_bomb_indices:  # only look at close bombs
        # get all tiles exploding with close bombs
        dangerous_tiles = np.array(exploding_tiles_map[bomb_tuples[j]])
        # if neighbor is on dangerous tile -> set danger value
        if np.any(np.sum(np.abs(dangerous_tiles-neighbor_pos[i]), axis=1) == 0):
            # danger value depends on the bomb timer, more dangerous if shortly before explosion
            channels[i, 6] = (4 - game_state['bombs'][j][1]) / 4

        if i == 3 and np.any(np.sum(np.abs(dangerous_tiles-player), axis=1) == 0):
            player_on_bomb = True  # if player is on a dangerous tile set boolean

    # are there already exploding tiles in the neighbors (remember: explosions last for 1 step even after bomb exploded)
    # check if there are current explosions
    if len(np.where(explosion_map != 0)[0]):
        if explosion_map[neighbor_pos[i, 0], neighbor_pos[i, 1]] != 0:
            channels[i, 6] = 1  # set highest danger value of 1

    return channels, player_on_bomb


def find_closest_free_tile(game_state, player_pos, close_bomb_indices, bomb_position, neighbor_pos):
    '''
    For each neighbor determine the priority for bomb dodging if our player stands on a dangerous tile. 
    Priorities are determined by min(distances to free tile from all neighbors)/(shortest distance to free tile for that neighbor). (min = non-zero-min)
    Free tiles are 0 in game_state['field'] and not dangerous tiles. 
    '''

    field = game_state['field']  # get field
    # set crates and walls to 1 for convenience
    field = np.where(field == -1, 1, field)
    # tuples to use as keys for the exploding_tiles_map
    bomb_tuples = [tuple(x) for x in bomb_position]

    for j in close_bomb_indices:  # only look at close bombs
        # get all tiles exploding with close bombs
        dangerous_tiles = np.array(exploding_tiles_map[bomb_tuples[j]])
        for tile in dangerous_tiles:
            if field[tile[0], tile[1]] == 0:
                # set all dangerous tiles to 2 in field
                field[tile[0], tile[1]] = 2

    # since other players and bombs block movement, look at them as walls
    for enemy in game_state['others']:
        field[enemy[3][0], enemy[3][1]] = 1
    for bomb in bomb_position:
        field[bomb[0], bomb[1]] = 1

    tile_count = np.zeros(4)
    # for each neighbor set inverted distance to closest free tile in tile_count
    for j, neighbor in enumerate(neighbor_pos):
        tile_count[j] = width_search_danger(field, neighbor, player_pos)

    if np.sum(tile_count) == 0:
        return tile_count  # no free tile was found -> our agent is doomed

    # multiply with minimal distance found. Note that width_search_danger retruns inverted distances, so
    tile_count_ratio = tile_count * 1/max(tile_count)
    #  1/max(tile_count) = 1/ (1/min(dist)) = min(dist)

    return tile_count_ratio


def width_search_danger(field, neighbor_pos, player_pos):
    '''
    short width searach algorithm to search for free tiles if player is located on a dangerous tile.

    :param field: game_state['field'] where crates = walls = 1, dangerous tiles = 2 and free tiles = 0
    :param neighbor_pos: position of neighbor [x,y] (np.array)
    :param player_pos: position of player [x,y] (np.array)

    :returns: inverted distance to closest free tile for neighbor in neighbor_pos. If no free tile is found returns 0 
    '''

    if field[neighbor_pos[0], neighbor_pos[1]] == 1:  # neighbor is bomb,crate,player or wall
        # -> return 0
        return 0

    tiles = []
    # history array that notes which tiles have been visited
    history = [player_pos]
    q = deque()  # deque used as queue
    q.append(neighbor_pos)

    while len(q) > 0:  # while there are elements in the queue

        pos = q.popleft()
        history.append(pos)  # get position of neighbor and add it to history

        neighbors = get_neighbor_pos(pos)  # get neighbor tiles of neighbor

        for neighbor in neighbors:

            if field[neighbor[0], neighbor[1]] == 0:  # neighbor is on not exploding tile
                tiles.append(neighbor)

            # check if neighbor is wall or crate, if not...
            if field[neighbor[0], neighbor[1]] == 2:

                # if neighbor is already in the q, dont append
                if not np.any(np.sum(np.abs(history - neighbor), axis=1) == 0):

                    # neighbor is not yet in q
                    q.append(neighbor)
                    history.append(neighbor)

    # no free tiles were found for this neighbor
    if tiles == []:
        return 0
    # one free tile was found
    if len(tiles) == 1:
        return 1/np.linalg.norm(tiles - player_pos)

    # multiple free tiles were found
    dist = np.linalg.norm(tiles - player_pos, axis=1)
    # -> return inverted distance to closest free tile
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

    crates_position = np.array([np.where(field == 1)[0], np.where(
        field == 1)[1]]).T  # get positiosn of crates

    densities = []  # array that holds the density for each segment

    if crates_position.size > 0:
        for segment in segments:

            crates_in_segment = np.where((crates_position[:, 0] > segment[0, 0]) & (crates_position[:, 0] < segment[0, 1]) & (
                crates_position[:, 1] > segment[1, 0]) & (crates_position[:, 0] < segment[1, 1]))

            # no crates in segment -> conitnue
            if len(crates_in_segment[0]) == 0:
                densities.append(0)
                continue

            d_crates = np.subtract(
                crates_position[crates_in_segment[0]], player)

            dist_norm = np.linalg.norm(d_crates, axis=1)

            # get ratio of (crates in this segment)/(all crates)
            density = len(dist_norm)/len(crates_position)

            densities.append(density)

        return densities, crates_position

    return densities, crates_position


def get_segments(player):
    '''
    Determine segments of the field. 
    Segments are segments of the field according to our player. For example left_half is the game_state['field'] but reduced to everything
    located to the left of the player position. 

    :param player: player position

    :returns: segments in order [UP, DOWN, LEFT, RIGHT] 
    '''

    left_half = np.array([[0, player[0]], [0, 17]])

    right_half = np.array([[player[0], 17], [0, 17]])

    up_half = np.array([[0, 17], [0, player[1]]])

    low_half = np.array([[0, 17], [player[1], 17]])

    segments = np.array([up_half, low_half, left_half, right_half])

    return segments


def search_for_obj(player_pos, neighbor_pos, field):
    '''
    Search for certain object in game_state['field'] via width-first search. 

    :param player_pos: postion of play [x,y] (np.array)
    :param neighbor_pos: position of neighbor [x,y] (np.array)
    :param field: game_state['field']. The object to be searched for needs to be set to 3 in this field before calling the function

    :returns: inverted number of steps to closest object
    '''

    if field[neighbor_pos[0], neighbor_pos[1]] != 0:
        if field[neighbor_pos[0], neighbor_pos[1]] == 3:
            return 1  # neighbor holds object -> one step to it
        return 0  # neighbor is wall or crate -> return 0

    parents = [None]*17**2  # parent array needed to determine shortest path

    # indices for the parents-array are flattened coordinates: i=17*x+y
    flat_neighbor = 17 * neighbor_pos[0] + neighbor_pos[1]
    flat_player = 17 * player_pos[0] + player_pos[1]
    flat_obj = None  # object index is initialized with None

    # parents for player and neighbor are themselves
    parents[flat_neighbor] = flat_neighbor
    parents[flat_player] = flat_player

    q = deque()  # deque used as queue
    q.append(neighbor_pos)

    while len(q) > 0:

        node = q.popleft()
        if field[node[0], node[1]] == 3:  # object found
            # set flat coordinates. When obj is found it needs to have a parent
            flat_obj = 17 * node[0] + node[1]
            break

        for neighbor in get_neighbor_pos(node):

            # neighbor is not crate or wall
            if field[neighbor[0], neighbor[1]] == 0 or field[neighbor[0], neighbor[1]] == 3:

                # neighbor does not yet have a parent
                if parents[17 * neighbor[0] + neighbor[1]] is None:

                    parents[17 * neighbor[0] + neighbor[1]
                            ] = 17 * node[0] + node[1]
                    q.append(neighbor)

    if flat_obj == None:  # no object was found
        return 0

    path = [flat_obj]  # object was found -> determine length of path
    while path[-1] != flat_neighbor:

        path.append(parents[path[-1]])

    return 1/len(path)
