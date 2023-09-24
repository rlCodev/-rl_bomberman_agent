import numpy as np

from settings import BOMB_POWER, BOMB_TIMER, COLS, ROWS


def state_to_features(game_state: dict) -> np.array:
    # For example, you could construct several channels of equal shape, ...
    channels = []

    field_matrix = get_field_state(game_state)
    own_position = get_own_position(game_state)
    others_positions = get_others_positions(game_state)
    positions_desirability = get_position_desirability(game_state)

    field_matrix = (field_matrix + own_position + others_positions + positions_desirability)
    channels.append(field_matrix)

    positions_danger = get_position_danger(game_state)
    channels.append(positions_danger)

    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)


def state_to_features_old(game_state: dict) -> np.array:
    # For example, you could construct several channels of equal shape, ...
    channels = []

    field_matrix = get_field_state(game_state)
    channels.append(field_matrix)

    own_position = get_own_position(game_state)
    others_positions = get_others_positions(game_state)
    channels.append(own_position + others_positions)

    positions_danger = get_position_danger(game_state)
    channels.append(positions_danger)

    positions_desirability = get_position_desirability(game_state)
    channels.append(positions_desirability)

    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)


# -1.5: stone walls, 0: free tiles, 1.5: crates
def get_field_state(game_state: dict) -> np.array:
    field = game_state["field"]
    normalized_field = np.where(field == -1, -1.5, field)
    normalized_field = np.where(field == 1, 1.5, field)
    # normalized_field = np.where(normalized_field==-1, 0, normalized_field)
    return normalized_field


# 2: own position
def get_own_position(game_state: dict) -> np.array:
    own_position_x = game_state["self"][3][0]
    own_position_y = game_state["self"][3][1]
    own_position = np.zeros((game_state["field"].shape[0], game_state["field"].shape[1]))
    own_position[own_position_x][own_position_y] = 2
    return own_position

# -2: others position
def get_others_positions(game_state: dict) -> np.array:
    others_positions = np.zeros((game_state["field"].shape[0], game_state["field"].shape[1]))
    for agent in game_state["others"]:
        other_position_x = agent[3][0]
        other_position_y = agent[3][1]
        others_positions[other_position_x][other_position_y] = -2
    return others_positions

# Between -1 and approx. -0.14
def get_position_danger(game_state: dict) -> np.array:
    positions_danger = np.zeros((game_state["field"].shape[0], game_state["field"].shape[1]))
    for bomb in game_state["bombs"]:
        time_passed = BOMB_TIMER - bomb[1]
        time_needed_to_explode = BOMB_TIMER
        bomb_x = bomb[0][0]
        bomb_y = bomb[0][1]
        # set horizontally
        for number in range(1, BOMB_POWER+1):
            if bomb_y + number < ROWS:
                positions_danger[bomb_x][bomb_y + number] = _calculate_danger(time_passed, time_needed_to_explode,
                                                                              number)
            if bomb_y - number >= 0:
                positions_danger[bomb_x][bomb_y - number] = _calculate_danger(time_passed, time_needed_to_explode,
                                                                              number)
        # set vertically
        for number in range(1, BOMB_POWER+1):
            if bomb_x + number < COLS:
                positions_danger[bomb_x + number][bomb_y] = _calculate_danger(time_passed, time_needed_to_explode,
                                                                              number)
            if bomb_x - number >= 0:
                positions_danger[bomb_x - number][bomb_y] = _calculate_danger(time_passed, time_needed_to_explode,
                                                                              number)
        # set center
        positions_danger[bomb_x][bomb_y] = -(time_passed / time_needed_to_explode)
    return positions_danger


def _calculate_danger(time_passed, time_needed_to_explode, distance):
    return -np.round((time_passed / time_needed_to_explode) / np.sqrt(distance), 2)

# 1: coins
def get_position_desirability(game_state: dict) -> np.array:
    coin_positions = np.zeros((game_state["field"].shape[0], game_state["field"].shape[1]))
    for coin in game_state["coins"]:
        coin_x = coin[0]
        coin_y = coin[1]
        coin_positions[coin_x][coin_y] = 1
    return coin_positions