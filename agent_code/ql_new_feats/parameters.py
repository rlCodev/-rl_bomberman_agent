import events

# Architecture
CURRENT_MODEL = 'models/26_03_2021_21_38_41'  # 31.000 Episoden trainiert
CURRENT_BUFFER = 'buffers/26_03_2021_21_38_41'
INPUT_SIZE = 578
HIDDEN_LAYER_1_SIZE = 2048
HIDDEN_LAYER_2_SIZE = 2048
OUTPUT_SIZE = 6
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Hyperparameters
LEARNING_RATE = 0.01

# Prioritized Experience Buffer
ALPHA = 0.65
BETA = 0.4
TD_EPSILON = 0.01
DISCOUNT_FACTOR = 0.001
NUMBER_EPISODE = 100

EPS_START = 1  # 0.3 When using PER

UPDATE_TARGET_MODEL_EVERY = 5
UPDATE_TENSORBOARD_EVERY = 5
SAVE_MODEL_EVERY = 100

# RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Rewards
# TODO: Add custom events for reward shaping
REWARDS = {
    events.MOVED_LEFT: 0,
    events.MOVED_RIGHT: 0,
    events.MOVED_UP: 0,
    events.MOVED_DOWN: 0,
    events.WAITED: -1,
    events.INVALID_ACTION: -1,
    events.BOMB_DROPPED: 0.4,  # Reward higher than collecting coins
    events.BOMB_EXPLODED: 0,
    events.CRATE_DESTROYED: 0.7,
    events.COIN_FOUND: 0,
    events.COIN_COLLECTED: 0.2,
    events.KILLED_OPPONENT: 1,
    events.KILLED_SELF: -1,
    events.GOT_KILLED: -1,
    events.OPPONENT_ELIMINATED: 1,
    events.SURVIVED_ROUND: 0,
}