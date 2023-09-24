import numpy as np

# Hyper parameters -- TODO modify/optimize
TRANSITION_HISTORY_SIZE = 100000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# DISCOUNT_FACTOR is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LEARNING_RATE is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 64
# Discound factor or gamma
DISCOUNT_FACTOR = 0.5
EPS_START = 0.99
EPS_END = 0.1
STATIC_EPS = 0.1
EPS_DECAY_FACTOR = 1000000
TAU = 0.0001
LEARNING_RATE = 0.0001

NUM_EPISODES = 1000

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
STEP = np.array([(1,0), (-1,0), (0,1), (0,-1), (0,0)])


INPUT_SIZE = 35
HIDDEN_SIZE = 20
HIDDEN_SIZE_2 = 30
HIDDEN_SIZE_3 = 20
OUTPUT_SIZE = len(ACTIONS)