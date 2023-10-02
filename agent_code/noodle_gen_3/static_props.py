# Discound factor or gamma
DISCOUNT_FACTOR = 0.95
EPS_START = 0.99
EPS_END = 0.0
LEARNING_RATE = 0.0001

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

#Number Episodes has to match the number of episodes set in the json
NUMBER_EPISODE = 400
INPUT_SIZE = 33
HIDDEN_LAYER_1_SIZE = 256
HIDDEN_LAYER_2_SIZE = 256
OUTPUT_SIZE = len(ACTIONS)

# Enable training plots here:
PLOT_TRAINING = False
ENABLE_LOGS = False