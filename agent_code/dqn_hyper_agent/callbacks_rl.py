import calendar
from collections import deque
import os
from agent_code.dqn_hyper_agent.features import state_to_features
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
        self.policy_net.load_state_dict(torch.load(f"models/{hp.CURRENT_MODEL}.pth"))
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
        self.logger.info("epsilon {} random number {}".format(self.epsilon, rand))
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