from collections import namedtuple, deque
import torch
import pickle
from typing import List

import events as e
from .callbacks import state_to_features
import numpy as np
from torch import nn
from torch.optim import SGD
from .MLP import CustomMLP

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- TODO modify/optimize
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

DISCOUNT_FACTOR = 0.95
EPS = 0.5
EPS_DECAY_FACTOR = 0.999
LEARNING_RATE = 0.01 # crucial against exploding gradients
NUM_EPISODES = 500

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Set up an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    # Set hyperparameters for agent
    self.discount_factor = DISCOUNT_FACTOR
    self.eps = EPS
    self.eps_decay_factor = EPS_DECAY_FACTOR
    self.learning_rate = LEARNING_RATE
    self.num_episodes = NUM_EPISODES

    self.exploration_rate = 0.9
    self.exploration_rate_decay = 0.99999975
    self.exploration_rate_min = 0.1

    # Use SGD optimizer and Mean Squared Error loss function
    self.optimizer = SGD(self.model.parameters(), lr=self.learning_rate)
    self.loss_function = nn.MSELoss()


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # # Idea: Add your own events to hand out rewards
    # if ...:
    #     events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py

    reward = reward_from_events(self, events)
    self.transitions.append(
        Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward))

    # Add batch dimension
    old_state_features = torch.tensor(state_to_features(old_game_state), dtype=torch.float32).unsqueeze(0)
    new_state_features = torch.tensor(state_to_features(new_game_state), dtype=torch.float32).unsqueeze(0)

    q_values_old = self.model(old_state_features)
    q_values_new = self.model(new_state_features)
    max_q_new = torch.max(q_values_new).detach()  # Max Q-value for the next state
    print(q_values_old)
    # Calculate target Q-value using Q-learning update
    target_q_value = reward + self.discount_factor * max_q_new

    # Compute the loss
    action_index = ACTIONS.index(self_action)
    loss = self.loss_function(q_values_old[0][action_index], target_q_value)

    # Backpropagation and optimization
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(
        Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    # with open("my-saved-model.pt", "wb") as file:
    #     pickle.dump(self.model, file)

    # Calculate reward from events
    reward = reward_from_events(self, events)
    torch.save(self.model, 'custom_mlp_model.pth')


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    #       Kill a player 100
    #       Break a wall 30
    #       Perform action -1
    #       Perform impossible action -2
    #       Die -300
    game_rewards = {
        e.COIN_COLLECTED: 50,
        e.CRATE_DESTROYED: 30,
        e.INVALID_ACTION: -2,
        e.KILLED_OPPONENT: 100,
        e.GOT_KILLED: -300
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
