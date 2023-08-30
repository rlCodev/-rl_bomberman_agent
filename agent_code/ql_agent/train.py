from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features
import numpy as np

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

DISCOUNT_FACTOR = 0.95
EPS = 0.5
EPS_DECAY_FACTOR = 0.999
LEARNING_RATE = 0.8
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
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    # Set hyperparameters for agent
    self.discount_factor = DISCOUNT_FACTOR
    self.eps = EPS
    self.eps_decay_factor = EPS_DECAY_FACTOR
    self.learning_rate = LEARNING_RATE
    self.num_episodes = NUM_EPISODES

    self.exploration_rate = 1
    self.exploration_rate_decay = 0.99999975
    self.exploration_rate_min = 0.1



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
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))


    # From turorial:
    # q_table[state, action] += reward + learning_rate * (discount_factor * np.max(q_table[new_state, :]) - q_table[state, action])
    # Calculate reward from events
    reward = reward_from_events(self, events)

    # Update Q-table using Q-learning
    old_state = state_to_features(old_game_state)
    new_state = state_to_features(new_game_state)
    old_q_value = self.q_table[old_state, ACTIONS.index(self_action)]
    max_future_q = np.max(self.q_table[new_state, :])
    new_q_value = (1 - self.learning_rate) * old_q_value + self.learning_rate * (reward + self.discount_factor * max_future_q)
    self.q_table[old_state, ACTIONS.index(self_action)] = new_q_value


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
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    # with open("my-saved-model.pt", "wb") as file:
    #     pickle.dump(self.model, file)

    # Calculate reward from events
    reward = reward_from_events(self, events)

    # Update Q-table using Q-learning for the last action-state pair
    old_state = state_to_features(last_game_state)
    old_q_value = self.q_table[old_state, ACTIONS.index(last_action)]
    new_q_value = old_q_value + self.learning_rate * (reward - old_q_value)
    self.q_table[old_state, ACTIONS.index(last_action)] = new_q_value

    # Store the Q-table
    with open("coin-collector-qtable.pt", "wb") as file:
        pickle.dump(self.q_table, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 5,
        e.GOT_KILLED: -15,
        e.INVALID_ACTION: -10
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
