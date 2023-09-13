from collections import namedtuple, deque
import math
from matplotlib import pyplot as plt
import torch
import pickle
from typing import List
import os
import events as e
from .callbacks import state_to_features
import numpy as np
from torch import nn
from torch.optim import AdamW, SGD
from .MLP import MLP
import random
from .utils import action_index_to_string, action_string_to_index

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- TODO modify/optimize
TRANSITION_HISTORY_SIZE = 20  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# DISCOUNT_FACTOR is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LEARNING_RATE is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 12
# Discound factor or gamma
DISCOUNT_FACTOR = 0.9
NUMBER_EPISODE = 200
EPS_START = 0.9
EPS_END = 0.2
STATIC_EPS = 0.1
EPS_DECAY_FACTOR = 10000
TAU = 0.001
LEARNING_RATE = 0.0001

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    # Use SGD optimizer and Mean Squared Error loss function
    # self.optimizer = SGD(self.model.parameters(), lr=self.learning_rate)
    self.loss_function = nn.MSELoss()
    self.losses = []

    # Setup models
    input_size = 1445
    hidden_size = 128
    output_size = len(ACTIONS)
    self.model = MLP(input_size, hidden_size, output_size)
    self.optimizer = AdamW(self.model.parameters(), lr=LEARNING_RATE, amsgrad=True)

    if os.path.isfile("custom_mlp_policy_model.pth"):
        self.model.load_state_dict(torch.load('custom_mlp_policy_model.pth'))
    self.steps_done = 0

    # self.eps_threshold = EPS_END + (EPS_START - EPS_END) * \
    #     math.exp(-1. * self.steps_done / EPS_DECAY_FACTOR)
    self.eps_threshold = EPS_START
    self.episode_durations = []
    if os.path.isfile("training_episodes.pkl"):
        with open('training_episodes.pkl', 'rb') as f:
            eps_dur = pickle.load(f)
            if len(eps_dur) > 0:
                self.episode_durations = eps_dur
    self.episodes_round = 0
    self.logger.info(f"Current epsilon threshold: {self.eps_threshold}")

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

    old_game_state_feature = torch.tensor(state_to_features(old_game_state), dtype=torch.float32).unsqueeze(0)
    new_game_state_feature = torch.tensor(state_to_features(new_game_state), dtype=torch.float32).unsqueeze(0)
    rewards = reward_from_events(self, events, new_game_state_feature)

    # Put rewards into tensor
    reward = torch.tensor([rewards])
    action = torch.tensor([[action_string_to_index(self_action)]], dtype=torch.long)

    self.episodes_round += 1
    self.eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * self.steps_done / EPS_DECAY_FACTOR)
    self.steps_done += 1
    new_state_q_values = self.model(new_game_state_feature).max(1)[0]

    td_target = reward + DISCOUNT_FACTOR * new_state_q_values
    former_q_values = self.model(old_game_state_feature).gather(1, action)

    mse_loss = nn.MSELoss()
    tdError = mse_loss(td_target,former_q_values)

    # Optimize the model
    self.optimizer.zero_grad()
    tdError.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
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
    # Compute final reward based on the last events
    last_game_state_feature = torch.tensor(state_to_features(last_game_state), dtype=torch.float32).unsqueeze(0)
    final_reward = reward_from_events(self, events, last_game_state_feature)

    # Convert final reward to a tensor
    final_reward = torch.tensor([final_reward], dtype=torch.float32)

    # Compute Q-value of the last state-action pair
    last_state_feature = torch.tensor(state_to_features(last_game_state), dtype=torch.float32).unsqueeze(0)
    last_action = torch.tensor([[action_string_to_index(last_action)]], dtype=torch.long)
    last_q_value = self.model(last_state_feature).gather(1, last_action)

    # Compute the loss
    loss = nn.MSELoss()(last_q_value, final_reward.detach())

    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)  # Gradient clipping if needed
    self.optimizer.step()

    self.episode_durations.append(self.episodes_round)
    self.episodes_round = 0
    plot_durations(self, last_game_state)
    # Save model to file
    # torch.save(self.policy_net, 'custom_mlp_policy_model.pth')
    torch.save(self.model.state_dict(), 'custom_mlp_policy_model.pth')
    # Save current training episodes to file
    with open('training_episodes.pkl', 'wb') as f:
        pickle.dump(self.episode_durations, f)
    # Update epsilon threshold for new round
    self.eps_threshold = EPS_START * (1 - (len(self.episode_durations) / NUMBER_EPISODE))

def reward_from_events(self, events: List[str], new_game_state) -> int:
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
        e.COIN_COLLECTED: 100,
        e.CRATE_DESTROYED: 0,
        e.INVALID_ACTION: 0,
        e.KILLED_OPPONENT: 0,
        e.GOT_KILLED: 0,
        e.KILLED_SELF: 0,
        e.BOMB_DROPPED: 0,
        e.SURVIVED_ROUND: 0,
    }

    made_action = {
        e.MOVED_LEFT: 0,
        e.MOVED_RIGHT: 0,
        e.MOVED_UP: 0,
        e.MOVED_DOWN: 0,
        e.WAITED: 0,
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
        if event in made_action:
            reward_sum += made_action[event]
    # TODO: Check if agent in danger zone
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def plot_durations(self, gamestate, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    print("Coins collected = ", 50 - len(gamestate['coins']))
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    
    plt.pause(0.001)  # pause a bit so that plots are updated
    # Save plot to file
    plt.savefig('./plots/training_plot.png')