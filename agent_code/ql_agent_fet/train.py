from collections import namedtuple, deque
import math
from matplotlib import pyplot as plt
import torch
import pickle
from typing import List
import os
import events as e
import numpy as np
from torch import nn
from torch.optim import AdamW, SGD
from .MLP import MLP
import random
from .utils import action_index_to_string, action_string_to_index
from agent_code.ql_agent_fet.feature_builder import state_to_features_special

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- TODO modify/optimize
TRANSITION_HISTORY_SIZE = 2000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# DISCOUNT_FACTOR is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LEARNING_RATE is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 120
# Discound factor or gamma
DISCOUNT_FACTOR = 0.99
EPS_START = 0.99
EPS_END = 0.1
STATIC_EPS = 0.1
EPS_DECAY_FACTOR = 1000000
TAU = 0.001
LEARNING_RATE = 0.0001
NUM_EPISODES = 10000

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
    self.tiles_visited = []
    self.coins_collected = 0

    # Setup models
    input_size = 23
    hidden_size = 128
    output_size = len(ACTIONS)
    self.policy_net = MLP(input_size, hidden_size, output_size)
    self.target_net = MLP(input_size, hidden_size, output_size)
    if os.path.isfile("custom_mlp_policy_model.pth"):
        self.policy_net.load_state_dict(torch.load('custom_mlp_policy_model.pth'))
        self.target_net.load_state_dict(torch.load('custom_mlp_target_model.pth'))
        # self.policy_net = torch.load('custom_mlp_policy_model.pth')
        # self.target_net = torch.load('custom_mlp_target_model.pth')
    self.steps_done = 0
    self.eps_threshold = EPS_START
    
    # self.eps_threshold = STATIC_EPS
    # Load episodes from file
    self.episode_durations = []
    self.episodes_coins_collected = []
    if os.path.isfile("training_episodes.pkl"):
        with open('training_episodes.pkl', 'rb') as f:
            eps_dur = pickle.load(f)
            if len(eps_dur) > 0:
                self.episode_durations = eps_dur
    if os.path.isfile("training_coins_collected.pkl"):
        with open('training_coins_collected.pkl', 'rb') as f:
            coins = pickle.load(f)
            if len(coins) > 0:
                self.episodes_coins_collected = coins
    self.episodes_round = 0
    self.logger.info(f"Loaded {len(self.episode_durations)} training episodes from file")
    self.logger.info(f"Current epsilon threshold: {self.eps_threshold}")
    self.memory = ReplayMemory(TRANSITION_HISTORY_SIZE)
    self.t = 0

    self.bomb_buffer = 0

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
    old_game_state_feature = torch.tensor(state_to_features_special(self, old_game_state), dtype=torch.float32).unsqueeze(0)
    new_game_state_feature = torch.tensor(state_to_features_special(self, new_game_state), dtype=torch.float32).unsqueeze(0)
    rewards = reward_from_events(self, events, new_game_state)
    # Put rewards into tensor
    reward = torch.tensor([rewards])
    action = torch.tensor([[action_string_to_index(self_action)]], dtype=torch.long)
    self.memory.push(old_game_state_feature, action, new_game_state_feature, reward)
    self.episodes_round += 1
    # self.eps_threshold = EPS_END + (EPS_START - EPS_END) * \
    #     math.exp(-1. * self.steps_done / EPS_DECAY_FACTOR)
    
    self.eps_threshold = EPS_START * (1 - (len(self.episode_durations) / NUM_EPISODES))

    self.steps_done += 1
    # print("Rewards: ", reward)

    update_model(self)

    # # Idea: Add your own events to hand out rewards
    # if ...:
    #     events.append(PLACEHOLDER_EVENT)

    # state_to_features_special is defined in callbacks.py
    # self.update_model(old_game_state, self_action, new_game_state, events)

    


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
    # self.memory.append(
    #     Transition(state_to_features_special(last_game_state), last_action, None, reward_from_events(self, events)))
    last_game_state_feature = torch.tensor(state_to_features_special(self, last_game_state), dtype=torch.float32).unsqueeze(0)
    rewards = reward_from_events(self, events, last_game_state)
    # Put rewards into tensor
    reward = torch.tensor([rewards])
    action = torch.tensor([[action_string_to_index(last_action)]], dtype=torch.long)
    self.memory.push(last_game_state_feature, action, None, reward)
    # Translate events to a list of integers
    # print("Rewards: ", reward)
    # Update the model
    update_model(self)
    self.episode_durations.append(self.episodes_round)
    self.episodes_coins_collected.append(self.coins_collected)
    self.episodes_round = 0
    self.tiles_visited = []
    self.coins_collected = 0
    # Only plot every 20 episodes
    # if len(self.episode_durations) % 100 == 0:
    plot_durations(self, gamestate=last_game_state)
    # Save model to file
    # torch.save(self.policy_net, 'custom_mlp_policy_model.pth')
    torch.save(self.policy_net.state_dict(), 'custom_mlp_policy_model.pth')
    # torch.save(self.target_net, 'custom_mlp_target_model.pth')
    torch.save(self.target_net.state_dict(), 'custom_mlp_target_model.pth')
    # Save current training episodes to file
    with open('training_episodes.pkl', 'wb') as f:
        pickle.dump(self.episode_durations, f)
    with open('training_coins_collected.pkl', 'wb') as f:
        pickle.dump(self.episodes_coins_collected, f) 
    # Clean memory after each round for preventing memory leakage
    self.memory = ReplayMemory(TRANSITION_HISTORY_SIZE)
    # Update epsilon threshold for new round
    # self.eps_threshold = EPS_END + (EPS_START - EPS_END) * \
    #     math.exp(-1. * self.steps_done / EPS_DECAY_FACTOR)
    self.eps_threshold = EPS_START * (1 - (len(self.episode_durations) / NUM_EPISODES))


def reward_from_events(self, events: List[str], new_game_state: dict) -> int:
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
        e.COIN_COLLECTED: 500,
        e.CRATE_DESTROYED: 100,
        e.INVALID_ACTION: -2,
        e.KILLED_OPPONENT: 100,
        e.GOT_KILLED: -300,
        e.KILLED_SELF: -300,
        e.BOMB_DROPPED: 1,
    }

    made_action = {
        e.MOVED_LEFT: -1,
        e.MOVED_RIGHT: -1,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.WAITED: -1,
    }

    reward_sum = 0
    for event in events:
        if event == e.COIN_COLLECTED:
            self.coins_collected += 1
        if event in game_rewards:
            reward_sum += game_rewards[event]
        if event in made_action:
            reward_sum += made_action[event]
    # TODO: Check if agent in danger zone
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    
    # Add reward for visiting new tiles
    if 'self' in new_game_state and len(new_game_state['self']) > 3:
        if new_game_state['self'][3] not in self.tiles_visited:
            self.tiles_visited.append(new_game_state['self'][3])
            reward_sum += 10

    return reward_sum

def update_model(self):
    target_net_state_dict = self.target_net.state_dict()
    policy_net_state_dict = self.policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
    self.target_net.load_state_dict(target_net_state_dict)

    # Based on https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    self.optimizer = AdamW(self.policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
    if len(self.memory) < BATCH_SIZE:
        return
    transitions =self.memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = self.policy_net(state_batch).squeeze(1).gather(1, action_batch)


    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE)
    with torch.no_grad():
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).squeeze(1).max(1)[0]
        # next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(dim=2).item()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * DISCOUNT_FACTOR) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    self.losses.append(loss)

    print(f"Episode {len(self.episode_durations)} Loss: {loss.detach().numpy().item()}")
    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
    self.optimizer.step()

def plot_durations(self, gamestate, show_result=False):
    fig = plt.figure(1)
    durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
    coins_collected_t = torch.tensor(self.episodes_coins_collected, dtype=torch.float)

    if show_result:
        fig.suptitle('Result')
    else:
        plt.clf()
        fig.suptitle('Training...')
    plt.clf()
    ax1 = fig.add_subplot(211)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Duration')
    ax1.plot(durations_t.numpy(), label='Duration')
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        ax1.plot(means.numpy(), color='red', label='Duration (Avg)')

    # Create a secondary y-axis for coins collected
    ax2 = fig.add_subplot(212)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Coins Collected')
    ax2.plot(coins_collected_t.numpy(), label='Coins Collected', color='orange')

    # Calculate and plot the moving average of coins collected
    moving_average_window = 100  # Adjust this window size as needed
    moving_avg = np.convolve(coins_collected_t.numpy(), np.ones(moving_average_window)/moving_average_window, mode='valid')
    ax2.plot(np.arange(len(moving_avg)) + moving_average_window - 1, moving_avg, linestyle='--', color='green', label='Coins Collected (Avg)')
    
    plt.pause(0.001)  # pause a bit so that plots are updated
    # Save plot to file
    plt.savefig('./plots/training_plot.png')

# def plot_durations(self, gamestate, show_result=False):
#     plt.figure(1)
#     durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
#     if show_result:
#         plt.title('Result')
#     else:
#         plt.clf()
#         plt.title('Training...')
#     plt.xlabel('Episode')
#     plt.ylabel('Duration')
#     plt.plot(durations_t.numpy())
#     # Take 100 episode averages and plot them too
#     if len(durations_t) >= 100:
#         means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
#         means = torch.cat((torch.zeros(99), means))
#         plt.plot(means.numpy())
    
#     plt.pause(0.001)  # pause a bit so that plots are updated
#     # Save plot to file
#     plt.savefig('./plots/training_plot.png')