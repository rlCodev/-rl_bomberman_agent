from collections import namedtuple, deque
from matplotlib import pyplot as plt
import torch
import pickle
from typing import List
import os
import events as e
from .callbacks import state_to_features
import numpy as np
from torch import nn
from torch.optim import AdamW
from .MLP import MLP
from .utils import action_string_to_index
import agent_code.ql_agent.helper as helper

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
# Discound factor or gamma
DISCOUNT_FACTOR = 0.95
EPS_START = 0.99
EPS_END = 0.0
LEARNING_RATE = 0.0001

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

#Number Episodes has to match the number of episodes set in the json
NUMBER_EPISODE = 1000
INPUT_SIZE = 30
HIDDEN_LAYER_1_SIZE = 20
HIDDEN_LAYER_2_SIZE = 12
OUTPUT_SIZE = len(ACTIONS)

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.loss_function = nn.MSELoss()
    self.losses = []
    self.coins_collected = 0

    # Setup models
    self.model = MLP(INPUT_SIZE, HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE, OUTPUT_SIZE)
    self.optimizer = AdamW(self.model.parameters(), lr=LEARNING_RATE, amsgrad=True)

    if os.path.isfile("custom_mlp_policy_model.pth"):
        self.model.load_state_dict(torch.load('custom_mlp_policy_model.pth'))
    self.steps_done = 0

    self.eps_threshold = EPS_START
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

    # Get feature Matrix
    # old_feature_matrix = helper.state_to_features_matrix(self, old_game_state)
    # new_feature_matrix = helper.state_to_features_matrix(self, new_game_state)

    # Calculate Rewards
    rewards = reward_from_events(self, events, old_game_state, new_game_state)

    # Convert feature matrices to tensors
    old_game_state_feature = torch.tensor(state_to_features(self, old_game_state), dtype=torch.float32).unsqueeze(0)
    new_game_state_feature = torch.tensor(state_to_features(self, new_game_state), dtype=torch.float32).unsqueeze(0)

    # Put rewards into tensor
    reward = torch.tensor([rewards])
    action = torch.tensor([[action_string_to_index(self_action)]], dtype=torch.long)

    self.episodes_round += 1
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
    # Compute final reward based on the last events=
    final_reward = reward_from_events(self, events, last_game_state, None)

    # Convert final reward to a tensor
    final_reward = torch.tensor([final_reward], dtype=torch.float32)

    # Compute Q-value of the last state-action pair
    last_state_feature = torch.tensor(state_to_features(self, last_game_state), dtype=torch.float32).unsqueeze(0)
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
    self.episodes_coins_collected.append(self.coins_collected)
    self.episodes_round = 0
    self.tiles_visited = set()
    self.coins_collected = 0
    plot_durations(self, last_game_state)
    # Save model to file
    # torch.save(self.policy_net, 'custom_mlp_policy_model.pth')
    torch.save(self.model.state_dict(), 'custom_mlp_policy_model.pth')
    # Save current training episodes to file
    with open('training_episodes.pkl', 'wb') as f:
        pickle.dump(self.episode_durations, f)
    with open('training_coins_collected.pkl', 'wb') as f:
        pickle.dump(self.episodes_coins_collected, f) 
    # Update epsilon threshold for new round
    self.eps_threshold = EPS_START * (1 - (len(self.episode_durations) / NUMBER_EPISODE))
    if (len(self.episode_durations) % 1000 == 0):
        torch.save(self.model.state_dict(), f'mlp_model_after_{len(self.episode_durations)}.pth')

def reward_from_events(self, events: List[str], old_game_state: dict, new_game_state: dict) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.CRATE_DESTROYED: 1,
        e.INVALID_ACTION: -2,
        e.KILLED_OPPONENT: 1,
        e.GOT_KILLED: -4,
        e.KILLED_SELF: -4,
        e.WAITED: -1,
    }

    moves = {
        e.MOVED_UP: 3,
        e.MOVED_DOWN: 2,
        e.MOVED_LEFT: 1,
        e.MOVED_RIGHT: 0
    }

    reward_sum = 0

    # Punish for choosing invalid actions
    if e.INVALID_ACTION in events:
        reward_sum += game_rewards[e.INVALID_ACTION]


    if new_game_state is not None:
        new_features = helper.state_to_features_matrix(self, new_game_state)
        old_features = helper.state_to_features_matrix(self, old_game_state)
        for event in events:
            # Check if we are even allowed to place a bomb:
            bomb_droppable = old_features[4][5]

            # Punish for not dropping bomb, when possible:
            if bomb_droppable != -1 and e.BOMB_DROPPED not in events:
                reward_sum -= 1

            # Tile, where bomb effectiveness is max
            max_bomb_effect_index = np.argmax(old_features[:, 5])
            if event == e.BOMB_DROPPED:
                # If we are not allowed to drop a bomb, punish
                if bomb_droppable == -1:
                    reward_sum -= 1
                else:
                    # Check if we dropped effective bomb
                    if max_bomb_effect_index == 4:
                        reward_sum += 1
                    # If we are now in certain death, punish
                    if new_features[4][4] > 0:
                        reward_sum -= 5
            elif event in moves:
                # If we make move landing us in certain death, punish
                if new_features[4][4] > 0 and new_features[4][4] != old_features[4][4]:
                    reward_sum -= 2
                
                # If in danger, move away!
                old_danger = old_features[4][3]
                if old_danger > 0:
                    # find best move:
                    old_features[:, 3][old_features[:, 3] == -1] = 10
                    max_danger_avoidance_idx = np.argmin(old_features[:, 3])
                    # If we made the best move to avoid danger, reward
                    if moves[event] == max_danger_avoidance_idx:
                        reward_sum += 4
                    else:
                        reward_sum -= 5
                else:
                    # Reward for making best move toward coins
                    if old_game_state['coins'] is not None:
                        # Find best move towards coins
                        old_features[:, 0][old_features[:, 0] == -1] = 100
                        min_coindistance_idx = np.argmin(old_features[:, 0])
                        # If we made best move, reward
                        if moves[event] == min_coindistance_idx:
                            reward_sum += 1
                        # If the move was not the best, but still good, reward
                        elif old_features[4][moves[event]] - old_features[4][min_coindistance_idx] < 2:
                            reward_sum += 0
                        else:
                            reward_sum -= 1
                    else:
                        # Reward / punish for moving towards / away from enemies
                        distance_to_enemies_old = old_features[4][1]
                        distance_to_enemies_new = new_features[4][1]
                        if distance_to_enemies_new < distance_to_enemies_old:
                            reward_sum += (distance_to_enemies_old - distance_to_enemies_new)
                        else:
                            reward_sum -= (distance_to_enemies_new - distance_to_enemies_old)
                    
                    # If we moved but should've dropped a bomb, punish
                    if max_bomb_effect_index == 4:
                        reward_sum -= 1

                # Reward for exploring a new tile
                # If the move we made leads to exploring new tile => reward
                if old_features[2][moves[event]] == 1:
                    reward_sum += 2
                else:
                    reward_sum -= 1

            elif event == e.COIN_COLLECTED:
                self.coins_collected += 1
            else:
                if event in game_rewards:
                    reward_sum += game_rewards[event]
    # TODO: Check if agent in danger zone
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")

    return reward_sum

def plot_durations(self, gamestate, show_result=False):
    fig = plt.figure(1)
    durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
    coins_collected_t = torch.tensor(self.episodes_coins_collected, dtype=torch.float)

    if show_result:
        fig.suptitle('Result')
    else:
        plt.clf()
        fig.suptitle('Training...')
    
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
    ax2.plot(np.arange(len(moving_avg)) + moving_average_window - 1, moving_avg, color='green', label='Coins Collected (Avg)')
    
    plt.pause(0.001)  # pause a bit so that plots are updated
    # Save plot to file
    plt.savefig('./plots/training_plot.png')