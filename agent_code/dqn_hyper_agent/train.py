from matplotlib import pyplot as plt
import torch
import pickle
from typing import List
import os
import numpy as np
from torch import nn
from torch.optim import AdamW
from agent_code.dqn_hyper_agent.replay_memory import ReplayMemory, Transition
from tensorboardX import SummaryWriter
from datetime import datetime
from typing import List, Tuple
from agent_code.dqn_hyper_agent.features import state_to_features
from agent_code.dqn_hyper_agent.utils import action_string_to_index
import events as e

import numpy as np

from agent_code.dqn_hyper_agent.static_params import (
    ACTIONS, BATCH_SIZE, CURRENT_MODEL, EPSILON_DECAY, EPSILON_END, EPSILON_START,
    EXPERIENCE_BUFFER_SIZE_MIN, EXPERIENCE_BUFFER_SIZE_MAX, GAMMA, LEARNING_RATE, LOSS_FUNCTION, REWARDS, SAVE_MODEL_EVERY,
    TRAINING_ROUNDS, UPDATE_TARGET_MODEL_EVERY, UPDATE_EVERY)
from settings import ROWS, COLS

def setup_training(self):
    self.memory = ReplayMemory(EXPERIENCE_BUFFER_SIZE_MAX)

    self.epsilon = EPSILON_START

    self.episode_rewards = []
    self.episode_reward = 0

    self.episode_losses = []
    self.episode_loss = 0

    # Create a TensorboardX writer
    self.writer = SummaryWriter(log_dir='logs')

    self.training_timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

    # Setup monitoring to track progress
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

    # Track visited tiles for giving rewards for visiting many tiles
    self.coins_collected = 0
    
    self.steps_done = 0

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    if old_game_state is None or new_game_state is None or self_action is None:
        return

    action = ACTIONS.index(self_action)
    reward = reward_from_events(events)
    reward += get_movement_reward(old_game_state['self'][3], new_game_state['self'][3], old_game_state['field'],
                                  old_game_state['bombs'], old_game_state['coins'], old_game_state['others'])
    # td_error = get_td_error(self, old_game_state, action, reward, new_game_state)
    # ex_feartur = state_to_features(old_game_state)
    reward = torch.tensor([reward])
    action = torch.tensor([[action_string_to_index(self_action)]], dtype=torch.long)
    state = state_to_features(old_game_state)
    self.memory.push(state_to_features(old_game_state), action, state_to_features(new_game_state), reward)

    update_q_values(self)
    self.epsilon = self.epsilon * EPSILON_DECAY if self.epsilon > EPSILON_END else EPSILON_END
    self.episode_reward += reward

    self.episodes_round += 1
    if e.COIN_COLLECTED in events:
        self.coins_collected += 1


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.episode_rewards.append(self.episode_reward)
    self.episode_reward = 0

    self.episode_losses.append(self.episode_loss)
    self.episode_loss = 0

    if not last_game_state["round"] % UPDATE_EVERY:
        # Replace these with your actual metrics
        average_reward = sum(self.episode_rewards[-UPDATE_EVERY:]) / UPDATE_EVERY
        average_loss = sum(self.episode_losses[-UPDATE_EVERY:]) / UPDATE_EVERY
        epsilon = self.epsilon

        # Log metrics to TensorboardX
        self.writer.add_scalar('Average Reward', average_reward, self.steps_done)  # 'step' is your current training step
        self.writer.add_scalar('Average Loss', average_loss, self.steps_done)
        self.writer.add_scalar('Epsilon', epsilon, self.steps_done)
        # You can also log more metrics as needed

        # Close the self.writer when done
        self.writer.close()
        
        self.steps_done += 1

    if not last_game_state["round"] % UPDATE_TARGET_MODEL_EVERY:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    if not last_game_state["round"] % SAVE_MODEL_EVERY:
        torch.save(self.policy_net.state_dict(), f"models/{CURRENT_MODEL}.pth")
        print(f"EXPERT BUFFER SIZE: {len(self.memory)} of {EXPERIENCE_BUFFER_SIZE_MAX}")
        # self.experience_buffer.save_samples("buffers/{}".format(self.training_timestamp))

    if last_game_state["round"] == TRAINING_ROUNDS:
        torch.save(self.policy_net.state_dict(), f"models/{CURRENT_MODEL}.pth")
        print(f"EXPERT BUFFER SIZE: {len(self.memory)} of {EXPERIENCE_BUFFER_SIZE_MAX}")
        # self.experience_buffer.save_samples("buffers/{}".format(self.training_timestamp))


    self.episode_durations.append(self.episodes_round)
    self.episodes_coins_collected.append(self.coins_collected)
    self.episodes_round = 0
    self.tiles_visited = set()
    self.coins_collected = 0
    # Only plot every 20 episodes
    # if len(self.episode_durations) % 100 == 0:
    plot_durations(self, last_game_state)
    # Save current training episodes to file
    with open('training_episodes.pkl', 'wb') as f:
        pickle.dump(self.episode_durations, f)
    with open('training_coins_collected.pkl', 'wb') as f:
        pickle.dump(self.episodes_coins_collected, f)


def reward_from_events(occurred_events: List[str]) -> int:
    reward_sum = 0
    for event in occurred_events:
        reward_sum += REWARDS[event]
    return reward_sum


def get_td_error(self, old_game_state: dict, action: int, reward: float, new_game_state: dict) -> float:
    old_game_state = np.array([state_to_features(old_game_state)])
    new_game_state = np.array([state_to_features(new_game_state)])

    q_next_actual = self.target_net(new_game_state)
    q_next_train = self.policy_net(new_game_state)
    q_target_old = self.policy_net(old_game_state)
    q_target_new = q_target_old.copy()

    q_target_new[0, action] = reward + GAMMA * q_next_actual[0, np.argmax(q_next_train, axis=1)]
    return np.sum(np.abs(q_target_old - q_target_new), axis=1)[0]


def get_movement_reward(old_pos: Tuple[int, int], new_pos: Tuple[int, int], field: np.ndarray,
                        bombs: List[Tuple[Tuple[int, int], int]], coins: List[Tuple[int, int]], others: List[Tuple[str, int, bool, Tuple[int, int]]]) -> float:

    movement_reward = 0

    # TODO: weight with bomb timer
    danger_tile_coordinates = []
    for bomb in bombs:
        bomb_x = bomb[0][0]
        bomb_y = bomb[0][1]
        for x, y in zip([1, 0, -1, 0], [0, 1, 0, -1]):
            for r in range(1, 4):
                if bomb_x + x*r >= 0 and bomb_x + x*r < ROWS and bomb_y + y*r >= 0 and bomb_y + y*r < COLS and field[bomb_x + x*r, bomb_y + y*r] == 0:
                    danger_tile_coordinates.append((bomb_x + x*r, bomb_y + y*r))
        danger_tile_coordinates.append((bomb_x, bomb_y))

    if new_pos in danger_tile_coordinates:
        movement_reward -= 0.3

    if old_pos in danger_tile_coordinates and new_pos not in danger_tile_coordinates:
        movement_reward += 0.3

    old_pos = np.asarray(old_pos)
    new_pos = np.asarray(new_pos)

    # TODO: add enemy position recognition
    if len(coins) > 0:
        coins = np.asarray(coins)
        coin_to_collect = np.argmin(np.sum(np.abs(coins - old_pos), axis=1))

        old_distance = np.sum(np.abs(coins[coin_to_collect] - old_pos))
        new_distance = np.sum(np.abs(coins[coin_to_collect] - new_pos))
        if new_distance < old_distance:
            movement_reward += 0.1
        else:
            movement_reward -= 0.1

    if len(others) > 0:
        others = np.asarray([other[3] for other in others])
        opponent_to_attack = np.argmin(np.sum(np.abs(others - old_pos), axis=1))

        old_distance = np.sum(np.abs(others[opponent_to_attack] - old_pos))
        new_distance = np.sum(np.abs(others[opponent_to_attack] - new_pos))
        if new_distance < old_distance:
            movement_reward += 0.05
        else:
            movement_reward -= 0.05

    return movement_reward


def update_q_values(self):
    self.optimizer = AdamW(self.policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
    if len(self.memory) < EXPERIENCE_BUFFER_SIZE_MIN:
        return

    transitions =self.memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    next_states_batch = torch.cat(batch.next_state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    size = next_states_batch.size()

    qs_next_actual = self.target_net(next_states_batch)
    qs_next_train = self.policy_net(next_states_batch)
    qs_target_old = self.policy_net(state_batch)
    qs_target_new = qs_target_old.clone()

    batch_index = np.arange(BATCH_SIZE)

    # qs_target_new[batch_index, action_batch] = reward_batch + GAMMA * qs_next_actual[batch_index, np.argmax(qs_next_train, axis=1)]

    # Calculate the maximum Q-value for the next state for each sample in the batch
    max_q_values_next = qs_next_train.max(dim=1).values

    # Compute the target Q-values using the Bellman equation
    target_q_values = reward_batch + GAMMA * max_q_values_next

    # Update the qs_target_new tensor with the calculated target Q-values
    qs_target_new[batch_index, action_batch] = target_q_values

    # In case of bad performance one could try to recalculate the sumtree every x times
    # First approximations will still be accurate in the first few rounds
    # td_errors = np.sum(np.abs(qs_target_new - qs_target_old), axis=1)
    abs_td_errors = torch.abs(qs_target_new - qs_target_old)
    # Sum along axis 1 to compute the TD errors for each sample in the batch
    td_errors = torch.sum(abs_td_errors, dim=1)

    # self.experience_buffer.update(indices, td_errors)
    # self.episode_loss += np.sum(td_errors)
    self.episode_loss += torch.sum(td_errors).item()

    # Update model
    loss = LOSS_FUNCTION(torch.tensor(qs_target_old, requires_grad=True), torch.tensor(qs_target_new, requires_grad=True))

    self.optimizer.zero_grad()
    self.logger.debug(f'Loss: {loss}')
    loss.backward()
    # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
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