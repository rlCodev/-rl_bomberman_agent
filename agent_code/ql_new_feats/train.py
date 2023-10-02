from collections import namedtuple, deque
from matplotlib import pyplot as plt
import torch
import pickle
from typing import List
import os
import numpy as np
from torch import nn
from torch.optim import AdamW
import events as e
from .MLP import MLP
from .utils import action_string_to_index
from tensorboardX import SummaryWriter
from agent_code.ql_new_feats.features import state_to_features
from agent_code.ql_new_feats.parameters import INPUT_SIZE, HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE, OUTPUT_SIZE, ACTIONS, DISCOUNT_FACTOR, EPS_START, LEARNING_RATE, NUMBER_EPISODE, REWARDS

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.loss_function = nn.MSELoss()
    self.losses = []
    self.coins_collected = 0

    self.episode_rewards = []
    self.episode_reward = 0

    self.episode_losses = []
    self.episode_loss = 0

    # Create a TensorboardX writer
    self.writer = SummaryWriter(log_dir='logs')

    # Setup models
    self.model = MLP(INPUT_SIZE, HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE, OUTPUT_SIZE)
    self.optimizer = AdamW(self.model.parameters(), lr=LEARNING_RATE, amsgrad=True)

    if os.path.isfile("model/custom_mlp_policy_model.pth"):
        self.model.load_state_dict(torch.load('model/custom_mlp_policy_model.pth'))
    self.steps_done = 0

    self.eps_threshold = EPS_START
    self.episode_durations = []
    self.episodes_coins_collected = []
    if os.path.isfile("model/training_episodes.pkl"):
        with open('model/training_episodes.pkl', 'rb') as f:
            eps_dur = pickle.load(f)
            if len(eps_dur) > 0:
                self.episode_durations = eps_dur
    if os.path.isfile("model/training_coins_collected.pkl"):
        with open('model/training_coins_collected.pkl', 'rb') as f:
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
    rewards = reward_from_events(self, events)
    self.episode_reward = rewards

    # Convert feature matrices to tensors
    old_game_state_feature = torch.tensor(state_to_features(old_game_state), dtype=torch.float32).unsqueeze(0)
    new_game_state_feature = torch.tensor(state_to_features(new_game_state), dtype=torch.float32).unsqueeze(0)

    # Put rewards into tensor
    reward = torch.tensor([[rewards]])
    action = torch.tensor([[action_string_to_index(self_action)]], dtype=torch.long)

    self.episodes_round += 1
    self.steps_done += 1
    new_state_q_values = self.model(new_game_state_feature).max(1)[0]

    td_target = reward + DISCOUNT_FACTOR * new_state_q_values
    former_q_values = self.model(old_game_state_feature).gather(1, action)

    mse_loss = nn.MSELoss()
    tdError = mse_loss(td_target, former_q_values)
    self.episode_loss = tdError.item()
    # Log metrics to TensorboardX
    self.writer.add_scalar('Reward', self.episode_reward, self.steps_done)  # 'step' is your current training step
    self.writer.add_scalar('Loss', self.episode_loss, self.steps_done)
    # You can also log more metrics as needed
    # Close the self.writer when done
    self.writer.close()


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
    # Log metrics to TensorboardX
    self.writer.add_scalar('Reward', self.episode_reward, self.steps_done)  # 'step' is your current training step
    self.writer.add_scalar('Loss', self.episode_loss, self.steps_done)
    self.episode_loss = 0
    self.episode_reward = 0
    # You can also log more metrics as needed
    # Close the self.writer when done
    self.writer.close()
    
    self.steps_done += 1

    # Compute final reward based on the last events=
    final_reward = reward_from_events(self, events)

    # Convert final reward to a tensor
    final_reward = torch.tensor([[final_reward]], dtype=torch.float32)

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
    self.episodes_coins_collected.append(self.coins_collected)
    self.episodes_round = 0
    self.tiles_visited = set()
    self.coins_collected = 0
    # plot_durations(self, last_game_state)
    # Save model to file
    # torch.save(self.policy_net, 'custom_mlp_policy_model.pth')
    torch.save(self.model.state_dict(), 'model/custom_mlp_policy_model.pth')
    # Save current training episodes to file
    with open('model/training_episodes.pkl', 'wb') as f:
        pickle.dump(self.episode_durations, f)
    with open('model/training_coins_collected.pkl', 'wb') as f:
        pickle.dump(self.episodes_coins_collected, f) 
    # Update epsilon threshold for new round
    self.eps_threshold = EPS_START * (1 - (len(self.episode_durations) / NUMBER_EPISODE))
    if (len(self.episode_durations) % 1000 == 0):
        torch.save(self.model.state_dict(), f'mlp_model_after_{len(self.episode_durations)}.pth')

def reward_from_events(self, occurred_events: List[str]) -> int:
    reward_sum = 0
    for event in occurred_events:
        if event == e.COIN_COLLECTED:
            reward_sum += REWARDS[event]
        if event == e.INVALID_ACTION:
            reward_sum += REWARDS[event]
        if event == e.WAITED:
            reward_sum += REWARDS[event]
        if event == e.KILLED_SELF:
            reward_sum += REWARDS[event]
        if event == e.COIN_COLLECTED:
            self.coins_collected += 1
    return reward_sum