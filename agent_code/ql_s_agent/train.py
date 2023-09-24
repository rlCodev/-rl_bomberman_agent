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
from tensorboardX import SummaryWriter

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
NUMBER_EPISODE = 500
INPUT_SIZE = 29
HIDDEN_LAYER_1_SIZE = 256
HIDDEN_LAYER_2_SIZE = 256
OUTPUT_SIZE = len(ACTIONS)

# Auxillary events
WAITING_EVENT = "WAIT"
VALID_ACTION = "VALID_ACTION"
COIN_CHASER = "COIN_CHASER"
MOVED_OUT_OF_DANGER = "MOVED_AWAY_FROM_EXPLODING_TILE"
MOVED_INTO_DANGER = "MOVED_INTO_DANGER"
STAYED_NEAR_BOMB = 'STAYED_ON_EXPLODING_TILE'
CRATE_CHASER = 'CRATE_CHASER'
BOMB_NEXT_TO_CRATE = 'BOMB_NEXT_TO_CRATE'
BOMB_DESTROYED_NOTHING = 'BOMB_DESTROYED_NOTHING'
BOMB_NOT_NEXT_TO_CRATE = 'BOMB_NOT_NEXT_TO_CRATE'
DROPPED_BOMB_NEAR_ENEMY = 'DROPPED_BOMB_NEAR_ENEMY'
DROPPED_BOMB_NEXT_TO_ENEMY ='DROPPED_BOMB_NEXT_TO_ENEMY'
OPPONENT_CHASER = 'CHASED_OPPONENT'

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.loss_function = nn.SmoothL1Loss()
    self.losses = []
    self.coins_collected = 0

    self.episode_rewards = []
    self.episode_reward = 0

    self.episode_losses = []
    self.episode_loss = 0

    with open('explosion_map.pt', 'rb') as file:
        self.exploding_tiles_map = pickle.load(file)

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
    events = aux_events(self, old_game_state, self_action, new_game_state, events)
    rewards = reward_from_events(self, events)
    self.episode_reward = rewards

    # Convert feature matrices to tensors
    old_game_state_feature = state_to_features(old_game_state)
    new_game_state_feature = state_to_features(new_game_state)

    # Put rewards into tensor
    reward = torch.tensor([rewards])
    action = torch.tensor([action_string_to_index(self_action)], dtype=torch.long)

    self.episodes_round += 1
    self.steps_done += 1
    new_state_q_values = self.model(new_game_state_feature).max(-1)[0]

    td_target = reward + DISCOUNT_FACTOR * new_state_q_values
    former_q_values = self.model(old_game_state_feature).gather(-1, action)

    huber_loss = nn.SmoothL1Loss()
    tdError = huber_loss(td_target, former_q_values)

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

    # Compute final reward based on the last events=
    # events = aux_events(self, last_game_state, last_action, None, events)
    final_reward = reward_from_events(self, events)
    self.episode_reward = final_reward

    # Convert final reward to a tensor
    final_reward = torch.tensor([final_reward], dtype=torch.float32)

    # Compute Q-value of the last state-action pair
    last_state_feature = state_to_features(last_game_state)
    last_action = torch.tensor([action_string_to_index(last_action)], dtype=torch.long)
    last_q_value = self.model(last_state_feature).gather(-1, last_action)

    # Compute the loss
    loss = nn.SmoothL1Loss()(last_q_value, final_reward.detach())
    self.episode_loss = loss.item()
    # Log metrics to TensorboardX
    self.writer.add_scalar('Reward', self.episode_reward, self.steps_done)  # 'step' is your current training step
    self.writer.add_scalar('Loss', self.episode_loss, self.steps_done)
    self.episode_loss = 0
    self.episode_reward = 0
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

def reward_from_events(self, events: List[str]) -> int:
    '''
        Input: self, list of events
        Output: sum of rewards resulting from the events
    '''
    game_rewards = {
        e.COIN_COLLECTED: 8,
        e.KILLED_OPPONENT: 20,
        e.KILLED_SELF: -80,
        e.WAITED: -1,
        e.INVALID_ACTION: -4,
        e.MOVED_DOWN: -1,
        e.MOVED_LEFT: -1,
        e.MOVED_RIGHT: -1,
        e.MOVED_UP: -1,
        COIN_CHASER: 3,             
        MOVED_OUT_OF_DANGER: 5,
        STAYED_NEAR_BOMB: -5,
        MOVED_INTO_DANGER: -5,
        CRATE_CHASER: 1.5,
        BOMB_NEXT_TO_CRATE: 2,
        BOMB_NOT_NEXT_TO_CRATE: -2,
        DROPPED_BOMB_NEAR_ENEMY: 1,
        DROPPED_BOMB_NEXT_TO_ENEMY: 8, 
        OPPONENT_CHASER: 2
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum



def aux_events(self, old_game_state, self_action, new_game_state, events):
    '''Defining auxillary events for auxillary rewards to optimize training'''
    
    # get positions of the player in old state and new state (tuples (x,y) in this case)
    old_player_coor = old_game_state['self'][3]     
    new_player_coor = new_game_state['self'][3]
        
 
    #define event coin_chaser
    coin_coordinates = old_game_state['coins']      #get coin coordinates(also tuples in form (x,y))
    if len(coin_coordinates) != 0:                  #now calculate distance to all coins in respect to...
        old_coin_distances = np.linalg.norm(np.subtract(coin_coordinates,old_player_coor), axis=1) #...old player position
        new_coin_distances = np.linalg.norm(np.subtract(coin_coordinates,new_player_coor), axis=1) #...new player position

        if min(new_coin_distances) < min(old_coin_distances):   #if the distance to closest coin got smaller
            events.append(COIN_CHASER)                          # -> reward

    
    #define events with bombs
    old_bomb_coors = old_game_state['bombs']                #get bomb coordinates (careful: timer still included: ((x,y),t)) for each bomb)

    dangerous_tiles = []                                    #this array will store all tuples with 'dangerous' tile coordinates
    for bomb in old_bomb_coors:
        for coor in self.exploding_tiles_map[bomb[0]]:      #for each bomb get all tiles that explode with that bomb...
            dangerous_tiles.append(coor)                    ##... and append them to dangerous_tiles


    if dangerous_tiles != []:

        #event in case the agent sucsessfully moved away from a dangerous tile -> reward     
        if old_player_coor in dangerous_tiles and new_player_coor not in dangerous_tiles:
            events.append(MOVED_OUT_OF_DANGER)

        #event in case agent stayed on a dangerous tile -> penalty
        if old_player_coor in dangerous_tiles and ("WAITED" in events or "INVALID_ACTION" in events):
            events.append(STAYED_NEAR_BOMB)
        
        #event in case agent moved onto a dangerous tile -> penalty
        if old_player_coor not in dangerous_tiles and new_player_coor in dangerous_tiles:
            events.append(MOVED_INTO_DANGER)
    
    #define crate chaser: the agent gets rewarded if he moves closer to crates ONLY if he currently has a bomb
    field = old_game_state['field']
    rows,cols = np.where(field == 1)
    crates_position = np.array([rows,cols]).T       #all crate coordinates in form [x,y] in one array
    old_crate_distance = np.linalg.norm(crates_position-np.array([old_player_coor[0],old_player_coor[1]]),axis = 1)
    new_crate_distance = np.linalg.norm(crates_position-np.array([new_player_coor[0],new_player_coor[1]]),axis = 1)

    if old_crate_distance.size > 0:                 #if agent moved closer to the nearest crate and BOMB action is possible 
        if min(new_crate_distance) < min(old_crate_distance) and old_game_state['self'][2]: 
            events.append(CRATE_CHASER)
        
    #get opponents
    enemys = []
    for others_coor in old_game_state['others']:
        enemys.append(others_coor[3])

    if self_action == 'BOMB' and e.INVALID_ACTION not in events:    #if bomb is placed...  

        #define event for bomb next to crate
        for i in range(len(np.where(old_crate_distance==1)[0])):    # ... give reward for each crate neighbouring bomb position                   
            events.append(BOMB_NEXT_TO_CRATE)        

        #define event for bomb not next to crate           
        if len(np.where(old_crate_distance==1)[0]) == 0 :           #bomb is not placed next to crate
            events.append(BOMB_NOT_NEXT_TO_CRATE)                   # -> penalty
            
        
        #define event for bomb near/next to opponent
        if len(old_game_state['others']) !=0:

            for others_coor in old_game_state['others']:

                if np.linalg.norm(np.subtract(old_player_coor,others_coor[3])) <=3:     #bomb placed near enemy -> reward
                    events.append(DROPPED_BOMB_NEAR_ENEMY)
                
                if np.linalg.norm(np.subtract(old_player_coor,others_coor[3])) == 1:    #bomb placed net to enemy -> reward
                    events.append(DROPPED_BOMB_NEXT_TO_ENEMY)
        

    #define opponent chaser
    if len(enemys) != 0:                                                            
        distances_old = np.linalg.norm(np.subtract(old_player_coor,enemys),axis=1)
        distances_new = np.linalg.norm(np.subtract(new_player_coor,enemys),axis=1)
        if min(distances_new) < min(distances_old):                                 #if agent moved closer to the closest enemy -> rewards
            events.append(OPPONENT_CHASER)
    return events