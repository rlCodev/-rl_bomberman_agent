from collections import namedtuple, deque
from matplotlib import pyplot as plt
import torch
import pickle
from typing import List
import os
from agent_code.ql_s_agent.feature import get_danger_map
import events as e
from agent_code.ql_s_agent.feature import state_to_features
from agent_code.ql_s_agent.old_feature import state_to_features as old_state_to_features
import agent_code.ql_s_agent.static_props as hp
# import agent_code.ql_s_agent.feature as feature
import numpy as np
from torch import nn
from torch.optim import AdamW
from .MLP import MLP
import agent_code.ql_s_agent.utils as utils
from tensorboardX import SummaryWriter


# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

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

    # TODO: remove before submission!
    with open('explosion_map.pt', 'rb') as file:
        self.exploding_tiles_map = pickle.load(file)

    # Create a TensorboardX writer
    self.writer = SummaryWriter(log_dir='logs')

    # Setup models
    self.optimizer = AdamW(self.model.parameters(), lr=hp.LEARNING_RATE, amsgrad=True)

    if os.path.isfile("model/custom_mlp_policy_model.pth"):
        self.model.load_state_dict(torch.load('model/custom_mlp_policy_model.pth'))
    self.steps_done = 0

    self.eps_threshold = hp.EPS_START
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
    rewards = reward_from_events(self, old_game_state, self_action, new_game_state, events)
    self.episode_reward = rewards

    # Convert feature matrices to tensors
    old_game_state_feature, b = state_to_features(old_game_state)
    test = old_state_to_features(old_game_state)
    new_game_state_feature, c = state_to_features(new_game_state)
    #test2 = feature.state_to_features(new_game_state)

    # Put rewards into tensor
    reward = torch.tensor([rewards])
    action = torch.tensor([utils.action_string_to_index(self_action)], dtype=torch.long)

    self.episodes_round += 1
    self.steps_done += 1
    new_state_q_values = self.model(new_game_state_feature).max(-1)[0]

    td_target = reward + hp.DISCOUNT_FACTOR * new_state_q_values
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
    final_reward = reward_from_events(self, last_game_state, last_action, None, events)
    self.episode_reward = final_reward

    # Convert final reward to a tensor
    final_reward = torch.tensor([final_reward], dtype=torch.float32)

    # Compute Q-value of the last state-action pair
    last_state_feature, b = state_to_features(last_game_state)
    last_action = torch.tensor([utils.action_string_to_index(last_action)], dtype=torch.long)
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
    if hp.PLOT_TRAINING:
        with open('model/training_episodes.pkl', 'wb') as f:
            pickle.dump(self.episode_durations, f)
        with open('model/training_coins_collected.pkl', 'wb') as f:
            pickle.dump(self.episodes_coins_collected, f) 
    # Update epsilon threshold for new round
    self.eps_threshold = hp.EPS_START * (1 - (len(self.episode_durations) / hp.NUMBER_EPISODE))
    if (len(self.episode_durations) % 1000 == 0):
        torch.save(self.model.state_dict(), f'mlp_model_after_{len(self.episode_durations)}.pth')

def reward_from_events(self, old_game_state, self_action, new_game_state, events: List[str]) -> int:
    '''
        Input: self, list of events
        Output: sum of rewards resulting from the events
    '''
    # TODO: Divide rewards by 10 :Done
    # TODO: Split rewards into game rewards and move rewards. If event in moves, then += -1: Done
    game_rewards = {
        e.COIN_COLLECTED: 0.8,
        e.KILLED_OPPONENT: 2,
        e.KILLED_SELF: -8,
        e.INVALID_ACTION: -0.4
    }
    move_rewards = {
        e.WAITED: -0.1,
        e.MOVED_DOWN: -0.1,
        e.MOVED_LEFT: -0.1,
        e.MOVED_RIGHT: -0.1,
        e.MOVED_UP: -0.1,
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
        if event in move_rewards:
            reward_sum += -0.1
    
    reward_sum += get_custom_rewards(self, old_game_state, self_action, new_game_state, events)
        
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum * 10

def get_custom_rewards(self, old_game_state, self_action, new_game_state, events):
    '''
        Input: self, old_game_state, self_action, new_game_state, events
        Output: sum of rewards resulting from the events
    '''

    cust_rewards = 0

    if new_game_state is not None:

        old_self_position = utils.get_own_position(old_game_state)
        new_self_position = utils.get_own_position(new_game_state)

        # Reward for moving closer to coins
        if len(old_game_state['coins']) > 0 and len(new_game_state['coins']) > 0:
            try:
                old_coin_distance = utils.get_min_distance(old_self_position, old_game_state['coins'])
                new_coin_distance = utils.get_min_distance(new_self_position, new_game_state['coins'])
                coin_dist = old_coin_distance - new_coin_distance
                if coin_dist > 0:
                    # cust_rewards += coin_dist * 0.1
                    cust_rewards += 0.3
            except:
                pass

        # Evaluate danger of new position
        # Give positive rewards for avoiding danger and negative for staying or moving into danger
        explosion_present = np.any(old_game_state['explosion_map'] != 0)
        if len(old_game_state['bombs']) > 0 or explosion_present:
            danger_map = get_danger_map(old_game_state['field'], old_game_state['bombs'], old_game_state['explosion_map'])
            danger_of_new_position = danger_map[new_self_position[0], new_self_position[1]]
            danger_of_old_position = danger_map[old_self_position[0], old_self_position[1]]
            if danger_of_new_position != 0 or danger_of_new_position != 0:
                if danger_of_new_position < danger_of_old_position:
                    cust_rewards += 0.5
                else: 
                    cust_rewards -= 0.5

        # Reward for moving closer to crates if agent could place a bomb
        crate_positions_old = utils.get_crate_positions(old_game_state)
        crate_positions_new = utils.get_crate_positions(new_game_state)
        if utils.is_bomb_available(old_game_state) and len(crate_positions_old) > 0 and len(crate_positions_new) > 0:
            try:
                old_crate_distance = utils.get_min_distance(old_self_position, crate_positions_old)
                new_crate_distance = utils.get_min_distance(new_self_position, crate_positions_new)
                crate_dist = old_crate_distance - new_crate_distance
                if crate_dist > 0:
                    cust_rewards += 0.15
            except:
                pass

        # Rewards for good bomb placement
        others_positions = utils.get_others_positions(old_game_state)
        if self_action == "BOMB" and (e.INVALID_ACTION not in events) and len(others_positions) > 0:
            # Reward for bomb next to crate
            if old_crate_distance == 1:
                cust_rewards += 0.2
            # Penalty for bomb not next to crate
            else:
                cust_rewards -= 0.2

            # Reward for bomb next to enemy
            try:
                old_others_distance = utils.get_min_distance(old_self_position, others_positions)
                new_others_distance = utils.get_min_distance(new_self_position, others_positions)
                if len(others_positions) != 0:
                    if old_others_distance == 1:
                        cust_rewards += 0.8

                    if old_others_distance <= 3:
                        cust_rewards += 0.1
            except:
                pass
        if len(others_positions) != 0:
            old_others_distance = utils.get_min_distance(old_self_position, others_positions)
            new_others_distance = utils.get_min_distance(new_self_position, others_positions)            
            # Reward for chasing opponent
            if old_others_distance > new_others_distance:
                cust_rewards += 0.2
            else:
                cust_rewards -= 0.2
    comparison = 0
    comparison = aux_events(self, old_game_state, self_action, new_game_state, events)
    bla = ""
    return cust_rewards


    

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

def reward_from_events_old(self, events: List[str]) -> int:
    '''
        Input: self, list of events
        Output: sum of rewards resulting from the events
    '''
    # TODO: Divide rewards by 10 :Done
    # TODO: Split rewards into game rewards and move rewards. If event in moves, then += -1: Done
    old_game_rewards = {
        e.COIN_COLLECTED: 0.8,
        e.KILLED_OPPONENT: 2,
        e.KILLED_SELF: -8,
        e.INVALID_ACTION: -0.4,
        COIN_CHASER: 0.3,             
        MOVED_OUT_OF_DANGER: 0.5,
        STAYED_NEAR_BOMB: -0.5,
        MOVED_INTO_DANGER: -0.5,
        CRATE_CHASER: 0.15,
        BOMB_NEXT_TO_CRATE: 0.2,
        BOMB_NOT_NEXT_TO_CRATE: -0.2,
        DROPPED_BOMB_NEAR_ENEMY: 0.1,
        DROPPED_BOMB_NEXT_TO_ENEMY: 0.8, 
        OPPONENT_CHASER: 0.2
    }
    move_rewards = {
        e.WAITED: -0.1,
        e.MOVED_DOWN: -0.1,
        e.MOVED_LEFT: -0.1,
        e.MOVED_RIGHT: -0.1,
        e.MOVED_UP: -0.1,
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
        if event in move_rewards:
            reward_sum += -0.1
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


# TODO: Change this to get_custom_rewards()
# Return int, sum of custom rewards
# Each custom reward is seperate method. Sum up custom rewards and return value
def aux_events(self, old_game_state, self_action, new_game_state, events):
    old_game_rewards = {
        COIN_CHASER: 0.3,             
        MOVED_OUT_OF_DANGER: 0.5,
        STAYED_NEAR_BOMB: -0.5,
        MOVED_INTO_DANGER: -0.5,
        CRATE_CHASER: 0.15,
        BOMB_NEXT_TO_CRATE: 0.2,
        BOMB_NOT_NEXT_TO_CRATE: -0.2,
        DROPPED_BOMB_NEAR_ENEMY: 0.1,
        DROPPED_BOMB_NEXT_TO_ENEMY: 0.8, 
        OPPONENT_CHASER: 0.2
    }
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
    cust_rew = 0
    for event in events:
        if event in old_game_rewards:
            cust_rew += old_game_rewards[event]
    return cust_rew