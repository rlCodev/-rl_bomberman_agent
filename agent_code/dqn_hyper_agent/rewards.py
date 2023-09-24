
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