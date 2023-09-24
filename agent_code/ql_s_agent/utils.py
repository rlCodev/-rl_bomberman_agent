ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
# Function to convert action space index to string action
def action_index_to_string(action_index):
    return ACTIONS[action_index]

# Function to convert string action to action space index
def action_string_to_index(action_string):
    if action_string in ACTIONS:
        return ACTIONS.index(action_string)
    else:
        raise ValueError(f"Invalid action: {action_string}")