import numpy as np

def all_actions():
    return np.array(['pointR', 'pointL', 'Substi', 'DbHit', 'OutofB']) # same order as all_actions_training

def all_actions_training():
    return np.array(['pointR', 'pointL', 'Substi', 'DbHit', 'OutofB', 'Nothing']) # We add the a 'Nothing' action for training

def action_fullname(action):
    if action == 'pointL':
        return 'Point Left'
    elif action == 'pointR':
        return 'Point Right'
    elif action == 'Substi':
        return 'Substitution'
    elif action == 'DbHit':
        return 'Double Hit'
    elif action == 'OutofB':
        return 'Out of Bounds'