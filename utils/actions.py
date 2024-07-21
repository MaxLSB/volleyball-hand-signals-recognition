import numpy as np

def all_actions():
    return np.array(['Neutral', 'pointL', 'pointR', 'Substi', 'DbHit', 'OutofB'])

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
    else:
        return 'Neutral'