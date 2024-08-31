import os
import numpy as np

def pre_processing(actions, DATA_PATH, sequence_length):
    label_map = {label:num for num, label in enumerate(actions)}
    sequences, labels = [], []
    for action in actions:
        for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
                window.append(res)
            sequences.append(window)
            sig_label = [0]*(len(actions)-1) # -1 because we don't want to include the 'Nothing' action
            if action != 'Nothing':
                sig_label[label_map[action]] = 1
            labels.append(sig_label)
            
    return sequences, labels
