import os
import numpy as np
import mediapipe as mp
import utils.dataCollection as dataCollection

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

DATA_PATH = os.path.join('data/')
actions = np.array(['pointL', 'pointR']) # Actions that we try to detect
no_sequences = 30 # Thirty videos worth of data
sequence_length = 30 # Videos are going to be 30 frames in length
start_folder = 1 # Folder start

dataCollection.data_collection(actions, DATA_PATH, no_sequences, sequence_length, start_folder)

