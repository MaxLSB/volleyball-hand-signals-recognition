import cv2
import os
import time
import numpy as np
from matplotlib import pyplot as plt
import mediapipe as mp
import src.dataCollection as dataCollection

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

DATA_PATH = os.path.join('data') # Path for exported data, numpy arrays
actions = np.array(['pointLeft', 'pointRight']) # Actions that we try to detect
no_sequences = 30 # Thirty videos worth of data
sequence_length = 30 # Videos are going to be 30 frames in length
start_folder = 30 # Folder start

dataCollection.data_collection(actions, DATA_PATH, no_sequences, sequence_length, start_folder)

# cap = cv2.VideoCapture(0)
# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#     while cap.isOpened():
#         ret, frame = cap.read()
        
#         image, results = poseDetection.mediapipe_detection(frame, holistic)
        
#         poseDetection.draw_landmarks(image, results)
        
#         cv2.imshow('Raw Webcam Feed', image)
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()
#     print('Done')
