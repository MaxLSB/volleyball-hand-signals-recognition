import cv2
import os
import numpy as np
import mediapipe as mp
import utils.poseDetection as poseDetection
import utils.extraction as extraction

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def data_collection(actions, DATA_PATH, no_sequences, sequence_length, start_folder):
    for action in actions: 
        for sequence in range(1,no_sequences+1):
            try: 
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass

    cap = cv2.VideoCapture(0)
    height = 1080
    width = 1920
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # Set mediapipe
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        # Loop through actions
        for action in actions:
            # Loop through sequences
            for sequence in range(start_folder, start_folder+no_sequences):
                # Loop through the frames
                for frame_num in range(sequence_length):

                    # Read frame
                    ret, frame = cap.read()

                    # Make detections
                    image, results = poseDetection.mediapipe_detection(frame, holistic)

                    # Draw landmarks
                    poseDetection.draw_landmarks(image, results)
                    
                    if frame_num == 0: 
                        cv2.putText(image, 'STARTING COLLECTION', (400,300), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (200,50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
                        
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(500)
                    else: 
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (200,50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
                        
                        cv2.imshow('OpenCV Feed', image)
                    
                    # Extract keypoints
                    keypoints = extraction.extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    # Exit if needed
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                        
        cap.release()
        cv2.destroyAllWindows()
        
