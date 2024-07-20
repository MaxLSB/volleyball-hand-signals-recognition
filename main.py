import cv2
import torch
import numpy as np
import utils.extraction as extraction
import utils.poseDetection as poseDetection
from models.model import DetectionModel
import mediapipe as mp
import torch.nn.functional as F 

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

actions = np.array(['pointL', 'pointR', 'TimeOut', 'OutofBd', 'NetFault', 'Substi'])
model = DetectionModel(len(actions))
model.load_state_dict(torch.load('trained_model/refereeSignalsModel.pth'))
model.eval()

colors = [(245,117,16), (117,245,16), (16,117,245), (255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

# 1. New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.7

cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = poseDetection.mediapipe_detection(frame, holistic)
        
        # Draw landmarks
        poseDetection.draw_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = extraction.extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            # res = model.predict(np.expand_dims(sequence, axis=0))[0]
            # print(actions[np.argmax(res)])
            # predictions.append(np.argmax(res))
            sequence_tensor = torch.tensor(np.expand_dims(sequence, axis=0), dtype=torch.float32)
            with torch.no_grad():
                logits = model(sequence_tensor)
                res = F.softmax(logits, dim=1).numpy()[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))
            
            
        #3. Viz logic
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])
                        
            if len(sentence) > 5: 
                sentence = sentence[-5:]

            # Viz probabilities
            image = prob_viz(res, actions, image, colors)
            
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()