import cv2
import torch
import numpy as np
import mediapipe as mp
import torch.nn.functional as F 
import utils.extraction as extraction
import utils.poseDetection as poseDetection
from models.model import DetectionModel
from utils.actions import action_fullname, all_actions

################################################
# Try the model in real time on your webcam
################################################

def prob_viz(res, actions, input_frame):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.putText(output_frame, action_fullname(actions[num]) + ': ' + str(round(prob.item(), 2)), (0, 185+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

def main(ViewProbabilities=True, ViewLandmarks=True):
    actions = all_actions()
    model = DetectionModel(len(actions))
    model.load_state_dict(torch.load('trained_model/refereeSignalsModel.pth'))
    model.eval()

    sequence = []
    sentence = []
    printed_sentence = []
    predictions = []
    threshold = 0.5

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mp_holistic = mp.solutions.holistic

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:  
        while cap.isOpened(): 
            ret, frame = cap.read()
            
            image, results = poseDetection.mediapipe_detection(frame, holistic)
            
            if ViewLandmarks:
                poseDetection.draw_landmarks(image, results) # Uncomment to see the landmarks on the video
            
            keypoints = extraction.extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                sequence_tensor = torch.tensor(np.expand_dims(sequence, axis=0), dtype=torch.float32)
                with torch.no_grad():
                    res = model(sequence_tensor)[0]
                    predicted_label = res.argmax(dim=0).item()
                    proba = res[predicted_label].item()
                if proba > threshold: 
                    predictions.append(predicted_label)

                # Check if at least 15 out of the last 20 predictions are the same as the predicted label
                if predictions[-15:].count(predicted_label) >= 12:
                        if len(sentence) > 0: 
                            # We don't want to detect the same action multiple times for a single motion
                            if actions[predicted_label] != sentence[-1]:
                                sentence.append(actions[predicted_label])               
                                printed_sentence.append(action_fullname(actions[predicted_label]))
                        else:
                            sentence.append(actions[predicted_label])
                            printed_sentence.append(action_fullname(actions[predicted_label]))
        
                if len(sentence) > 3: 
                    sentence = sentence[-3:]
                    printed_sentence = printed_sentence[-3:]

                if ViewProbabilities:
                    image = prob_viz(res, actions, image)
                
            rect_x1, rect_x2 = (frame_width - 900) // 2, (frame_width + 900) // 2
            rect_y1, rect_y2 = frame_height - 110, frame_height -50
            text_size = cv2.getTextSize(' -> '.join(printed_sentence), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (frame_width - text_size[0]) // 2

            cv2.rectangle(image, (rect_x1, rect_y1), (rect_x2, rect_y2), (40, 40, 40), -1)
            cv2.putText(image, ' -> '.join(printed_sentence), (text_x, frame_height -70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', image)
            
            # q to quit
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        
if __name__ == '__main__':
    main(ViewProbabilities=True, ViewLandmarks=True)
