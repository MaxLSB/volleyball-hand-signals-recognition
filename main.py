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

actions = np.array(['Neutral', 'pointL', 'pointR', 'TimeOut', 'OutofBd', 'NetFault', 'DbHit'])
model = DetectionModel(len(actions))
model.load_state_dict(torch.load('trained_model/refereeSignalsModel.pth'))
model.eval()

colors = [(245,117,16), (117,245,16), (16,117,245), (255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255)]

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,160+num*40), (int(prob*17*len(action_fullname(actions[num]))), 190+num*40), colors[num], -1)
        cv2.putText(output_frame, action_fullname(actions[num]), (0, 185+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

# Change the action name to a more readable format (adjust if needed)
def action_fullname(action):
    if action == 'pointL':
        return 'Point Left'
    elif action == 'pointR':
        return 'Point Right'
    elif action == 'TimeOut':
        return 'Time Out'
    elif action == 'OutofBd':
        return 'Out of Bounds'
    elif action == 'NetFault':
        return 'Net Fault'
    elif action == 'DbHit':
        return 'Double Hit'
    else:
        return 'Neutral'

sequence = []
sentence = []
printed_sentence = []
predictions = []
threshold = 0.4


# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cap = cv2.VideoCapture('exemple/video.mp4')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

output_video_path = 'exemple/test.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # while cap.isOpened():
    while True:

        ret, frame = cap.read()
        if not ret:
            break
        
        image, results = poseDetection.mediapipe_detection(frame, holistic)
        
        # Draw landmarks if you want
        # poseDetection.draw_landmarks(image, results)
        
        keypoints = extraction.extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            sequence_tensor = torch.tensor(np.expand_dims(sequence, axis=0), dtype=torch.float32)
            with torch.no_grad():
                logits = model(sequence_tensor)
                res = F.softmax(logits, dim=1).numpy()[0]
                predicted_label = np.argmax(res)
            if res[predicted_label] > threshold: 
                predictions.append(predicted_label)
            else :
                predictions.append(0) # Neutral label index (change if needed)

            # Check if at least 15 out of the last 20 predictions are the same
            if predictions[-20:].count(predicted_label) >= 15:
                    if len(sentence) > 0: 
                        # We don't want to detect the same action multiple times for a single motion
                        if actions[predicted_label] != sentence[-1]:
                            sentence.append(actions[predicted_label])
                            if actions[predicted_label] != 'Neutral':
                                printed_sentence.append(action_fullname(actions[predicted_label]))
                    else:
                        sentence.append(actions[predicted_label])
                        if actions[predicted_label] != 'Neutral':
                                printed_sentence.append(action_fullname(actions[predicted_label]))
    
            if len(sentence) > 3: 
                sentence = sentence[-3:]
                printed_sentence = printed_sentence[-3:]

            #Viz probabilities
            image = prob_viz(res, actions, image, colors)
            
        rect_x1, rect_x2 = (frame_width - 900) // 2, (frame_width + 900) // 2
        text_size = cv2.getTextSize(' -> '.join(printed_sentence), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (frame_width - text_size[0]) // 2

        # Draw text and background
        cv2.rectangle(image, (rect_x1, 0), (rect_x2, 60), (40, 40, 40), -1)
        cv2.putText(image, ' -> '.join(printed_sentence), (text_x, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('OpenCV Feed', image)
        
        # Write the frame into the output video
        out.write(image)
        
        # q to quit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()