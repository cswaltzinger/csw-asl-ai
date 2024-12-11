import mediapipe as mp
import cv2 
from tensorflow.keras.models import load_model
import numpy as np
from config import *

#load embedings 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
# Initialize MediaPipe Drawing module for drawing landmarks
mp_drawing = mp.solutions.drawing_utils
#load our trained model 
model = load_model(MODEL)



# Open a video capture object (0 for the default camera)
cap = cv2.VideoCapture(0)
lastPrinted = 0

prev = None 
sameCount = 0 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the frame to detect hands
    results = hands.process(frame_rgb)
    # Check if hands are detected
    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:
            # extract needed position of landmarks 
            myLandmarks = [[int(landmark.x*100),int(landmark.y*100),int(landmark.z*100)] for landmark in hand_landmarks.landmark]                    
            # if all landmarks are detected, then reahape and predict
            if len(myLandmarks) == 21 :
                # make new landmarks to put in model
                newLandmark = np.array(myLandmarks).flatten()
                newLandmark = np.array([newLandmark])
                predictions = model.predict(newLandmark)
                conf = np.argmax(predictions[0])
                #print the result if decected multiple times in a row
                if prev == None:
                    prev = conf
                else:
                    if prev != conf:
                        prev = conf
                        sameCount = 0
                    else:
                        sameCount += 1
                         # if the same result is detected multiple times in a row, print it and reset the count 
                        if sameCount > 10:
                            print(chr(conf+65),end=" ",flush=True)
                            sameCount = 0
            # Draw the landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            break
    else:
        # if no hand is detected, reset the count and the prev variable and print a space 
        if prev != None:
            print("",end=" ",flush=True)
        prev = None
        sameCount = 0
    frame = cv2.flip(frame, 1)
    # Display the frame with hand landmarks
    cv2.imshow('Hand Recognition', frame)
    
    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the video capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
print("\n\n")