import cv2
import mediapipe as mp
import os 
import pickle
from config import *

EMBEDDINGS = []
THE_EMBEDDINGS = {}
CURRENT_EMBEDDING = None
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)


def generate():
    #intialize the embeddings to be stored 
    global THE_EMBEDDINGS
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        THE_EMBEDDINGS[letter] = []
    #generate the embeddings
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        for i in range(20):
            print(f"Generating embeddings for {letter} {i}",end="\r")
            # Read the image.
            in_file = f'raw/{letter}/{i}.png'
            out_file = f'emb/{letter}/{i}.png'
            image = cv2.imread(in_file)
            #convert to rgb
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Process the image.
            results = hands.process(image_rgb)
            # Draw the hand embeddings on the image.
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # modifing the landmarks to be stored
                    myLandmarks = [[int(landmark.x*100),int(landmark.y*100),int(landmark.z*100)] for landmark in hand_landmarks.landmark]                    
                    THE_EMBEDDINGS[letter].append(myLandmarks)
                   

def pickleEmbeddings(file = PICKLE_FILE):
    global THE_EMBEDDINGS
    generate()
    with open(file, 'wb') as file:
        pickle.dump(THE_EMBEDDINGS, file)

def printEmbedings(file = PICKLE_FILE):
    with open(file, 'rb') as file:
        THE_EMBEDDINGS = pickle.load(file)
        for i in THE_EMBEDDINGS:
            cur = THE_EMBEDDINGS[i]
            print(i," : ",len(cur),end=" ")
            for j in cur:
                print(len(j),end=" ")
            print()


if __name__ == '__main__':
    pickleEmbeddings(PICKLE_FILE)
