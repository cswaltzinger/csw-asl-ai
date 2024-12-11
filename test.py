import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import random
from config import *
from embedings import *

model = load_model(MODEL)

# randomly select 100 entries from the dataset
tries = [random.randint(0,len(EMB)-1) for i in range(100)]
x_test = np.array([EMB[i] for i in tries])
y_test =[EMBL[i] for i in tries]

# predict the output
predictions = model.predict(np.array(x_test))

# count correct predictions
correctCount = 0

#print the key for the output
print("\tPrediction|Actual|Confidance\tPredictions")
for index in range(len(predictions)):
    # get/format the prediction and actual values 
    label = y_test[index]
    prediction = np.argmax(predictions[index])
    showPredictions = [f'{pred:.2f}'[:5] if pred > 0.01 else "0" for pred in predictions[index]]
    predicted_accuracy = str(predictions[index][label])
    if "e" in predicted_accuracy:
        predicted_accuracy = predicted_accuracy[:5]+predicted_accuracy[predicted_accuracy.index("e"):]
    if len(predicted_accuracy) > 9:
        predicted_accuracy = predicted_accuracy[:9]
    while len(predicted_accuracy) < 9:
        predicted_accuracy += " "
    label = chr(label+ord('a'))
    prediction = chr(prediction+ord('a'))
    if label == prediction:
        correctCount += 1
    spred =str(showPredictions).replace("'","")#.replace(",","  ").replace("[","").replace("]","")
    print(f"\t{prediction}|{label}|{predicted_accuracy}\t\t{spred}")

# print the accuracy
print(f"Accuracy: {correctCount/len(predictions)}")
