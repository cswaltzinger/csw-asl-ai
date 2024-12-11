import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
from config import *
from embedings import *

# Define the model
model = Sequential([
    Dense(64, activation=ACT_FUNC, input_shape=(FEATURES,)), # Input layer with FEATURES input parameters
    Dense(64, activation=ACT_FUNC), 
    Dense(128, activation=ACT_FUNC),
    Dense(64, activation=ACT_FUNC), 
    Dense(32, activation=ACT_FUNC), 
    Dense(OUTPUT_CLASSES, activation='softmax') # Output layer with 26 outputs
])


# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Convert labels to one-hot encoding if they are not already
labels = tf.keras.utils.to_categorical(EMBL, num_classes=26)

# Train the model
TRAIN = np.array(EMB)
model.fit(TRAIN, labels, 
          epochs=EPOCHS, 
          batch_size=BATCH_SIZE
          )


# Save the model
model.save(f"{MODEL_DIR}/{MODEL_NAME}", save_format='h5')