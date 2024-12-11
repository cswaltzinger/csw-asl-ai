#config module for the project


#embeding file
PICKLE_FILE = "xyz-3d.pkl"

#model config
FEATURES_PER_INPUT = 3
INPUT = 21
FEATURES = FEATURES_PER_INPUT*INPUT
OUTPUT_CLASSES = 26
#best are relu seli and tanh
ACT_FUNC = ['relu','linear',
            'selu','sigmoid',
            'tanh'][1]



MODEL_DIR="models/xyz"
MODEL_DIR="."
MODEL_NAME = "the-model.h5"
MODEL_NAME = "THE_MODEL.h5"
MODEL = f"{MODEL_DIR}/{MODEL_NAME}"

EPOCHS = 100
BATCH_SIZE = 26

#eval config
BSWITCH = True
NN = True