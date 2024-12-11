
import pickle
import numpy as np
from config import *
import random 


#global variables
FILTERED_EMBEDDINGS = []
FILTERED_EMBEDDINGS_LABEL =[]
EMB = FILTERED_EMBEDDINGS
EMBL = FILTERED_EMBEDDINGS_LABEL 
THE_EMBEDDINGS = None


#reading the embeddings from the pickle file
with open(PICKLE_FILE, 'rb') as file:
    THE_EMBEDDINGS = pickle.load(file)
    for letter in THE_EMBEDDINGS:
        cur = THE_EMBEDDINGS[letter]
        for j in cur:
            if len(j) == 21:
                new_j = np.array(j).flatten()
                FILTERED_EMBEDDINGS.append(new_j)
                FILTERED_EMBEDDINGS_LABEL.append(ord(letter)-ord('a'))



#print metrics 
def metrics():
    print("NUM: ",len(EMB))
    print()
    sum = EMB[0].copy()
    for index in range(1,len(EMB)):
        emb = EMB[index]
        embl = EMBL[index]
        sum = np.add(sum,emb)
    avg = np.divide(sum,len(EMB))
    print("AVG: ",avg)
    print()
    avg_dif = np.subtract(EMB[0],avg)
    for index in range(1,len(EMB)):
        emb = EMB[index]
        embl = EMBL[index]

        dif = np.subtract(emb,avg)
        avg_dif = np.add(avg_dif,dif)
    avg_dif = np.divide(avg_dif,len(EMB))
    print("MAD: ",avg_dif)
    print()
    i= random.randint(0,len(EMB)-1)
    print(EMBL[i],": ",EMB[i])




if __name__ == '__main__':
    metrics()