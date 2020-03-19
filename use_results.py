import keras
import tensorflow as tf
import adv_model_script as ams
from keras.engine.topology import Layer
from keras.layers import Input
from keras.applications import inception_v3, inception_resnet_v2, resnet
from keras.preprocessing import image
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import backend as K
from sklearn.model_selection import train_test_split
import glob
import math
import numpy as np
import h5py

import sys
import os

def returnProbabilities(position):
    ProbabilityList = []
    for i in results.keys():
        allClasses = []
        classCount = []
        for j in results[i]:
            allClasses.append(j[position][1])
        for j in set(allClasses):
            classCount.append((j, allClasses.count(j)/len(allClasses)))
            classCount.sort(key=lambda x: x[1], reverse=True)
        ProbabilityList.append((i, classCount))
    ProbabilityList.sort(key=lambda x: x[0])
    for i,j in ProbabilityList:
        print(i)
        print(j)
    return ProbabilityList


inception = inception_v3.InceptionV3(weights='imagenet', input_tensor=Input(shape=(299, 299, 3)))
inception.trainable = False

dp = ams.DataPreprocessing("squares", 35, [1, 2, 3, 4, 5, 6, 7, 8, 9], 100)
x, y = dp.createData()

filename = sys.argv[1]
if not os.path.isfile(filename):
    print('file %s not found' % filename)
    exit(1)

y = to_categorical(y, num_classes=1000, dtype='float32')
hf = h5py.File(filename)
array = np.array(hf['model_weights']['adv_layer_1']['adv_layer_1']['kernel:0'])

results = ams.testResults(inception, 299, 35, array, list(zip(x, y)))

Probability1 = returnProbabilities(0)
Probability2 = returnProbabilities(1)
Probability3 = returnProbabilities(2)
for i in range(0, len(Probability1)):
    print("ID = "+str(Probability1[i][0]))
    print("First choice: "+str(Probability1[i][1]))
    print("Second choice: "+str(Probability2[i][1]))
    print("Third choice: "+str(Probability3[i][1]))
    
#print(tf.nn.l2_loss(array))
