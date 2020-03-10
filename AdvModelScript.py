from keras.datasets import mnist
import numpy as np
import json
import os
import os.path
import keras
import tensorflow as tf
from keras.layers import Input, Layer
from keras_applications import inception_v3, inception_resnet_v2, resnet
from keras.preprocessing import image
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from math import log, inf
from PIL import Image
from keras import backend as K


def evaluate(images: {str: []}, adv_program: []):
    results = {}
    for k, class_images in images.items():
        class_results = []
        for img in class_images:
            adv_program[start:end, start:end, :] = img
#             Append best match
            class_results.append(classify(adv_program, I_V3)[0][1])
        
        results[k] = class_results
    return results

flatten = lambda l: [item for sublist in l for item in sublist]
valueDict = dict()
keyDict = dict()

def results_to_matrix(results):
    keys = results.keys()
    for key in keys:
        if(key not in keyDict.keys()):
            keyDict[key]=len(keyDict.keys());
    values = set(flatten(results.values()))
    for val in values:
        if(val not in valueDict.keys()):
            valueDict[val]=len(valueDict.keys())
 #   if(isinstance(matrix, list)):
    matrix = np.zeros((len(keys), len(valueDict.keys())))
    for i in results.keys():
        for j in results[i]:
            matrix[keyDict[i]][valueDict[j]]+=1
    #for value in results.values():
    #    el=[]
    #    for c in classes: el.append(value.count(c))
    #    matrix.append(el)
    return matrix

def computeMatrixLoss(matrix):
    matrix = matrix/matrix.sum(axis=1)[:,None]
    usedLabels = []
    loss=0
    matrix = matrix.transpose()
    for i in matrix:
        highestProbability = 0
        probabilityID = inf
        for j in range(0,len(i)):
            if(j not in usedLabels):
                if(i[j]>=highestProbability):
                    highestProbability=i[j]
                    probabilityID=j
        usedLabels.append(probabilityID)
        if(highestProbability==0):
            loss = loss+10
        else:
            loss = loss - log(highestProbability)
    loss = loss+(len(matrix[1])-len(matrix))*10
    return loss

def computeLoss(results: {str: []}):
    label_map = dict()
    loss = 0
    for k, class_result in results.items():
        
        most_common = max(set(class_result), key = class_result.count)
        count = class_result.count(most_common)
        probability = count/len(class_result)
        
        label_map[k]= most_common
        loss = loss - log(probability)
            
    print(loss)
    return loss

# computeLoss({'1':[1,2,2,2,2,3], '2': [2,3,4,3,3,3,1,3,34,4,4,4,4,4]})

CENTER_SIZE = 29
LABELS = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
import glob

def loadImages():
    images = dict()
    
    for label in LABELS:
        files = glob.glob("allSquares/squares"+label+"_*.png")
        images[label] = [image.load_img(f, target_size=(CENTER_SIZE, CENTER_SIZE)) for f in files[:10]]

    return images

def train():
    best_adv_program = []
    best_loss = inf
    
    images = loadImages()
    bestMatrix = []
    try:
        for i in itertools.count(0):

            adv_program = np.random.rand(299,299,3) * 255
            adv_program = adv_program.astype(int)

            result = evaluate(images, adv_program)
            mresult = results_to_matrix(result)
            #loss = computeLoss(result)
            loss = computeMatrixLoss(mresult)
            if loss < best_loss:
                best_loss = loss
                best_adv_program = adv_program
                bestMatrix = mresult
                clear_output(wait=True)
                display(i, best_loss)
            print(i, end='\r')

            if loss < 0.05:
                break
        return (best_adv_program, bestMatrix)
    except:
        return (best_adv_program, bestMatrix)


(xt, yt), (xte, yte) = mnist.load_data()
print(xt[0], yt[0])
center_size = 29
input_tensor = Input(shape=(299, 299, 3))

class AdvLayer(Layer):

    def __init__(self, **kwargs):
        super(AdvLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        img_shape = (299,299,3)
        self.adv_weights = self.add_weight(name='kernel', 
                                      shape=img_shape,
                                      initializer='uniform',
                                      trainable=True)
        super(AdvLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        
        start = int((299 - center_size) / 2)
        end = int((299 - center_size) / 2 + center_size)

        adv_img = np.full((299,299,3), 1, dtype=np.float)
        adv_img[start:end, start:end, :] = 0
        padx = tf.pad(tf.concat([x], axis=-1),
                      paddings = tf.constant([[0,0], [start, start], [start, start], [0,0]]))
        print(padx.shape)
        adv_img = tf.nn.tanh(tf.multiply(self.adv_weights, adv_img))+padx
        #adv_img[start:end, start:end, :] = x
        self.out_shape = adv_img.shape

        return adv_img

    def compute_output_shape(self, input_shape):
        return self.out_shape


inputs = Input(shape=(center_size, center_size, 3))
inception = inception_v3.InceptionV3(weights='imagenet', input_tensor=input_tensor,
    backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
inception.trainable = False
al = AdvLayer()(inputs)
outputs = inception(al)
model = Model(inputs=[inputs], outputs=[outputs])
print(model.summary())
model.compile(optimizer = Adam(lr=0.05, decay=0.96),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

images = loadImages()
inputList = []
outputList = []
for key in images.keys():
    for value in images[key]:
        array = np.zeros(1000)
        array[int(key)]=1
        newvalue = np.asarray(value)/255
        newvalue = (newvalue-0.5)*2
        inputList.append(np.asarray(value))
        outputList.append(array)
    totalNr = len(images[key])
    while(totalNr<10):
        inputList.append(np.asarray(value))
        outputList.append(array)
        totalNr+=1

inputa = np.array(inputList)
outputa = np.array(outputList)
print(inputa.shape)
print(outputa.shape)

model.fit(inputa, outputa,
                epochs=1,
                batch_size=25,
                validation_split=0.1)

# Save model .hd5
model.save("adv_reprogramming_inception3.h5")
# Save weights .json
adv_layer_weights = model.get_layer('adv_layer_1').get_weights() # return numpy array containing 299 elements of size 299x3
adv_layer = {}
adv_layer["weights"] = adv_layer_weights[0].tolist()
print(adv_layer)
with open("adv_layer.json", 'w') as outfile:
    json.dump(adv_layer, outfile)
