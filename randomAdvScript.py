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
import glob
from math import log, inf
import itertools
import pickle
from datetime import datetime



def loadImages():
    images = dict()
    
    for label in LABELS:
        files = glob.glob("images/square_35p/squares_35p_"+label+"_*.png")
        print('images for label', label, len(files))
        images[label] = [image.load_img(f, target_size=(CENTER_SIZE, CENTER_SIZE)) for f in files[:10]]

    return images

def classify(img, model='I_V3'):

    # ugly
    if model == 'RN50':
        model = rn50_model
        application = resnet
    elif model == 'RN152':
        model = rn152_model
        application = resnet
    elif model == 'I_V3':
        model = i_v3_model
        application = inception_v3
    elif model == 'I_RN_V2':
        model = i_rn_v2_model
        application = inception_resnet_v2
        
    # preprocess
    img_prep = image.img_to_array(img)
    img_prep = np.expand_dims(img_prep, axis=0)
    img_prep = application.preprocess_input(img_prep,
        backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
    # predict
    predictions = model.predict(img_prep)
    #results
    return application.decode_predictions(predictions, top=2, utils=keras.utils)[0]

def evaluate(images: {str: []}, adv_program: []):
    results = {}
    for k, class_images in images.items():
        class_results = []
        for img in class_images:
            adv_program[START:END, START:END, :] = img
#             Append best match
            class_results.append(classify(adv_program, 'I_V3')[0][1])
        
        results[k] = class_results
    return results

flatten = lambda l: [item for sublist in l for item in sublist]
valueDict = dict()
keyDict = dict()

def results_to_matrix(results):
    keys = results.keys()
    for key in keys:
        if(key not in keyDict.keys()):
            keyDict[key]=len(keyDict.keys())
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

            loss = computeMatrixLoss(mresult)
            if loss < best_loss:
                best_loss = loss
                best_adv_program = adv_program
                bestMatrix = mresult

            print('round: ', i, 'best loss: ', best_loss, end='\r')

            if best_loss < 0.01:
                break
        return (best_adv_program, bestMatrix)
    except KeyboardInterrupt as e:
        print('\nkeyboardInterupt:', e)
        return (best_adv_program, bestMatrix)
    except Exception as e:
        print('\nexception:', e)
        return (best_adv_program, bestMatrix)


if __name__ == "__main__":
    CENTER_SIZE = 35
    LABELS = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    START = int((299 - CENTER_SIZE) / 2)
    END = int((299 - CENTER_SIZE) / 2 + CENTER_SIZE)
    
    input_tensor = Input(shape=(299, 299, 3))
    i_v3_model = inception_v3.InceptionV3(weights='imagenet', input_tensor=input_tensor,
        backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
    
    try:
        best_program, best_matrix = train()
        now = datetime.now()
        now_string = now.strftime("%d-%m-%Y_%H-%M-%S")
        pickle.dump(best_program, open('results/random/adv_program-'+now_string, 'wb'))
        pickle.dump(best_matrix, open('results/random/best_matrix-'+now_string, 'wb'))
    except Exception as e:
        print('error', e)

