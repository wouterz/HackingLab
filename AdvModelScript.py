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
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from math import log, inf
from PIL import Image
from keras import backend as K
from sklearn.model_selection import train_test_split

CENTER_SIZE = 29
LABELS = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
import glob

def loadImages():
    images = dict()
    
    for label in LABELS:
        files = glob.glob("images/square_29p/squares_29p_"+label+"_*.png")
        images[label] = [image.load_img(f, target_size=(CENTER_SIZE, CENTER_SIZE)) for f in files[:10]]

    return images


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

def expandImages(images):
    for key in images.keys():
        ls = images[key]
        if(len(ls)<1000):
            while(len(ls)<1000):
                ls = ls+ls
            images[key]=ls
            

images = loadImages()
expandImages(images)
inputList = []
outputList = []
for key in images.keys():
    for value in images[key][:1000]:
        newvalue = np.asarray(value)/255
        newvalue = (newvalue-0.5)*2
        inputList.append(np.asarray(value))
        outputList.append(key)
   #while(totalNr<100):
   #     inputList.append(np.asarray(value))
   #     outputList.append(array)
   #     totalNr+=1

inputa = np.array(inputList)
outputa = np.array(outputList)
outputa = to_categorical(outputa, num_classes=1000)
x_train, x_valid, y_train, y_valid = train_test_split(inputa, outputa, test_size=0.10, shuffle= True)
model.fit(x_train, y_train,
                epochs=100,
                batch_size=25,
                validation_data=(x_valid, y_valid))
predVals = []
for i in inputList:
    predVal = i.reshape(1,29,29,3)
    predVals.append(np.argmax(model.predict(predVal)))
# Save model .hd5
#model.save("adv_reprogramming_inception3.h5")
# Save weights .json
adv_layer_weights = model.get_layer('adv_layer_1').get_weights() # return numpy array containing 299 elements of size 299x3
adv_layer = {}
adv_layer["weights"] = adv_layer_weights[0].tolist()
print(adv_layer)
with open("adv_layer.json", 'w') as outfile:
    json.dump(adv_layer, outfile)
