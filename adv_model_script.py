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
import glob

CENTER_SIZE = 35 
LABELS = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
MAX_IMAGES_PER_CLASS = 10

def load_images():
    images = dict()
    
    for label in LABELS:
        files = glob.glob("images/square_%dp/squares_%dp_"+label+"_*.png" % (CENTER_SIZE, CENTER_SIZE)
        images[label] = [image.load_img(f, target_size=(CENTER_SIZE, CENTER_SIZE)) for f in files[:IMAGES_PER_CLASS]]

    return images



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
        
        start = int((299 - CENTER_SIZE) / 2)
        end = int((299 - CENTER_SIZE) / 2 + CENTER_SIZE)

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

input_tensor = Input(shape=(299, 299, 3))
inputs = Input(shape=(CENTER_SIZE, CENTER_SIZE, 3))

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

def expand_images(images):
    for key in images.keys():
        ls = images[key]
        if(len(ls)<1000):
            while(len(ls)<1000):
                ls = ls+ls
            images[key]=ls
            

images = load_images()
expand_images(images)
input_list = []
output_list = []
for key in images.keys():
    for value in images[key][:1000]:
        new_value = np.asarray(value)/255
        new_value = (new_value-0.5)*2
        input_list.append(np.asarray(new_value))
        output_list.append(key)
   #while(totalNr<100):
   #     input_list.append(np.asarray(value))
   #     output_list.append(array)
   #     totalNr+=1

input_a = np.array(input_list)
output_a = np.array(output_list)
output_a = to_categorical(output_a, num_classes=1000)
x_train, x_valid, y_train, y_valid = train_test_split(input_a, output_a, test_size=0.10, shuffle= True)
model.fit(x_train, y_train,
                epochs=100,
                batch_size=25,
                validation_data=(x_valid, y_valid))
pred_vals = []
for i in input_list:
    pred_val = i.reshape(1,CENTER_SIZE,CENTER_SIZE,3)
    pred_vals.append(np.argmax(model.predict(pred_val)))
# Save model .hd5
#model.save("adv_reprogramming_inception3.h5")
# Save weights .json
adv_layer_weights = model.get_layer('adv_layer_1').get_weights() # return numpy array containing 299 elements of size 299x3
adv_layer = {}
adv_layer["weights"] = adv_layer_weights[0].tolist()
print(adv_layer)
now = datetime.now()
now_string = now.strftime("%d-%m-%Y_%H-%M-%S")

if not path.exists("results/adv/"):
            os.makedirs("results/adv/")

with open("results/adv/adv_layer-%d.json" % now_string, 'w') as outfile:
    json.dump(adv_layer, outfile)
