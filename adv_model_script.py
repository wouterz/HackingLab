from keras.datasets import mnist
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from math import log, inf
from PIL import Image
import glob

import keras
import tensorflow as tf
from keras.engine.topology import Layer
from keras.layers import Input
from keras.applications import inception_v3, inception_resnet_v2, resnet
from keras.preprocessing import image
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import backend as K
from sklearn.model_selection import train_test_split




def load_images() -> {str:[]}:
    images = dict()
    
    for label in LABELS:
        files = glob.glob("images/square_%dp/squares_%dp_%s_*.png" % (CENTER_SIZE, CENTER_SIZE, label))
        images[label] = [image.load_img(f, target_size=(CENTER_SIZE, CENTER_SIZE)) for f in files[:MAX_IMAGES_PER_CLASS]]

    return images

def expand_images(images: {str:[]}) -> {str:[]}:
    for key in images.keys():
        ls = images[key]
        if(len(ls)<MAX_IMAGES_PER_CLASS):
            while(len(ls)<MAX_IMAGES_PER_CLASS):
                ls = ls+ls
            images[key]=ls

class AdvLayer(Layer):

    def __init__(self, **kwargs):
        super(AdvLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        img_shape = (IMAGE_SIZE,IMAGE_SIZE,3)
        self.adv_weights = self.add_weight(name='kernel', 
                                      shape=img_shape,
                                      initializer='uniform',
                                      trainable=True)
        super(AdvLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        
        start = int((IMAGE_SIZE - CENTER_SIZE) / 2)
        end = int((IMAGE_SIZE - CENTER_SIZE) / 2 + CENTER_SIZE)
        input_mask = np.pad(np.zeros([1,CENTER_SIZE,CENTER_SIZE,3]), [[0,0], [start, start], [start, start], [0,0]], 'constant', constant_values=1)
        mask = tf.constant(input_mask, dtype=tf.float32)
        #adv_img = np.full((IMAGE_SIZE,IMAGE_SIZE,3), 1, dtype=np.float)
        #adv_img[start:end, start:end, :] = 0
        padx = tf.pad(x,
                      paddings = tf.constant([[0,0], [start, start], [start, start], [0,0]]))
        print(padx)
        adv_img = tf.nn.tanh(tf.multiply(self.adv_weights, mask))+padx
        #adv_img[start:end, start:end, :] = x
        self.out_shape = adv_img.shape

        return adv_img

    def compute_output_shape(self, input_shape):
        return self.out_shape

def testResults(adv_layer, image_set):
    start = int((IMAGE_SIZE - CENTER_SIZE) / 2)
    end = int((IMAGE_SIZE - CENTER_SIZE) / 2 + CENTER_SIZE)
    predictions = []
    predictionDict = {}
    for (i,j) in image_set:
        key = np.argwhere(j==1)[0][0]
        adv_layer[start:end,start:end,:]=i;
        adv_layer2 = adv_layer.reshape(1,299,299,3)
        prediction = inception.predict(adv_layer2)
        if key not in predictionDict.keys():
            predictionDict[key]=[]
        
        predictionDict[key].append(inception_v3.decode_predictions(prediction, top=2, utils=keras.utils)[0])
    return predictionDict

def label_mapping():
    imagenet_label = np.zeros([1000, len(LABELS)])
    imagenet_label[0:len(LABELS), 0:len(LABELS)] = np.eye(len(LABELS))
    return tf.constant(imagenet_label, dtype=tf.float32)
if __name__ == "__main__":
    ### SETUP PARAMETERS ###
    CENTER_SIZE = 35
    IMAGE_SIZE = 299
    LABELS = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    MAX_IMAGES_PER_CLASS = 100

    ADAM_LEARN_RATE = 0.05
    ADAM_DECAY = 0.96

    TEST_SIZE= 0.10
    EPOCHS = 1000
    BATCH_SIZE = 25
    ### END SETUP PARAMETERS ###

    # Load / prepare data
    images = load_images()
    expand_images(images)
    input_list = []
    output_list = []
    for key in images.keys():
        for value in images[key][:MAX_IMAGES_PER_CLASS]:
            new_value = np.asarray(value, dtype=np.float32)
            new_value = tf.convert_to_tensor(new_value, dtype=tf.float32)
            new_value/=255.
            new_value-=0.5
            new_value*=2.0
            input_list.append(new_value)
            output_list.append(key)
    print("List compiled")
    #while(totalNr<100):
    #     input_list.append(np.asarray(value))
    #     output_list.append(array)
    #     totalNr+=1
    print(len(input_list))
    #input_a = np.array(input_list)
    input_a = np.array(tf.convert_to_tensor(input_list))
    print("Creating output")
    output_a = np.array(output_list, dtype=np.float32)
    output_a = to_categorical(output_a, num_classes=1000, dtype='float32')
    print("Creating train and validation")
    x_train, x_valid, y_train, y_valid = train_test_split(input_a, output_a, test_size=TEST_SIZE, shuffle= True)
    print("Train and validation compiled")
    # Setup model
##    for i in range(0,3):
##        oldImage = x_valid[i]
##        oldImage = oldImage/2.0
##        oldImage = oldImage+0.5
##        oldImage = oldImage*255
##        oldImage = oldImage.astype(np.uint8)
##        pil_image = Image.fromarray(oldImage, mode="RGB")
##        #plt.imshow(pil_image)
##        val = np.argmax(y_valid[i])
##        print(val)
##
##        pil_image.show(title=str(val))

    # Original model

    print("Compiling model")
    inception = inception_v3.InceptionV3(weights='imagenet', input_tensor=Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
    inception.trainable = False

    # Adv Layer
    inputs = Input(shape=(CENTER_SIZE, CENTER_SIZE, 3))
    al = AdvLayer()(inputs)

    print("Compiling model")
    # Combine layers
    outputs = inception(al)

    print("Compiling model")
    model = Model(inputs=[inputs], outputs=[outputs])
    optimizer = Adam(lr=ADAM_LEARN_RATE, decay=ADAM_DECAY);
  # tf.compat.v1.disable_eager_execution()
    #print(outputs)
    special_output = tf.convert_to_tensor(output_a[0], dtype=tf.float32)
    #print(output_a[0])
    #cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = model(inputs) ,logits = outputs))
    #reg_loss = 2e-6 * tf.nn.l2_loss(al.get_weights())
    #loss = cross_entropy_loss + reg_loss
    #var_list = lambda: model.trainable_weights
    #optimizer.minimize(loss, var_list)
    def _loss_tensor(y_true, y_pred):
        disLogits = tf.matmul(y_pred, label_mapping())
        cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_true[:,:len(LABELS)], logits=disLogits))
        reg_loss = 2e-6 * tf.nn.l2_loss(model.get_layer('adv_layer_1').adv_weights)
        loss = cross_entropy_loss + reg_loss
        print(loss)
        return loss
    print("Compiling model")
    model.compile(optimizer = optimizer,
                            loss=_loss_tensor
                            ,metrics=['accuracy'])

    # Train
    history = model.fit(x=x_train, y=y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE)
                    #,validation_data=(x_valid, y_valid))

    #Hardcoded for now
##    predDict = {"1":[],"2":[],"3":[],"4":[],"5":[],"6":[],"7":[],"8":[],"9":[]}
##    for i in range(0, len(input_list)):
##        #pred_val = input_list[i].reshape(1,CENTER_SIZE,CENTER_SIZE,3)
##        predDict[output_list[i]].append(np.argmax(model.predict(input_list[i])))
##    for i in predDict.keys():
##        print(predDict[i], max(set(predDict[i]), key = predDict[i].count))
        
    # Write results

    # Save weights .json
    adv_layer_weights = model.get_layer('adv_layer_1').get_weights() # return numpy array containing 299 elements of size 299x3
    adv_layer = {}
    adv_layer["weights"] = adv_layer_weights[0].tolist()
    input_aR = []
    results = testResults(adv_layer_weights[0], list(zip(input_a, output_a)))
    for i in results.keys():
        print(str(i)+"\n")
        for j in results[i]:
            print(j)
    #if not path.exists("results/adv/"):
    #        os.makedirs("results/adv/")

    #now = datetime.now()
    #now_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    #with open("results/adv/adv_layer-%s.json" % now_string, 'w') as outfile:
    #    json.dump(adv_layer, outfile)
    
    
