import numpy as np
import json
import math
from datetime import datetime
import os
import matplotlib.pyplot as plt
from math import log, inf
from PIL import Image
import glob
from datetime import datetime

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
        class_images = images[key]
        if(len(class_images) < MAX_IMAGES_PER_CLASS):
            while(len(class_images) < MAX_IMAGES_PER_CLASS):
                class_images = class_images+class_images
            images[key]=class_images[:MAX_IMAGES_PER_CLASS]


class AdvLayer(Layer):

    def __init__(self,image_size=0, center_size=0, **kwargs):
        self.image_size = image_size
        self.center_size = center_size
        super(AdvLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        img_shape = (self.image_size, self.image_size,3)
        self.adv_weights = self.add_weight(name='kernel', 
                                      shape=img_shape,
                                      initializer='uniform',
                                      trainable=True)
        super(AdvLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        start = math.ceil((self.image_size - self.center_size) / 2)
        start2 = math.ceil((self.image_size - self.center_size) / 2)
        if not self.center_size%2:
            start2 = int(math.floor((self.image_size - self.center_size) / 2))
        input_mask = np.pad(np.zeros([1,self.center_size,self.center_size,3]), [[0,0], [start, start2], [start, start2], [0,0]], 'constant', constant_values=1)
        mask = tf.constant(input_mask, dtype=tf.float32)
        padx = tf.pad(x,
                      paddings = tf.constant([[0,0], [start, start2], [start, start2], [0,0]]))
        print(padx)
        adv_img = tf.nn.tanh(tf.multiply(self.adv_weights, mask))+padx
        #adv_img[start:end, start:end, :] = x
        self.out_shape = adv_img.shape

        return adv_img

    def compute_output_shape(self, input_shape):
        return self.out_shape

def testResults(inception, adv_layer, image_set):
    start = int(math.floor(IMAGE_SIZE - CENTER_SIZE) / 2)
    end = int(math.ceil(IMAGE_SIZE - CENTER_SIZE) / 2 + CENTER_SIZE)
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

class DataPreprocessing():

    def __init__(self, imgID, size, labels, numberOfImages):
        self.imgID = imgID
        self.images = dict()
        self.size = size
        self.labels = labels
        self.numberOfImages = numberOfImages
        self.input_list = []
        self.output_list = []
        self.createData()

    def createData(self):
        self.loadImages()
        return self.makeLists()

    def loadImages(self):
        images = self.images
        for label in self.labels:
            if(self.imgID=="squares"):
                files = glob.glob("images/square_%dp/squares_%dp_%s_*.png" % (self.size, self.size, label))
                images[label] = [image.load_img(f, target_size=(self.size, self.size)) for f in files[:self.numberOfImages]]
        self.expandImages()

    def expandImages(self):
        images = self.images

        for key in images.keys():
            ls = images[key]
            if len(ls) < self.numberOfImages:

                images[key] = ls*math.ceil(self.numberOfImages/len(ls))

    def makeLists(self):
        images = self.images
        for key in images.keys():
            for value in images[key][:MAX_IMAGES_PER_CLASS]:
                new_value = np.asarray(value, dtype=np.float32)
                new_value = tf.convert_to_tensor(new_value, dtype=tf.float32)
                new_value /= 255.
                new_value -= 0.5
                new_value *= 2.0
                self.input_list.append(new_value)
                self.output_list.append(key)
        return self.input_list, self.output_list





class AdvModel():

    def __init__(self, epochs, batch_size, center_size, image_size, adam_learn_rate, adam_decay, step, model_name):
        self.step = step
        self.epochs = epochs
        self.batch_size = batch_size
        self.center_size = center_size
        self.image_size = image_size
        self.adam_learn_rate = adam_learn_rate
        self.adam_decay = adam_decay
        self.optimizer = Adam(lr = adam_learn_rate)
        self.image_model = self.make_image_model(model_name);
        self.adam_learn_rate = adam_learn_rate
        self.adam_decay = adam_decay
        self.build_model()

    def make_image_model(self, model_name):
        inception = [];
        if(model_name=="inception_v3"):
            inception = inception_v3.InceptionV3(weights='imagenet', input_tensor=Input(shape=(self.image_size, self.image_size, 3)))
            inception.trainable = False
        return inception

    def build_model(self):

        # Adv Layer
        inputs = Input(shape=(self.center_size, self.center_size, 3))
        al = AdvLayer(image_size=self.image_size, center_size=self.center_size)(inputs)
        #al.set_image_size(self.image_size)
        # Combine layers
        outputs = self.image_model(al)

        model = Model(inputs=[inputs],
                      outputs=[outputs])
        self.model = model
        lr_metric = self.get_lr_metric(self.optimizer)
        model.compile(optimizer=self.optimizer,
                      loss=self._loss_tensor,
                      metrics=[self._accuracy, lr_metric])

    def fit_model(self, x_train, y_train):
        cbks = [keras.callbacks.LearningRateScheduler(schedule=lambda epoch: self.step_decay(epoch=epoch, lr=self.optimizer.lr), verbose=0),
                keras.callbacks.ModelCheckpoint(filepath = "./results/adv/weights.{epoch:02d}-{loss:.2f}.hdf5", verbose=0,
                                                save_best_only=False, save_weights_only=False, mode='auto', period=1, monitor="loss")]
        history = self.model.fit(x=x_train, y=y_train,
                            epochs=self.epochs,
                            batch_size=self.batch_size, callbacks=cbks)
        return history

    # https://stackoverflow.com/questions/52277003/how-to-implement-exponentially-decay-learning-rate-in-keras-by-following-the-glo
    def step_decay(self, epoch, lr):
        # initial_lrate = 1.0 # no longer needed
        drop = self.adam_decay
        #epochs_drop = 2.0
        lrate = float(lr * math.pow(drop,  epoch//self.step))
        return lrate

    def get_model(self):
        return self.model

    def _loss_tensor(self, y_true, y_pred):
        disLogits = tf.matmul(y_pred, label_mapping())
        cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_true[:, :len(LABELS)], logits=disLogits))
        reg_loss = 2e-6 * tf.nn.l2_loss(self.model.get_layer('adv_layer_1').adv_weights)
        loss = cross_entropy_loss + reg_loss
        print(loss)
        return loss

    def _accuracy(self, y_true, y_pred):
        correct_predictions = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        return accuracy

    def get_lr_metric(self, optimizer):
        def lr(y_true, y_pred):
            return optimizer.lr

        return lr

if __name__ == "__main__":
    ### SETUP PARAMETERS ###
    CENTER_SIZE = 35
    IMAGE_SIZE = 299
    LABELS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    MAX_IMAGES_PER_CLASS = 112#1111

    ADAM_LEARN_RATE = 0.05
    ADAM_DECAY = 0.96
    DECAY_STEP = 2
    TEST_SIZE= 0.10
    EPOCHS = 10000
    BATCH_SIZE = 50 
    ### END SETUP PARAMETERS ###

    # Load / prepare data
    data = DataPreprocessing(imgID="squares", size=CENTER_SIZE, labels=LABELS, numberOfImages=MAX_IMAGES_PER_CLASS)
    input_list, output_list = data.createData()
    #while(totalNr<100):
    #     input_list.append(np.asarray(value))
    #     output_list.append(array)
    #     totalNr+=1
    #input_a = np.array(input_list)
    input_a = np.array(tf.convert_to_tensor(input_list))
    print("Creating output")
    output_a = np.array(output_list, dtype=np.float32)
    output_a = to_categorical(output_a, num_classes=1000, dtype='float32')
    print("Creating train and validation")
    x_train, x_valid, y_train, y_valid = train_test_split(input_a, output_a, test_size=TEST_SIZE, shuffle= True)
    print("Train and validation compiled")


    #Hardcoded for now
##    predDict = {"1":[],"2":[],"3":[],"4":[],"5":[],"6":[],"7":[],"8":[],"9":[]}
##    for i in range(0, len(input_list)):
##        #pred_val = input_list[i].reshape(1,CENTER_SIZE,CENTER_SIZE,3)
##        predDict[output_list[i]].append(np.argmax(model.predict(input_list[i])))
##    for i in predDict.keys():
##        print(predDict[i], max(set(predDict[i]), key = predDict[i].count))
        
    # Write results
    print("Compiling model")
    a_model = AdvModel(epochs=EPOCHS, model_name="inception_v3", batch_size=BATCH_SIZE, center_size=CENTER_SIZE, image_size=IMAGE_SIZE,
                       adam_learn_rate=ADAM_LEARN_RATE, adam_decay=ADAM_DECAY, step=DECAY_STEP)
    print("fit model")
    a_model.fit_model(x_train, y_train)
    model = a_model.get_model()
    image_model = a_model.image_model

    adv_layer_weights = model.get_layer(index=1).get_weights() # return numpy array containing 299 elements of size 299x3

    results = testResults(image_model, adv_layer_weights[0], list(zip(x_valid, y_valid)))
    ProbabilityList = []
    for i in results.keys():
        print(str(i) + "\n")
        allClasses = []
        classCount = []
        for j in results[i]:
            allClasses.append(j[0][1])
        for j in set(allClasses):
            classCount.append((j, allClasses.count(j)/len(allClasses)))
            classCount.sort(key=lambda x: x[1], reverse=True)
        ProbabilityList.append((i, classCount))
    ProbabilityList.sort(key=lambda x: x[0])
    for i,j in ProbabilityList:
        print(i)
        print(j)
    # Save weights .json
    adv_layer = {"weights": adv_layer_weights[0].tolist()}


    if not os.path.exists("results/adv/"):
            os.makedirs("results/adv/")

    now = datetime.now()
    now_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    with open("results/adv/adv_layer-%s.json" % now_string, 'w') as outfile:
        json.dump(adv_layer, outfile)
    
    
