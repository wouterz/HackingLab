import numpy as np
import math
import os
import glob
import sys

import tensorflow as tf
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.applications import inception_v3, inception_resnet_v2, resnet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split


class AdvLayer(Layer):

    def __init__(self, image_size=0, center_size=0, **kwargs):
        self.image_size = image_size
        self.center_size = center_size
        super(AdvLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'image_size': self.image_size,
            'center_size': self.center_size,
        })
        return config

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        img_shape = (self.image_size, self.image_size, 3)
        self.adv_weights = self.add_weight(name='kernel',
                                           shape=img_shape,
                                           initializer='random_normal',
                                           # regularizer = keras.regularizers.l2(l=0.01),
                                           trainable=True)
        super(AdvLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        start = math.ceil((self.image_size - self.center_size) / 2)
        start2 = math.ceil((self.image_size - self.center_size) / 2)
        if not self.center_size % 2:
            start2 = int(math.floor((self.image_size - self.center_size) / 2))
        input_mask = np.pad(np.zeros([1, self.center_size, self.center_size, 3]),
                            [[0, 0], [start, start2], [start, start2], [0, 0]], 'constant', constant_values=1)
        mask = tf.constant(input_mask, dtype=tf.float32)
        padx = tf.pad(x,
                      paddings=tf.constant([[0, 0], [start, start2], [start, start2], [0, 0]]))
        adv_img = tf.nn.tanh(tf.multiply(self.adv_weights, mask)) + padx
        # adv_img[start:end, start:end, :] = x
        self.out_shape = adv_img.shape

        return adv_img

    def compute_output_shape(self, input_shape):
        return self.out_shape


def testResults(inception, IMAGE_SIZE, CENTER_SIZE, adv_layer, image_set):
    start = int(math.floor(IMAGE_SIZE - CENTER_SIZE) / 2)
    end = int(math.ceil(IMAGE_SIZE - CENTER_SIZE) / 2 + CENTER_SIZE)
    predictionDict = {}
    for (i, j) in image_set:
        key = np.argwhere(j == 1)[0][0]
        adv_layer[start:end, start:end, :] = i
        adv_layer2 = adv_layer.reshape(1, 299, 299, 3)
        prediction = inception.predict(adv_layer2)
        if key not in predictionDict.keys():
            predictionDict[key] = []

        predictionDict[key].append(inception_v3.decode_predictions(prediction, top=5, utils=tf.keras.utils)[0])
    return predictionDict


def label_mapping():
    imagenet_label = np.zeros([1000, len(LABELS)])
    imagenet_label[0:len(LABELS), 0:len(LABELS)] = np.eye(len(LABELS))
    return tf.constant(imagenet_label, dtype=tf.float32)


class DataPreprocessing:

    def __init__(self, imgID, size, labels, numberOfImages):
        self.imgID = imgID
        self.images = dict()
        self.size = size
        self.labels = labels
        self.numberOfImages = numberOfImages
        self.input_list = []
        self.output_list = []

    def createData(self):
        self.loadImages()
        return self.makeLists()

    def loadImages(self):
        images = self.images
        for label in self.labels:
            if (self.imgID == "squares"):
                files = glob.glob("images/square_%dp/squares_%dp_%s_*.png" % (self.size, self.size, label))
                images[label] = [image.load_img(f, target_size=(self.size, self.size)) for f in
                                 files[:self.numberOfImages]]
        self.expandImages()

    def expandImages(self):
        images = self.images

        for key in images.keys():
            ls = images[key]
            if len(ls) < self.numberOfImages:
                images[key] = ls * math.ceil(self.numberOfImages / len(ls))

    def makeLists(self):
        images = self.images
        for key in images.keys():
            for value in images[key][:self.numberOfImages]:
                new_value = np.asarray(value, dtype=np.float32)
                new_value = tf.convert_to_tensor(new_value, dtype=tf.float32)
                new_value /= 255.
                new_value -= 0.5
                new_value *= 2.0
                self.input_list.append(new_value)
                self.output_list.append(key)
        return self.input_list, self.output_list


class AdvModel:

    def __init__(self, epochs, batch_size, center_size, image_size, adam_learn_rate, adam_decay, step, model_name,
                 labels, save_path):
        self.labels = labels
        self.previousEpoch = 0
        self.step = step
        self.epochs = epochs
        self.batch_size = batch_size
        self.center_size = center_size
        self.image_size = image_size
        self.adam_learn_rate = adam_learn_rate
        self.adam_decay = adam_decay
        self.optimizer = Adam(lr=adam_learn_rate)
        self.image_model = self.make_image_model(model_name)
        self.build_model()
        self.save_path = save_path

    def make_image_model(self, model_name):
        inception = []
        if (model_name == "inception_v3"):
            inception = inception_v3.InceptionV3(weights='imagenet',
                                                 input_tensor=Input(shape=(self.image_size, self.image_size, 3)))
            inception.trainable = False
        return inception

    def build_model(self):
        # Adv Layer
        inputs = Input(shape=(self.center_size, self.center_size, 3))
        al = AdvLayer(image_size=self.image_size, center_size=self.center_size)(inputs)
        # al.set_image_size(self.image_size)
        # Combine layers
        outputs = self.image_model(al)

        model = Model(inputs=[inputs],
                      outputs=[outputs])
        self.model = model
        lr_metric = self.get_lr_metric(self.optimizer)
        model.compile(optimizer=self.optimizer,
                      loss='categorical_crossentropy',
                      metrics=[self._accuracy, lr_metric])

    def fit_model(self, x_train, y_train, x_valid, y_valid, save_path, currentEpoch=0):
        savepath = "%sweights.{epoch:02d}-{loss:.2f}.hdf5" % save_path
        cbks = [
            tf.keras.callbacks.LearningRateScheduler(schedule=lambda epoch: self.step_decay(epoch=epoch), verbose=1),
            tf.keras.callbacks.ModelCheckpoint(filepath=savepath,
                                               verbose=0,
                                               save_best_only=True, save_weights_only=False, mode='auto', period=25,
                                               monitor="loss")]

        history = self.model.fit(x=x_train, y=y_train,
                                 epochs=self.epochs - currentEpoch,
                                 batch_size=self.batch_size, callbacks=cbks)
        return history

    def continue_model(self, currentEpoch, weights, x_train, y_train, x_valid, y_valid, save_path):
        self.model.load_weights(weights)
        self.previousEpoch = currentEpoch
        self.fit_model(x_train, y_train, x_valid, y_valid, save_path, currentEpoch)

    # https://stackoverflow.com/questions/52277003/how-to-implement-exponentially-decay-learning-rate-in-keras-by-following-the-glo
    def step_decay(self, epoch):
        lr = self.adam_learn_rate
        drop = self.adam_decay
        lrate = float(lr * math.pow(drop, (epoch + self.previousEpoch) // self.step))
        return lrate

    def get_model(self):
        return self.model

    def _loss_tensor(self, y_true, y_pred):
        # cross_entropy_loss = tf.reduce_mean(
        #    tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
        cross_entropy_loss = tf.reduce_mean(K.categorical_crossentropy(K.softmax(y_true), y_pred))
        reg_loss = 2e-6 * tf.nn.l2_loss(self.model.get_layer('adv_layer_1').adv_weights)
        loss = cross_entropy_loss + reg_loss
        return loss

    def _accuracy(self, y_true, y_pred):
        correct_predictions = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_pred, 1))
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
    LABELS = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    MAX_IMAGES_PER_CLASS = 2222

    ADAM_LEARN_RATE = 0.05
    ADAM_DECAY = 0.96

    DECAY_STEP = 2
    TEST_SIZE = 0.10
    EPOCHS = 10000
    BATCH_SIZE = 50

    SAVE_PATH = sys.argv[1]

    if len(sys.argv) == 4:
        CONTINUE_MODEL = sys.argv[2]
        CONTINUE_MODEL_EPOCHS = int(sys.argv[3])

    ### END SETUP PARAMETERS ###
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # Load / prepare data
    data = DataPreprocessing(imgID="squares", size=CENTER_SIZE, labels=LABELS, numberOfImages=MAX_IMAGES_PER_CLASS)
    input_list, output_list = data.createData()

    input_a = np.array(tf.convert_to_tensor(input_list))
    print("Creating output")
    output_a = np.array(output_list, dtype=np.float32)
    output_a = to_categorical(output_a, num_classes=1000, dtype='float32')

    print("Creating train and validation")
    x_train, x_valid, y_train, y_valid = train_test_split(input_a, output_a, test_size=TEST_SIZE, shuffle=True)
    print("Train and validation compiled")

    # Write results
    print("Compiling model")
    a_model = AdvModel(epochs=EPOCHS, model_name="inception_v3", batch_size=BATCH_SIZE, center_size=CENTER_SIZE,
                       image_size=IMAGE_SIZE,
                       adam_learn_rate=ADAM_LEARN_RATE, adam_decay=ADAM_DECAY, step=DECAY_STEP, labels=LABELS,
                       save_path=SAVE_PATH)
    print("fit model")

    if len(sys.argv) != 3:
        print("Fit new model")
        a_model.fit_model(x_train, y_train, x_valid, y_valid, SAVE_PATH)
    else:
        print("Continue model %s, from epoch %d" % (CONTINUE_MODEL, CONTINUE_MODEL_EPOCHS))
        a_model.continue_model(CONTINUE_MODEL_EPOCHS, CONTINUE_MODEL, x_train, y_train, x_valid, y_valid, SAVE_PATH)

    model = a_model.get_model()
    image_model = a_model.image_model

    adv_layer_weights = model.get_layer(
        index=1).get_weights()  # return numpy array containing 299 elements of size 299x3

    ##    results = testResults(image_model, IMAGE_SIZE, CENTER_SIZE, adv_layer_weights[0], list(zip(input_a, output_a)))
    ##    ProbabilityList = []
    ##    for i in results.keys():
    ##        print(str(i) + "\n")
    ##        allClasses = []
    ##        classCount = []
    ##        for j in results[i]:
    ##            allClasses.append(j[0][1])
    ##        for j in set(allClasses):
    ##            classCount.append((j, allClasses.count(j)/len(allClasses)))
    ##            classCount.sort(key=lambda x: x[1], reverse=True)
    ##        ProbabilityList.append((i, classCount))
    ##    ProbabilityList.sort(key=lambda x: x[0])
    ##    for i,j in ProbabilityList:
    ##        print(i)
    ##        print(j)
    ##    # Save weights .json
    ##    adv_layer = {"weights": adv_layer_weights[0].tolist()}

    ##    if not os.path.exists("results/adv/"):
    ##            os.makedirs("results/adv/")
    ##
    ##    now = datetime.now()
    ##    now_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    ##    with open("results/adv/adv_layer-%s.json" % now_string, 'w') as outfile:
    ##        json.dump(adv_layer, outfile)

    fun = K.function([model.layers[0].input, K.learning_phase()], [model.output])
    DictionaryList = []
    for i in range(0, len(LABELS)):
        DictionaryList.append(dict.fromkeys(np.arange(1000), 0))
    for i in range(0, 1):
        for i in range(0, len(LABELS)):
            DictionaryList[i] = dict.fromkeys(np.arange(1000), 0)

        a_model.fit_model(x_train, y_train, x_valid, y_valid)
        newOutput = []
        for i in range(0, len(LABELS)):
            print("Set " + str(i))
            for j in range(i * MAX_IMAGES_PER_CLASS, (i + 1) * MAX_IMAGES_PER_CLASS):
                output = np.array(fun([input_a[j].reshape(1, 35, 35, 3), 0]))
                newOutput.append(output)
                DictionaryList[i][np.argmax(output)] += 1

        # a_model.fit_model(x_train, y_train, x_valid, y_valid)
        newOutput = np.array(newOutput).reshape(200, 1000)
        for i in range(0, 200):
            print(str(np.argmax(output_a[i])) + " <> " + str(np.argsort(newOutput[i])[-2:]))
        for i in DictionaryList:
            print(dict(filter(lambda elem: elem[1] != 0, i.items())))
        a_model.previousEpoch += EPOCHS
