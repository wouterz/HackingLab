import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Layer


class AdvLayer(Layer):

    def __init__(self, image_size, center_size, channels, **kwargs):
        self.image_size = image_size
        self.center_size = center_size
        self.channels = channels
        super(AdvLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'image_size': self.image_size,
            'center_size': self.center_size,
            'channels': self.channels,
        })
        return config

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        img_shape = (self.image_size, self.image_size, self.channels)
        self.adv_weights = self.add_weight(name='kernel',
                                           shape=img_shape,
                                           initializer='random_normal',
                                           trainable=True)
        super(AdvLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, **kwargs):
        start = math.ceil((self.image_size - self.center_size) / 2)
        start2 = math.ceil((self.image_size - self.center_size) / 2)
        if not self.center_size % 2:
            start2 = int(math.floor((self.image_size - self.center_size) / 2))
        input_mask = np.pad(np.zeros([1, self.center_size, self.center_size, self.channels]),
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


class AdvModel:

    def __init__(self, epochs, batch_size, center_size, image_size, channels,
                 adam_learn_rate, adam_decay, step, model_name):
        self.previousEpoch = 0
        self.step = step
        self.epochs = epochs
        self.batch_size = batch_size
        self.center_size = center_size
        self.image_size = image_size
        self.channels = channels
        self.adam_learn_rate = adam_learn_rate
        self.adam_decay = adam_decay

        self.optimizer = Adam(lr=adam_learn_rate)

        self.image_model = self.make_image_model(model_name)
        self.build_model()

    def make_image_model(self, model_name):
        inception = []
        if (model_name == "inception_v3"):
            inception = inception_v3.InceptionV3(weights='imagenet', input_tensor=Input(
                shape=(self.image_size, self.image_size, 3)
            ), layers=tf.keras.layers)
            inception.trainable = False
        return inception

    def build_model(self):
        # Adv Layer
        inputs = Input(shape=(self.center_size, self.center_size, self.channels))
        al = AdvLayer(image_size=self.image_size, center_size=self.center_size, channels=self.channels)(inputs)


        # Combine layers
        outputs = self.image_model(al)

        model = Model(inputs=[inputs],
                      outputs=[outputs])
        self.model = model
        lr_metric = self.get_lr_metric(self.optimizer)

        model.compile(optimizer=self.optimizer,
                      loss=['categorical_crossentropy'],
                      metrics=['accuracy', lr_metric])

    def fit_model(self, x_train, y_train, x_valid, y_valid, weights, save_path=""):
        savepath = "%sweights.{epoch:02d}-{loss:.2f}.hdf5" % save_path

        cbks = [
            tf.keras.callbacks.LearningRateScheduler(schedule=lambda epoch: self.step_decay(epoch=epoch), verbose=1),
            tf.keras.callbacks.ModelCheckpoint(filepath=savepath, verbose=0,
                                               save_best_only=True, save_weights_only=False, mode='auto', period=5)]

        history = self.model.fit(x=x_train, y=y_train, validation_data=(x_valid, y_valid),
                                 epochs=self.epochs,
                                 initial_epoch = self.previousEpoch,
                                 batch_size=self.batch_size, callbacks=cbks, class_weight=weights, shuffle=True)
        return history

    def continue_model(self, current_epoch, model_weights, x_train, y_train, x_valid, y_valid, class_weights, save_path="", retrain_inception=False):
        self.model.load_weights(model_weights)

        if retrain_inception:
            print("set inception retrainable")
            self.model.get_layer('input_2').trainable = False
            self.model.get_layer('adv_layer').trainable = False
            self.model.get_layer('inception_v3').trainable = True
            for l in self.model.layers:
                print(l.name, l.trainable)
            self.model.compile(optimizer=self.optimizer,
                          loss=['categorical_crossentropy'],
                          metrics=['accuracy', self.get_lr_metric(self.optimizer)])
            print(self.model.summary())


        self.previousEpoch = current_epoch
        self.fit_model(x_train, y_train, x_valid, y_valid, class_weights, save_path)

    def step_decay(self, epoch):
        lr = self.adam_learn_rate
        drop = self.adam_decay
        lrate = float(lr * math.pow(drop, epoch // self.step))
        return lrate

    def get_model(self):
        return self.model

    def _loss_tensor(self, y_true, y_pred):
        cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        reg_loss = 2e-6 * tf.nn.l2_loss(self.model.get_layer('adv_layer').adv_weights)
        loss = cross_entropy_loss + reg_loss
        return loss

    # def accuracy(self, y_true, y_pred):
    #     correct_predictions = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_pred, 1))
    #     accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    #     return accuracy

    def get_lr_metric(self, optimizer):
        def lr(y_true, y_pred):
            return optimizer.lr

        return lr

