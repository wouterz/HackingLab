import argparse
import os

from sklearn.model_selection import train_test_split
import numpy as np
from AdvModel import AdvModel
from data_helper import get_data
from tensorflow.keras.utils import to_categorical

if __name__ == "__main__":
    ### SETUP PARAMETERS ###
    parser = argparse.ArgumentParser()
    parser.add_argument('save_path', help='Path where to save files, END WITH /')
    parser.add_argument('images_per_class', type=int)
    parser.add_argument('image_type', choices=['mnist', 'squares', 'captcha'], help='mnist or squares')
    parser.add_argument('--continue_path')
    parser.add_argument('--continue_start_epoch', type=int)
    parser.add_argument('--batch_size', type=int, default=50)
    args = parser.parse_args()

    dim_map = {
        'captcha': (35, 3),
        'mnist': (28, 3),
        'squares': (35, 3),
        'cifar10': (32, 3),
        'cifar100': (32, 3),
    }
    IMAGES = args.image_type
    CENTER_SIZE, CHANNELS = dim_map[IMAGES]
    MAX_IMAGES_PER_CLASS = args.images_per_class

    BATCH_SIZE = args.batch_size
    SAVE_PATH = args.save_path
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    CONTINUE_MODEL = args.continue_path
    CONTINUE_MODEL_EPOCHS = args.continue_start_epoch

    # Fixed
    IMAGE_SIZE = 299
    ADAM_LEARN_RATE = 0.05
    ADAM_DECAY = 0.96
    DECAY_STEP = 2
    TEST_SIZE = 0.10
    EPOCHS = 10000

    print("PARAMETERS: imageType {}, imagesPerClass {} SavePath {}".format(IMAGES, MAX_IMAGES_PER_CLASS, SAVE_PATH))

    ### END SETUP PARAMETERS ###

    # Load / prepare data
    x_train, y_train, weights = get_data(IMAGES, MAX_IMAGES_PER_CLASS, train=True, expand=False)
    y_train = to_categorical(y_train, 1000,  dtype='float32')
    print(x_train.shape, y_train.shape)

    print("Creating train and validation")
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=TEST_SIZE, shuffle=True)
    print(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)

    print("Train and validation compiled")

    # Write results
    print("Compiling model")
    a_model = AdvModel(epochs=EPOCHS, model_name="inception_v3", batch_size=BATCH_SIZE, center_size=CENTER_SIZE,
                       image_size=IMAGE_SIZE, channels=CHANNELS,
                       adam_learn_rate=ADAM_LEARN_RATE, adam_decay=ADAM_DECAY, step=DECAY_STEP)

    print(a_model.get_model().summary())

    print("fit model")
    if not CONTINUE_MODEL_EPOCHS and not CONTINUE_MODEL:
        print("Fit new model")
        a_model.fit_model(x_train, y_train, x_valid, y_valid, weights, SAVE_PATH)
    else:
        print("Continue model %s, from epoch %d" % (CONTINUE_MODEL, CONTINUE_MODEL_EPOCHS))
        a_model.continue_model(CONTINUE_MODEL_EPOCHS, CONTINUE_MODEL, x_train, y_train, x_valid, y_valid, weights, SAVE_PATH)
