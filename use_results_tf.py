import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from data_helper import get_data

import adv_model_script_tf as ams

K.set_learning_phase(0)

parser = argparse.ArgumentParser()
parser.add_argument('save_path', help='Path where to saved file')
parser.add_argument('images_per_class', type=int)
parser.add_argument('image_type', choices=['mnist', 'squares'], help='mnist or squares')

args = parser.parse_args()

filename = args.save_path
if not os.path.isfile(filename):
    print('file %s not found' % filename)
    exit(1)

CHANNELS = 3
IMAGE_SIZE = 299
CENTER_SIZE = 35
LABELS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
IMAGES = args.image_type
TEST_IMAGES_PER_CLASS = args.images_per_class

if IMAGES == 'mnist':
    CENTER_SIZE = 28
    LABELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


x_test, y_test, weights = get_data(IMAGES, TEST_IMAGES_PER_CLASS, train=False, expand=False)
y_test = to_categorical(y_test, num_classes=1000, dtype='float32')

a_model = ams.AdvModel(epochs=100, model_name="inception_v3", batch_size=50, center_size=CENTER_SIZE,
                       image_size=IMAGE_SIZE,
                       adam_learn_rate=0.05, adam_decay=0.95, step=2,  channels=CHANNELS)

model = a_model.get_model()
model.load_weights(filename)

results = model.evaluate(x_test, y_test)
print(results)
