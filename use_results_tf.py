import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

import adv_model_script_tf as ams

CENTER_SIZE = 35
IMAGE_SIZE = 299
LABELS = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
TEXT_IMAGES_PER_CLASS = 10

filename = sys.argv[1]
if not os.path.isfile(filename):
    print('file %s not found' % filename)
    exit(1)

dp = ams.DataPreprocessing("squares", CENTER_SIZE, LABELS, TEXT_IMAGES_PER_CLASS)
input_list, output_list = dp.createData()
output_list = to_categorical(output_list, num_classes=1000, dtype='float32')

a_model = ams.AdvModel(epochs=100, model_name="inception_v3", batch_size=50, center_size=CENTER_SIZE,
                       image_size=IMAGE_SIZE,
                       adam_learn_rate=0.05, adam_decay=0.95, step=2, labels=LABELS)
model = a_model.get_model()
model.load_weights(filename)
print(model.summary())

K.set_learning_phase(0)
fun = K.function([model.layers[0].input], [model.output])
DictionaryList = []
for i in range(0, len(LABELS)):
    DictionaryList.append(dict.fromkeys(np.arange(1000), 0))
for i in range(0, len(LABELS)):
    DictionaryList[i] = dict.fromkeys(np.arange(1000), 0)

newOutput = []
for i in range(0, len(LABELS)):
    print("Set " + str(i))
    for j in range(i * TEXT_IMAGES_PER_CLASS, (i + 1) * TEXT_IMAGES_PER_CLASS):
        output = np.array(
            fun([np.array(tf.convert_to_tensor(
                input_list[j]
            )
            ).reshape(1, 35, 35, 3), 1])
        )
        newOutput.append(output)
        DictionaryList[i][np.argmax(output)] += 1

newOutput = np.array(newOutput).reshape(len(LABELS) * TEXT_IMAGES_PER_CLASS, 1000)
# for i in range(0, len(LABELS) * MAX_IMAGES_PER_CLASS):
#     print(str(np.argmax(output_list[i])) + " <> " + str(np.argsort(newOutput[i])[-2:]))
for i in DictionaryList:
    print(dict(filter(lambda elem: elem[1] != 0, i.items())))
