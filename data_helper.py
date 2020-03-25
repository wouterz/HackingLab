import glob
import math

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

def get_data(img_id: str, number_of_images: int, train=True, labels=None, expand=True):
    x = list()
    y = list()

    if img_id in ['mnist', 'cifar10', 'cifar100']:
        mapper = {
            'mnist': tf.keras.datasets.mnist.load_data,
            'cifar10': tf.keras.datasets.cifar10.load_data,
            'cifar100': tf.keras.datasets.cifar100.load_data,
        }
        (x_train, y_train), (x_test, y_test) = mapper[img_id]()

        if not labels:
            labels = list(set(y_train))

        for label in labels:
            if train:
                label_filter = np.where(y_train == label)
                x_filter, y_filter = x_train[label_filter][:number_of_images], y_train[label_filter][:number_of_images]
            else:
                label_filter = np.where(y_test == label)
                x_filter, y_filter = x_train[label_filter][:number_of_images], y_train[label_filter][:number_of_images]

            x.extend(x_filter)
            y.extend(y_filter)

        if img_id == 'mnist':
            x_3d = list()
            for i in x:
                dumb_3_channel = np.repeat(i[:, :, np.newaxis], 3, axis=2)
                x_3d.append(dumb_3_channel)

            x = x_3d

    elif img_id == 'squares':
        if not labels:
            labels = [i for i in range(1, 10)]

        images = dict()
        # Load images in dict
        for label in labels:
            files = glob.glob("images/square_35p/squares_35p_%s_*.png" % label)
            images[label] = [np.array(image.load_img(f, target_size=(35, 35))) for f in
                             files[:number_of_images]]

        # Expand to same number of images per class by copying.
        if(expand):
            for key in images.keys():
                ls = images[key]
                if len(ls) < number_of_images:
                    images[key] = ls * math.ceil(number_of_images / len(ls))

        x = list()
        y = list()
        for k, v in images.items():
            x.extend(v)
            y.extend([k]*len(v))
    elif img_id == 'captcha':
        if not labels:
            labels = [i for i in range(0, 26)]
        
        labelNames = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
        images = dict()
        for label in labels:
            files = glob.glob("images/captcha_60/Letter_%s_*.png" % labelNames[label])
            images[label] = [np.array(image.load_img(f, target_size=(35, 35))) for f in
                             files[:number_of_images]]
        if(expand):
            for key in images.keys():
                ls = images[key]
                if len(ls) < number_of_images:
                    images[key] = ls * math.ceil(number_of_images / len(ls))

        x = list()
        y = list()
        for k, v in images.items():
            x.extend(v)
            y.extend([k]*len(v))
            
    x = np.asarray(x)
    y = np.asarray(y)
    weightTensor = np.arange(len(labels))
    print(len(labels))
    for i in range(0,len(labels)):
        weightTensor[i]=np.count_nonzero(y==labels[i])

    weightTensor=weightTensor.astype('float32')
    weightTensor/=np.max(weightTensor)
    weightTensor=1/weightTensor

    x = x.astype('float32')
    x /= 255
    # TODO ??
    # x -= 0.5
    # x *= 2
    return x, y, weightTensor
