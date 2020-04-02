import glob
import math
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img


def get_data(img_id: str, number_of_images: int, train=True, labels=None, expand=True, **kwargs):
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
        if (expand):
            for key in images.keys():
                ls = images[key]
                if len(ls) < number_of_images:
                    images[key] = ls * math.ceil(number_of_images / len(ls))

        x = list()
        y = list()
        for k, v in images.items():
            x.extend(v)
            y.extend([k] * len(v))

    elif img_id == 'captcha':
        if not labels:
            labels = [i for i in range(0, 26)]

        labelNames = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                      't', 'u', 'v', 'w', 'x', 'y', 'z']
        images = dict()
        for label in labels:
            files = glob.glob("images/captcha_60/Letter_%s_*.png" % labelNames[label])
            images[label] = [np.array(image.load_img(f, target_size=(35, 35))) for f in
                             files[:number_of_images]]
        if (expand):
            for key in images.keys():
                ls = images[key]
                if len(ls) < number_of_images:
                    images[key] = ls * math.ceil(number_of_images / len(ls))

        x = list()
        y = list()
        for k, v in images.items():
            x.extend(v)
            y.extend([k] * len(v))

    elif img_id == 'imagenet':
        if not labels:
            labels = [2, 3, 4, 5, 48, 235, 256, 716, 730, 742]

        images = dict()
        for label in labels:
            files = glob.glob("images/imagenet/%s/*.jpg" % label)
            # files = glob.glob("images/captcha_60/Letter_%s_*.png" % labelNames[label])

            target_size = kwargs.get('target_size', 35)
            images[label] = [np.array(image.load_img(f, target_size=(target_size, target_size))) for f in
                             files[:number_of_images]]


        for k, v in images.items():
            x.extend(v)
            y.extend([k] * len(v))

    x = np.asarray(x)
    y = np.asarray(y)
    weightTensor = np.arange(len(labels))
    print(len(labels))
    for i in range(0, len(labels)):
        weightTensor[i] = np.count_nonzero(y == labels[i])

    weightTensor = weightTensor.astype('float32')
    weightTensor /= np.max(weightTensor)
    weightTensor = 1 / weightTensor
    weightTensor = dict(zip(labels, weightTensor))
    x = x.astype('float32')
    x /= 255
    # TODO ??
    # x -= 0.5
    # x *= 2
    return x, y, weightTensor


def load_imagenet_data(amount, target_size, labels=[2, 3, 4, 5, 48, 235, 256, 716, 730, 742]):
    x = list()
    y = list()

    for label in labels:

        urls_file = './images/imagenet/urls/label_{}_imagenet.synset.txt'.format(label)
        save_path = 'C:/Users/woute/Documents/Git/HackingLab/images/imagenet/{}'
        with open(urls_file, 'r') as f:
            failed = 0
            for i, url in enumerate(f):
                print(i, len(x))
                url = url.rstrip("\n")
                if i >= amount + failed:
                    break

                try:
                    if not os.path.exists(save_path.format(label)):
                        os.makedirs(save_path.format(label))

                    name = '{}/{}'.format(save_path.format(label), url.split("/")[-1])
                    img_path = tf.keras.utils.get_file(name, origin=url)
                    img = load_img(img_path, target_size=(target_size, target_size))
                    x.append(np.asarray(img))
                    y.append(label)
                except Exception as e:
                    print("exception", e)
                    failed = failed + 1

    return x, y


if __name__ == '__main__':
    x, y = load_imagenet_data(20, target_size=35)
