import glob
import math

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from scipy import ndimage

def makeLists(images):
    x = list()
    y = list()
    for k, v in images.items():
        x.extend(v)
        y.extend([k]*len(v))
    return x,y;

def preprocessImage(f):
    f = ndimage.median_filter(np.repeat(np.array(image.load_img(f, target_size=(35, 35), color_mode='grayscale'))[:, :, np.newaxis], 3, axis=2), 5)
    median = np.percentile(f,17)
    idx = f[:,:,:] > median
    f[idx]=255
    return f
    
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

        x,y = makeLists(images)
    elif img_id == 'squares_6_6':
        if not labels:
            labels = [i for i in range(1, 26)]

        images = dict()
        # Load images in dict
        for label in labels:
            files = glob.glob("images/square_6_31p/squares_6_31p_%s_*.png" % label)
            images[label] = [np.array(image.load_img(f, target_size=(31, 31))) for f in
                             files[:number_of_images]]

        # Expand to same number of images per class by copying.
        if(expand):
            for key in images.keys():
                ls = images[key]
                if len(ls) < number_of_images:
                    images[key] = ls * math.ceil(number_of_images / len(ls))

        x,y = makeLists(images)

    elif 'captcha' in img_id :
        images = dict()
        if img_id=='captcha_seperator':
            if not labels:
                labels = [i for i in range(0,5)]
            lowercaseamfiles = glob.glob("images/captcha_lowercase_60/lowercase_[a-m]_*.png")
            images[0] = [np.array(image.load_img(f, target_size=(35, 35))) for f in
                                 lowercaseamfiles]
            lowercasenzfiles = glob.glob("images/captcha_lowercase_60/lowercase_[n-z]_*.png")
            images[1] = [np.array(image.load_img(f, target_size=(35, 35))) for f in
                                 lowercasenzfiles]
            uppercaseamfiles = glob.glob("images/captcha_uppercase_60/uppercase_[A-M]_*.png")
            images[2] = [np.array(image.load_img(f, target_size=(35, 35))) for f in
                                 uppercaseamfiles]
            uppercasenzfiles = glob.glob("images/captcha_uppercase_60/uppercase_[N-Z]_*.png")
            images[3] = [np.array(image.load_img(f, target_size=(35, 35))) for f in
                                 uppercasenzfiles]
            digitfiles = glob.glob("images/captcha_digits_60/digits_[0-9]_*.png")
            images[4] = [np.array(image.load_img(f, target_size=(35, 35))) for f in
                                 digitfiles]
        else:
            if 'lowercase' in img_id:
                labelNames = list("abcdefghijklmnopqrstuvwxyz")
            elif 'uppercase' in img_id:
                labelNames = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            elif 'digits' in img_id:
                labelNames = list("0123456789")
            if not labels:
                labels = [i for i in range(0, len(labelNames))]
            for label in labels:
                files = glob.glob("images/{}_60/{}_{}_*.png".format(img_id,img_id.split("_")[1],labelNames[label]))
                images[label] = [preprocessImage(f) for f in
                                 files[:number_of_images]]
        #x,y = makeLists(images)

        x,y = makeLists(images)
    x = np.asarray(x)
    y = np.asarray(y)
    weightTensor = np.arange(len(labels))
    print(len(labels))
    for i in range(0,len(labels)):
        weightTensor[i]=np.count_nonzero(y==labels[i])

    weightTensor=weightTensor.astype('float32')
    weightTensor/=np.max(weightTensor)
    weightTensor=1/weightTensor
    weightTensor = dict(zip(labels, weightTensor))
    x = x.astype('float32')
    x /= 255
    # TODO ??
    # x -= 0.5
    # x *= 2
    return x, y, weightTensor
