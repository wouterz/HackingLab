import argparse
import os
import keras
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import ZeroPadding2D, LocallyConnected2D, Activation
from tensorflow.keras.optimizers  import SGD

from project_utils import get_data
from keras_utils import construct_hrs_model
from block_split_config import get_split


def defend_adversarial_reprogramming(model_indicator, split, epochs):
    save_dir = './Adversarial_Reprogramming/' + args.model_indicator + '/'
    try: os.makedirs(save_dir)
    except: pass

    # get MNIST data
    [X_train, X_test, Y_train, Y_test] = get_data(dataset='MNIST', scale1=True, one_hot=False, percentage=0.01)

    #TODO: Add HRS_ADVModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_indicator', default='test_hrs[10][10]', help='model indicator, format: model_name[5][5] for'
                                                                            'a HRS model with 5 by 5 channels')
    parser.add_argument('--split', default='default', help='the indicator of channel structures in each block')
    parser.add_argument('--epochs', default=50, help='the number of epochs to train (reprogram).')

    args = parser.parse_args()
    defend_adversarial_reprogramming(model_indicator=args.model_indicator,
                                     split=args.split,
                                     epochs=args.epochs)



