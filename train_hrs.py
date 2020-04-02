import os
import tensorflow as tf
import keras.backend as K
import argparse

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.optimizers import SGD
from tensorflow_core.python.keras.utils.np_utils import to_categorical

from data_helper import get_data
from keras_utils import construct_switching_block
from block_split_config import get_split


def train_hrs(model_indicator, training_epoch, split='default', dataset='IMAGENET'):
    # get block definitions
    blocks_definition = get_split(split, dataset)
    print(blocks_definition[1]())

    # parse structure
    structure = [int(ss[:-1]) for ss in model_indicator.split('[')[1:]]
    nb_block = len(structure)

    # make sure model_indicator, training_epoch and split all have the same number of blocks
    assert nb_block == len(training_epoch) == len(blocks_definition), "The number of blocks indicated by " \
                                                                      "model_indicator, training_epoch and split must " \
                                                                      "be the same!"
    # create weights save dir
    save_dir = './Model/%s_models/' % dataset + model_indicator + '/'
    try:
        os.makedirs('./Model/%s_models/' % dataset + model_indicator + '/')
    except: pass

    IMAGES = 20
    IMAGE_SIZE = 299
    TEST_SIZE = 0.2
    x, y, weights = get_data('imagenet', IMAGES, target_size=IMAGE_SIZE, labels=[2,3,4])
    y = to_categorical(y, 1000, dtype='float32')
    print(x.shape, y.shape)

    print("Creating train and validation")
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=TEST_SIZE, shuffle=True)
    print(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)

    # loss definition
    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct, logits=predicted)

    '''
    Training HRS
    '''
    # start the training process
    for block_idx in range(nb_block):
        print("start training the %d\'s block" % block_idx)
        print(nb_block)
        # construct the trained part:
        # switching blocks up to the (block_idx - 1)'s block
        if block_idx == 0:
            print("BLOCK 0")
            model_input = InputLayer(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
            # note: for InputLayer the input and output tensors are the same one.
            trained_blocks_output = model_input.output
        else:
            print("block > 0")
            model_input = InputLayer(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

            # build switching blocks
            block_input = model_input.output
            for i in range(block_idx):
                weight_dir = save_dir + '%d_' % i + '%d'
                block_output = construct_switching_block(block_input, structure[i], blocks_definition[i], weight_dir)
                block_input = block_output
            trained_blocks_output = block_output

        # construct the part to train
        # normal blocks (with only one channel) from block_idx to the end
        for channel_idx in range(structure[block_idx]):
            print("channel: " + str(channel_idx))
            block_input = trained_blocks_output
            # the channel to train
            channel_to_train = blocks_definition[block_idx]()
            block_output = channel_to_train(block_input)
            block_input = block_output
            # add following blocks in any
            for j in range(block_idx+1, nb_block):
                print("Block: "+ str(j))
                print(nb_block)
                channel = blocks_definition[j]()
                block_output = channel(block_input)
                block_input = block_output

            # construct the model object
            model = Model(inputs=model_input.input, outputs=block_output)
            # optimizer
            sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            # training
            model.compile(loss=fn, optimizer=sgd, metrics=['accuracy'])
            model.fit(x_train, y_train, batch_size=128, validation_data=(x_valid, y_valid),
                      nb_epoch=training_epoch[block_idx], shuffle=True)

            # save weights of this channel
            channel_to_train.save_weights(save_dir + '%d_%d' % (block_idx, channel_idx))

        # after training all channels in this block, reset tf graph
        K.clear_session()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_indicator', default='test_hrs[10][10]', help='model indicator, format: model_name[5][5] for'
                                                                            'a HRS model with 5 by 5 channels')
    parser.add_argument('--split', default='default', help='the indicator of channel structures in each block')
    parser.add_argument('--train_schedule', default=[40, 40], help='number of epochs for training each block', type=int,
                        nargs='*')
    parser.add_argument('--dataset', default='IMAGENET', help='IMAGENET, CIFAR or MNIST')

    args = parser.parse_args()
    train_hrs(model_indicator=args.model_indicator,
              training_epoch=args.train_schedule,
              dataset=args.dataset,
              split=args.split)
    pass







