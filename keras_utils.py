import os
import tensorflow as tf
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.layers import Input, InputLayer, Layer
from tensorflow.keras.models import Model
from collections.abc import Iterable


class RandomMask(Layer):
    def __init__(self, nb_channels, **kwargs):
        self.nb_channels = nb_channels
        self.mask = None
        super().__init__(**kwargs)

    def build(self, input_shape):
        # the input_shape here should be a list
        assert isinstance(input_shape, list), "the input shape of RandomMask layer should be a list"
        shape = input_shape[0]

        # build the random mask
        if self.nb_channels == 1:
            self.mask = tf.ones((1,) + shape[1:])
        else:
            ones = tf.ones((1,) + shape[1:])
            zeros = tf.zeros((self.nb_channels - 1,) + shape[1:])

            mask = tf.concat([ones, zeros], 0)
            self.mask = tf.random_shuffle(mask)
            pass

    def call(self, x):
        # x should be a list
        assert isinstance(x, list), "the input of RandomMask layer should be a list"
        xs_expand = [tf.expand_dims(x_orig, axis=1) for x_orig in x]
        x = tf.concat(xs_expand, 1)
        x = x * self.mask
        x = tf.reduce_sum(x, axis=1)
        return x

    def compute_output_shape(self, input_shape):
        shape = input_shape[0]
        return shape


def construct_model_by_blocks(block_list):
    # connect all blocks
    if len(block_list) == 1:
        i = block_list[0].input
        o = block_list[0].output
    else:
        i = block_list[0].input
        o = block_list[0].output
        idx = 1
        while idx < len(block_list):
            o = block_list[idx](o)
            idx += 1
    model = Model(input=i, output=o)

    return model


def split_keras_model(model, starting_layer_name, end_layer_name):
    # this is the split point, i.e. the starting layer in our sub-model
    # create a new input layer for our sub-model we want to construct
    new_input = Input(batch_shape=model.get_layer(starting_layer_name).get_input_shape_at(0))

    layer_outputs = {}

    def get_output_of_layer(layer):

        # if we have already applied this layer on its input(s) tensors,
        # just return its already computed output
        if layer.name in layer_outputs:
            return layer_outputs[layer.name]

        # if this is the starting layer, then apply it on the input tensor
        if layer.name == starting_layer_name:
            out = layer(new_input)
            layer_outputs[layer.name] = out
            return out

        # find all the connected layers which this layer
        # consumes their output
        prev_layers = []
        for node in layer._inbound_nodes:
            if isinstance(node.inbound_layers, Iterable):
                prev_layers.extend(node.inbound_layers)
            else:
                prev_layers.append(node.inbound_layers)
        # get the output of connected layers
        pl_outs = []
        for pl in prev_layers:
            pl_outs.extend([get_output_of_layer(pl)])
        # apply this layer on the collected outputs
        out = layer(pl_outs[0] if len(pl_outs) == 1 else pl_outs)
        layer_outputs[layer.name] = out
        return out

    # note that we start from the last layer of our desired sub-model.
    # this layer could be any layer of the original model as long as it is
    # reachable from the starting layer
    new_output = get_output_of_layer(model.get_layer(end_layer_name))

    # create the sub-model
    model = Model(new_input, new_output)
    return model


def split_InceptionV3():
    base_model = inception_v3.InceptionV3(weights=None)
    block0 = split_keras_model(base_model, "input_1", "activation_4")
    block1 = split_keras_model(base_model, "max_pooling2d_1", "predictions")
    return block0, block1


def construct_switching_block(input, nb_channels, channel_definition, weights, freeze_channel=True):
    channel_output_list = []
    for channel_idx in range(nb_channels):
        channel = channel_definition()
        channel_output = channel(input)
        channel_output_list.append(channel_output)
        # load weights
        if weights:
            channel.load_weights(weights % channel_idx)
        if freeze_channel:
            for layer in channel.layers:
                layer.trainable = False

    # using a random mask to mask inactive channels
    print("nb_channel: " + str(nb_channels))
    block_output = RandomMask(nb_channels)(channel_output_list)
    return block_output


def construct_hrs_model(dataset, model_indicator, blocks_definition, load_weights=True):
    # get structure from model_indicator
    structure = [int(ss[:-1]) for ss in model_indicator.split('[')[1:]]  # [10][10]
    nb_block = len(structure)
    # assert nb blocks
    assert len(structure) == len(blocks_definition), 'arg structure and block_definition need to have the same length'

    # assert weights exist
    weights_dir = './Model/%s_models/%s' % (dataset, model_indicator)

    assert os.path.exists(weights_dir), '%s does not exist' % weights_dir

    # input
    # img_rows, img_cols, img_channels = get_dimensions(dataset)
    # TODO
    model_input = InputLayer(input_shape=(35, 35, 3))
    save_dir = './Model/%s_models/' % dataset + model_indicator + '/'

    # loop over block
    block_input = model_input.output
    for i in range(nb_block):
        weight_dir = save_dir + '%d_' % i + '%d'
        print(weights_dir)
        block_output = construct_switching_block(input=block_input, nb_channels=structure[i],
                                                 channel_definition=blocks_definition[i], weights=weight_dir)
        block_input = block_output

    # construct Model object
    model = Model(input=model_input.input, output=block_output)

    return model
