from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras_utils import split_InceptionV3


def get_split(indicator, dataset):
    if indicator == 'default':
        if dataset == 'CIFAR':
            # block definitions
            def block_0():
                channel = Sequential()
                channel.add(Conv2D(64, (3, 3), input_shape=(32, 32, 3)))
                channel.add(Activation('relu'))
                channel.add(Conv2D(64, (3, 3)))
                channel.add(Activation('relu'))
                channel.add(MaxPooling2D(pool_size=(2, 2)))
                channel.add(Conv2D(128, (3, 3)))
                channel.add(Activation('relu'))
                channel.add(Conv2D(128, (3, 3)))
                channel.add(Activation('relu'))
                channel.add(MaxPooling2D(pool_size=(2, 2)))
                channel.add(Flatten())
                channel.add(Dense(256))
                channel.add(Activation('relu'))
                channel.add(Dropout(0.7))

                return channel

            def block_1():
                channel = Sequential()
                channel.add(Dense(256, input_shape=(256,)))
                channel.add(Activation('relu'))
                channel.add(Dense(10))

                return channel

            generate_blocks = [block_0, block_1]
            return generate_blocks
        elif dataset == 'MNIST':
            # block definitions
            def block_0():
                channel = Sequential()
                channel.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
                channel.add(Activation('relu'))
                channel.add(Conv2D(32, (3, 3)))
                channel.add(Activation('relu'))
                channel.add(MaxPooling2D(pool_size=(2, 2)))
                channel.add(Conv2D(64, (3, 3)))
                channel.add(Activation('relu'))
                channel.add(Conv2D(64, (3, 3)))
                channel.add(Activation('relu'))
                channel.add(MaxPooling2D(pool_size=(2, 2)))
                channel.add(Flatten())
                channel.add(Dense(200))
                channel.add(Activation('relu'))
                channel.add(Dropout(0.3))

                return channel

            def block_1():
                channel = Sequential()
                channel.add(Dense(200, input_shape=(200,)))
                channel.add(Activation('relu'))
                channel.add(Dense(10))

                return channel

            generate_blocks = [block_0, block_1]
            return generate_blocks
        elif dataset == 'IMAGENET':
            # block definitions

            blocks = split_InceptionV3()

            def block_0():
                return blocks[0]

            def block_1():
                return blocks[1]

            generate_blocks = [block_0, block_1]
            return generate_blocks
        else:
            raise ValueError('Unrecognized dataset!')
    else:
        raise ValueError('Unrecognized split scheme indicator!')
