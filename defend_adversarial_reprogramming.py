import argparse
import os

from AdvModel import AdvModel


def defend_adversarial_reprogramming(model_indicator, split, epochs):
    save_dir = './Adversarial_Reprogramming/' + args.model_indicator + '/'
    try:
        os.makedirs(save_dir)
    except:
        pass

    dim_map = {
        'captcha': (35, 3),
        'mnist': (28, 3),
        'squares': (35, 3),
        'cifar10': (32, 3),
        'cifar100': (32, 3),
    }
    IMAGES = args.image_type
    CENTER_SIZE, CHANNELS = dim_map[IMAGES]
    BATCH_SIZE = args.batch_size
    SAVE_PATH = args.save_path
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    IMAGE_SIZE = 299
    ADAM_LEARN_RATE = 0.05
    ADAM_DECAY = 0.96
    DECAY_STEP = 2
    EPOCHS = 10000
    model = AdvModel(epochs=EPOCHS, model_name="inception_v3", batch_size=BATCH_SIZE, center_size=CENTER_SIZE,
                     image_size=IMAGE_SIZE, channels=CHANNELS,
                     adam_learn_rate=ADAM_LEARN_RATE, adam_decay=ADAM_DECAY, step=DECAY_STEP, hrs_defense=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_indicator', default='test_hrs[10][10]',
                        help='model indicator, format: model_name[5][5] for'
                             'a HRS model with 5 by 5 channels')
    parser.add_argument('--split', default='default', help='the indicator of channel structures in each block')
    parser.add_argument('--epochs', default=50, help='the number of epochs to train (reprogram).')

    args = parser.parse_args()
    defend_adversarial_reprogramming(model_indicator=args.model_indicator,
                                     split=args.split,
                                     epochs=args.epochs)
