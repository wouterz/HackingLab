import argparse

from tensorflow_core.python.keras import Input
from tensorflow_core.python.keras.models import Model

from AdvModel import AdvModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('save_path_model1', help='Path to model1')
    parser.add_argument('save_path_model2', help='Path to model2')

    args = parser.parse_args()

    # Model with trained adv layer
    a_model_1 = AdvModel(epochs=1000, model_name="inception_v3", batch_size=50, center_size=35,
                           image_size=299,
                           adam_learn_rate=0.05, adam_decay=0.95, step=2, channels=3)
    model_1 = a_model_1.get_model()
    model_1.load_weights(args.save_path_model1)

    # Model for inception
    a_model_2 = AdvModel(epochs=1000, model_name="inception_v3", batch_size=50, center_size=299,
                           image_size=299,
                           adam_learn_rate=0.05, adam_decay=0.95, step=2, channels=3)
    model_2 = a_model_2.get_model()
    model_2.load_weights(args.save_path_model2)


    inputs = Input(shape=(35, 35, 3))
    al = model_1.get_layer('adv_layer')(inputs)

    # Combine layers
    outputs = model_2.get_layer('inception_v3')(al)

    model = Model(inputs=[inputs],
                  outputs=[outputs])

    print(model.summary())






