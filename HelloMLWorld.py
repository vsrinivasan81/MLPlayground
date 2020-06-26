from silence_tensorflow import silence_tensorflow
import tensorflow as tf
from tensorflow import keras
import numpy as np


def build_execute_model(epochs_value, input_for_prediction):

    # Define the model
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')

    # Samples and Labels
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

    # Model training
    model.fit(xs, ys, epochs=epochs_value)

    # Predict using model
    print(model.predict([input_for_prediction]))


def main():

    # Suppress warnings from tensor-flow
    silence_tensorflow()
    # build and execute a linear regression model
    build_execute_model(500, 25)


if __name__ == "__main__":
    main()
