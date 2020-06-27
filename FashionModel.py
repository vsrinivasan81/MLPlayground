from silence_tensorflow import silence_tensorflow
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random


class AccuracyCallBack(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        accuracy_target = 0.92
        if logs.get('acc') > accuracy_target:
            print("\nReached {}% accuracy so cancelling training! \n".format(accuracy_target * 100))
            self.model.stop_training = True


def build_execute_model(no_of_epochs, batch_size, random_image_number):
    callbacks = AccuracyCallBack()
    input_data = keras.datasets.fashion_mnist

    (training_images, training_labels), (test_images, test_labels) = input_data.load_data()
    image_class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    training_images = training_images / 255.0
    test_images = test_images / 255.0

    print('Number of training images : ', len(training_images))
    print('Number of testing images : ', len(test_images))

    model = keras.models.Sequential([keras.layers.Flatten(input_shape=(28, 28)),
                                     keras.layers.Dense(512, activation=tf.nn.relu),
                                     keras.layers.Dense(10, activation=tf.nn.softmax)
                                     ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    training_results = model.fit(training_images,
                                 training_labels,
                                 batch_size=batch_size,
                                 epochs=no_of_epochs,
                                 callbacks=[callbacks])

    print('\nTraining Results')
    print(training_results.epoch,
          training_results.history['loss'],
          training_results.history['acc'],
          '\n')

    test_loss, test_acc = model.evaluate(test_images, test_labels)

    print('\nEvaluation Results')
    print(test_loss, test_acc)

    probability_model = keras.models.Sequential([model,
                                                 keras.layers.Softmax()])

    predictions = probability_model.predict(test_images)

    print('\nPrediction Results')
    print(random_image_number,
          'Predicted Label : ' + image_class_names[np.argmax(predictions[random_image_number])],
          'Actual Label : ' + image_class_names[test_labels[random_image_number]])


def main():
    print(tf.__version__)
    # Suppress warnings from tensor-flow
    silence_tensorflow()
    # build and execute a linear regression model
    build_execute_model(15, 64, random.randint(0, 10000))


if __name__ == "__main__":
    main()
