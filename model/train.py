import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def load_data():


    (train_images, train_labels), (test_images, test_labels)=tf.keras.datasets.fashion_mnist.load_data()
    
    # normalization

    train_images=train_images / 255.0
    test_images=test_images / 255.0

    return (train_images, train_labels), (test_images, test_labels)


def create_model():

    model=models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


# train the model

def train_model():

    (train_images, train_labels), (test_images, test_labels)=load_data()

    # reshaping images for CNN

    train_images=np.expand_dims(train_images, axis=-1)
    test_images=np.expand_dims(test_images, axis=-1)

    model=create_model()
    model.fit(train_images, train_labels, epochs=50, validation_data=(test_images, test_labels))

    # saving the model

    model.save('fashion_mnist_model.h5')


if __name__ == "__main__":
    train_model()