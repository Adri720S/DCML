# ======================================================================
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative to its
# difficulty. So your Category 1 question will score significantly less than
# your Category 5 question.
# ======================================================================
#
# Basic Datasets Question
#
# Create a classifier for the Fashion MNIST dataset
# Note that the test will expect it to classify 10 classes and that the
# input shape should be the native size of the Fashion MNIST dataset which is
# 28x28 monochrome. Do not resize the data. Your input layer should accept
# (28,28,1) as the input shape only. And make sure your model training and validations performance above 90.  If you amend this, the tests will fail.

import numpy as np
import tensorflow as tf
from tensorflow import keras

def solution_model():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # YOUR PREPROCESSING CODE HERE
    # Preprocessing: Normalisasi (0-255 → 0-1) dan reshape agar sesuai input (28,28,1)
    train_images = train_images.reshape(-1, 28, 28, 1) / 255.0
    test_images = test_images.reshape(-1, 28, 28, 1) / 255.0

    # YOUR MODEL CODE HERE
    # Define Model (CNN untuk klasifikasi gambar)
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')  # Output layer: 10 kelas
    ])

    # Compile model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model
    model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels), verbose=2)

    # YOUR PREPROCESSING CODE HERE

    # YOUR MODEL CODE HERE
    
    return model

if __name__ == '__main__':
    model = solution_model()
    model.save("DCML2.h5")
