# ======================================================================
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative to its
# difficulty. So your Category 1 question will score significantly less than
# your Category 5 question.
# ==============================================================================
#
# ABOUT THE DATASET
#
# The dataset contains traffic sign boards from the streets captured into
# image files. There are 43 unique classes in total. The images are of shape
# (30,30,3).
# ==============================================================================
#
# INSTRUCTIONS
#
# We have already divided the data for training and validation.
#
# Complete the code in following functions:
# 1. preprocess()
# 2. solution_model()
#
# Your code will fail to be graded if the following criteria are not met:
# 1. The input shape of your model must be (30,30,3), because the testing
#    infrastructure expects inputs according to this specification.
# 2. The last layer of your model must be a Dense layer with 43 neurons
#    activated by softmax since this dataset has 43 classes.
#
# HINT: Your neural network must have a training and validation accuracy of approximately
# 0.95 or above on the normalized validation dataset for top marks.

# This function downloads and extracts the dataset to the directory that
# contains this file.
import tensorflow as tf
import numpy as np
import zipfile
import urllib.request
import os

# DO NOT CHANGE THIS CODE
# (unless you need to change https to http)
def download_and_extract_data():
    url = 'http://storage.googleapis.com/download.tensorflow.org/data/certificate/germantrafficsigns.zip'
    urllib.request.urlretrieve(url, 'germantrafficsigns.zip')
    with zipfile.ZipFile('germantrafficsigns.zip', 'r') as zip_ref:
        zip_ref.extractall()

# COMPLETE THE CODE IN THIS FUNCTION
def preprocess(image, label):
    # NORMALIZE YOUR IMAGES HERE (HINT: Rescale by 1/.255)
    image = tf.cast(image, tf.float32) / 255.0

    return image, label


# This function loads the data, normalizes and resizes the images, splits it into
# train and validation sets, defines the model, compiles it and finally
# trains the model. The trained model is returned from this function.

# COMPLETE THE CODE IN THIS FUNCTION.
def solution_model():
    # Downloads and extracts the dataset to the directory that
    # contains this file.
    download_and_extract_data()

    BATCH_SIZE = 32
    IMG_SIZE = (30, 30)

    # The following code reads the training and validation data from their
    # respective directories, resizes them into the specified image size
    # and splits them into batches. You must fill in the image_size
    # argument for both training and validation data.
    # HINT: Image size is a tuple
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory="train",
        label_mode="categorical",
        image_size=IMG_SIZE,  # YOUR CODE HERE
        batch_size = BATCH_SIZE
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory="validation",
        label_mode="categorical",
        image_size=IMG_SIZE,  # YOUR CODE HERE
        batch_size = BATCH_SIZE
    )

    # Normalizes train and validation datasets using the
    # preprocess() function.
    # Also makes other calls, as evident from the code, to prepare them for
    # training.
    # Do not batch or resize the images in the dataset here since it's already
    # been done previously.

    train_ds = train_ds.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(
        tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Code to define the model
    model = tf.keras.models.Sequential([
        # If you don't adhere to the instructions in the following comments,
        # tests will fail to grade your model:
        # The input layer of your model must have an input shape of
        # (30,30,3). You need to use at least 1 Conv2D layer.
        # Make sure your last layer has 43 neurons activated by softmax.
        # ADD LAYERS OF THE MODEL HERE
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(30, 30, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),  # Mengurangi overfitting
        tf.keras.layers.Dense(43, activation='softmax')  # Output layer (43 kelas)
    ])

    # Compile Model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train Model
    model.fit(train_ds, epochs=5, validation_data=val_ds, verbose=2)

    return model
if __name__ == '__main__':
    model = solution_model()
    model.save("DCML3.h5")
