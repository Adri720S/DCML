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
# Getting Started Question
#
# Given this data, train a neural network to match the xs to the ys
# So that a predictor for a new value of X will give a float value
# very close to the desired answer
# i.e. print(model.predict([10.0])) would give a satisfactory result
# The test infrastructure expects a trained model that accepts
# an input shape of [1] with MSE less than 1e-05

# define library
import numpy as np
import tensorflow as tf
from tensorflow import keras

def solution_model():
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    # Your code here
    # Define a simple Sequential model with one Dense layer
    model = keras.Sequential([
        keras.layers.Dense(units=1, input_shape=[1])  # Linear model (y = x + 1)
    ])

    # Compile the model
    model.compile(optimizer='sgd', loss='mse')

    # Train the model
    model.fit(xs, ys, epochs=500, verbose=0)

    # Evaluasi model: Hitung Mean Squared Error (MSE)
    mse = model.evaluate(xs, ys, verbose=0)
    print(f"Mean Squared Error (MSE): {mse:.10f}")

    return model

if __name__ == '__main__':
    model = solution_model()
    model.save("DCML1.h5")
