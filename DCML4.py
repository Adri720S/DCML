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
# NLP QUESTION
#
# Build and train a classifier for the sarcasm dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid as shown.
# It will be tested against a number of sentences that the network hasn't previously seen
# and you will be scored on whether sarcasm was correctly detected in those sentences with minimal accuracy and validation is 80.
import json
import urllib.request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def solution_model():
    url = 'http://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    urllib.request.urlretrieve(url, 'sarcasm.json')

    # DO NOT CHANGE THIS CODE OR THE TESTS MAY NOT WORK
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []
    
    # YOUR CODE HERE LOAD DATASET
    with open('sarcasm.json', 'r') as f:
        data = json.load(f)

    # YOU NEED TO FIND OUT THE FEATURE/VARIABLE AND LABEL FROM SARCASM.JSON
    
    # DEFINE FEATURE AND LABELS
    # Extract sentences & labels
    sentences = [item['headline'] for item in data]
    labels = [item['is_sarcastic'] for item in data]

    # SPLIT TRAINING AND VALIDATION
    train_sentences = sentences[:training_size]
    train_labels = labels[:training_size]
    val_sentences = sentences[training_size:]
    val_labels = labels[training_size:]

    # Konversi label ke numpy array
    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)

    # TOKENIZER
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(train_sentences)

    # PAD SEQUENCES
    # Konversi teks ke sequence
    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    val_sequences = tokenizer.texts_to_sequences(val_sentences)

    # Padding sequences agar semua memiliki panjang yang sama
    train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    val_padded = pad_sequences(val_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    model = tf.keras.Sequential([
    # YOUR CODE HERE. KEEP THIS BINARY OUTPUT LAYER INTACT OR TESTS MAY FAIL
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer untuk binary classification
    ])

    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Train model
    model.fit(train_padded, train_labels,
              epochs=10,
              validation_data=(val_padded, val_labels),
              verbose=2)

    return model

if __name__ == '__main__':
    model = solution_model()
    model.save("DCML4.h5")
