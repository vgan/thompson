import argparse

parser = argparse.ArgumentParser(description='List of avaible commands.')

parser.add_argument('--save_path',dest="save", type=str, nargs='?', help='Path to checkpoint storage')
parser.add_argument('--gen_len',dest="len",metavar="Number (1000)", type=int, nargs='?', help='Size of the generated text', default=1000)
parser.add_argument('--temperature',dest="temp",metavar="Number (1.0)", type=float, nargs='?', help='Low temperatures results in more predictable text.\n Higher temperatures results in more surprising text.', default=1.0)
args = parser.parse_args()


import tensorflow as tf

import numpy as np
import os
import time
import pickle
import string
import random

from colorama import init, Fore
init(autoreset=True)

def load_model_configs(directory):
    path = os.path.join(directory, "parameters.bin")
    return pickle.loads(open(path,'rb').read())

confs = load_model_configs(args.save)
print(Fore.LIGHTBLUE_EX + 'Loading existent configurations')

# Creating a mapping from unique characters to indices
char2idx = confs['char2idx']
idx2char = confs['idx2char']
vocab_size = confs['vocab_size']
embedding_dim = confs['embedding']
rnn_units = confs['units']
nlayers = confs['layers']

def build_model(vocab_size, embedding_dim, rnn_units, batch_size, nlayers):
    layers = []
    layers.append(tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]))

    for n in range(nlayers):
        layers.append(tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'))

    layers.append(tf.keras.layers.Dense(vocab_size))

    model = tf.keras.Sequential(layers)
    return model

# Directory where the checkpoints will be saved
checkpoint_dir = args.save

model = build_model(vocab_size, embedding_dim, rnn_units,
                    batch_size=1,nlayers=nlayers)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

model.summary()

def generate_text(model, start_string):
      # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = args.len

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = args.temp

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the word returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))

def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return u''.join(random.choice(letters) for i in range(stringLength))

print(generate_text(model, start_string=randomString(8)))

