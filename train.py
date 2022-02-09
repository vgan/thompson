from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import pickle

from colorama import init, Fore
init(autoreset=True)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='List of avaible commands.')
parser.add_argument('--data_path',dest="path", type=str, nargs='?',
                    help='Path to learning data')

parser.add_argument('--save_path',dest="save", type=str, nargs='?',
                    help='Path to checkpoint storage')

parser.add_argument('--epochs',dest="epochs",metavar="100", type=int, nargs='?',
                    help='Number of training epochs', default=100)

parser.add_argument('--n_batch',dest="batch",metavar="64", type=int, nargs='?',
                    help='Batch size', default=64)

parser.add_argument('--n_units',dest="units",metavar="512", type=int, nargs='?',
                    help='Number of LSTM Units', default=512)

parser.add_argument('--n_layers',dest="layers",metavar="3", type=int, nargs='?',
                    help='Number of LSTM Layers', default=3)

parser.add_argument('--n_sequence',dest="seq",metavar="100", type=int, nargs='?',
                    help='The maximum length sentence for a single input in characters', default=100)

parser.add_argument('--n_embedding',dest="embedding",metavar="128", type=int,
                    nargs='?', help='The embedding dimension size', default=128)

parser.add_argument("--continue",dest="cont",metavar="False", type=str2bool, 
                    nargs='?',const=True, default=False,help="Continue from last save.")

args = parser.parse_args()

import tensorflow as tf
import numpy as np
import os
import time

def save_model_configs(directory, params):
    path = os.path.join(directory, "parameters.bin")
    dumped = pickle.dumps(params)
    f = open(path, 'wb+') 
    f.write(dumped)

def load_model_configs(directory):
    path = os.path.join(directory, "parameters.bin")
    return pickle.loads(open(path,'rb').read())

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

def build_model(vocab_size, embedding_dim, rnn_units, batch_size, nl):
    layers = []
    layers.append(tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]))

    for n in range(nl):
        layers.append(tf.keras.layers.LSTM(rnn_units, return_sequences=True,
                        stateful=True, recurrent_initializer='glorot_uniform'))

    layers.append(tf.keras.layers.Dense(vocab_size))

    model = tf.keras.Sequential(layers)
    return model

@tf.function
def train_step(inp, target):
    with tf.GradientTape() as tape:
        predictions = model(inp)
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                target, predictions, from_logits=True))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss

text = open(args.path, 'rb').read().decode(encoding='utf-8')

vocab = sorted(set(text))
checkpoint_dir = args.save
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

confs = None
if os.path.exists(args.save) and args.cont:
    print(Fore.LIGHTGREEN_EX + '[Loading existent configurations]')
    try:
        confs = load_model_configs(args.save)
        embedding_dim = confs['embedding']
        rnn_units = confs['units']
        n_layers = confs['layers']
    except Exception as e:
        print(Fore.RED + 'Error loading checkpoint ' + str(e))   
        confs = None 
elif args.cont:
    if not os.path.exists(args.save):
        os.mkdir(args.save)
        print(Fore.RED + '[Directory created]')
    print(Fore.RED + '[No configurations to load]')

if confs is None:
    embedding_dim = args.embedding
    rnn_units = args.units
    n_layers = args.layers

# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])

# The maximum length sentence we want for a single input in characters
seq_length = args.seq
examples_per_epoch = len(text)//seq_length

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
dataset = sequences.map(split_input_target)

# Batch size
BATCH_SIZE = args.batch
steps_per_epoch = examples_per_epoch//BATCH_SIZE

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Length of the vocabulary in chars
vocab_size = len(vocab)

model = build_model(
    vocab_size = len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE,
    nl=n_layers)

if os.path.exists(args.save) and args.cont:
    print(Fore.LIGHTBLUE_EX + '[Loading existent checkpoint]')
    try:
        model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    except Exception as e:
        print(Fore.RED + 'Error loading checkpoint ' + str(e))   
elif args.cont:
    print(Fore.RED + '[No checkpoints to load]')

if confs is None:
    embedding_dim = args.embedding
    rnn_units = args.units
    n_layers = args.layers

EPOCHS=args.epochs

if confs is None:
    confs = {
        'units': args.units,
        'embedding': args.embedding,
        'layers': args.layers,
        'vocab_size': vocab_size,
        'char2idx': char2idx,
        'idx2char': idx2char,
    }
    save_model_configs(args.save, confs)

model.summary()
print (Fore.CYAN + 'Length of text: {} characters'.format(len(text)))
print (Fore.CYAN + '{} unique characters'.format(len(vocab)))

optimizer = tf.keras.optimizers.Adam()

train_start = time.time()
for epoch in range(EPOCHS):
  start = time.time()

  # initializing the hidden state at the start of every epoch
  # initally hidden is None
  hidden = model.reset_states()
  for (batch_n, (inp, target)) in enumerate(dataset):
    loss = train_step(inp, target)

    if batch_n % 100 == 0:
      template = Fore.LIGHTYELLOW_EX + 'Epoch [{}/{}] Batch [{}/{}] Loss {}'
      print(template.format(epoch+1,EPOCHS, batch_n, steps_per_epoch, loss))

  # saving (checkpoint) the model every 5 epochs
  if (epoch + 1) % 5 == 0:
    model.save_weights(checkpoint_prefix.format(epoch=epoch))
    print (Fore.LIGHTYELLOW_EX + '[Model saved]\n')  

  print (Fore.LIGHTWHITE_EX + '\n[Epoch {} Loss {:.4f}]'.format(epoch+1, loss))
  print (Fore.GREEN + '[Time taken for 1 epoch {} sec]\n'.format(time.time() - start))

print (Fore.LIGHTGREEN_EX + '\n[Total time {} mins]\n'.format((time.time() - train_start)/60))
model.save_weights(checkpoint_prefix.format(epoch=epoch))

