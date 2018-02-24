from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import time
import csv
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed
import argparse
from RNN_utils import *

# Parsing arguments for Network definition
ap = argparse.ArgumentParser()
ap.add_argument('-data_dir', default='./HG2U.txt')
ap.add_argument('-batch_size', type=int, default=50)
ap.add_argument('-layer_num', type=int, default=2)
ap.add_argument('-seq_length', type=int, default=50)
ap.add_argument('-hidden_dim', type=int, default=500)
ap.add_argument('-generate_length', type=int, default=210)
ap.add_argument('-nb_epoch', type=int, default=20)
ap.add_argument('-mode', default='train')
ap.add_argument('-weights', default='')
args = vars(ap.parse_args())

DATA_DIR = args['data_dir']
BATCH_SIZE = args['batch_size']
HIDDEN_DIM = args['hidden_dim']
SEQ_LENGTH = args['seq_length']
WEIGHTS = args['weights']

GENERATE_LENGTH = args['generate_length']
LAYER_NUM = args['layer_num']

# Creating training data
print('Creating Training Data')
X, y, VOCAB_SIZE, ix_to_char = load_data(DATA_DIR, SEQ_LENGTH)


# Creating and compiling the Network
print('compiling the network \n')
model = Sequential()
model.add(LSTM(HIDDEN_DIM, input_shape=(None, VOCAB_SIZE), return_sequences=True))
for i in range(LAYER_NUM - 1):
  model.add(LSTM(HIDDEN_DIM, return_sequences=True))
model.add(TimeDistributed(Dense(VOCAB_SIZE)))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
Logger = Log()
print('network compiled \n')

# Generate some sample before training to know how bad it is!
print('Gen samp txt')
sampleText =generate_text(model, args['generate_length'], VOCAB_SIZE, ix_to_char)

print('Sample text data from before training!!!!! \n')
print(sampleText)
Logger.AddEvent(-1, sampleText)

if not WEIGHTS == '':
  model.load_weights(WEIGHTS)
  nb_epoch = int(WEIGHTS[WEIGHTS.rfind('_') + 1:WEIGHTS.find('.')])
else:
  nb_epoch = 0


print ('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n \n')
print ('Training started this is going to take a while \n\n\n')
# Training if there is no trained weights specified
if args['mode'] == 'train' or WEIGHTS == '':
  print('training started!!! This may take a while.')
  while True:
    print('\n\nEpoch: {}\n'.format(nb_epoch))
    model.fit(X, y, batch_size=BATCH_SIZE, verbose=1, nb_epoch=1)
    nb_epoch += 1
    print('creating text from epoch {} \n'.format(nb_epoch))
    GenText = generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char)
    print("---------------Generated Text-------------\n")
    print(GenText)
    print('\n -----------------------------------------')
    Logger.AddEvent(nb_epoch, GenText)
    print('NB Epoch {}'.format(nb_epoch))
    if nb_epoch % 10 == 0:
      model.save_weights('checkpoint_layer_{}_hidden_{}_epoch_{}.hdf5'.format(LAYER_NUM, HIDDEN_DIM, nb_epoch))

# Else, loading the trained weights and performing generation only
elif WEIGHTS == '':
  # Loading the trained weights
  model.load_weights(WEIGHTS)
  generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char)
  print('\n\n')
else:
  print('\n\nNothing to do!')
