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
import twitter as twit
from configparser import ConfigParser
from RNN_utils import *
UseTwitter = False 

config = ConfigParser()
config.read('DNA_Bot.ini')

# Check your twitter connection 
if UseTwitter == True:
  api = twit.Api(consumer_key= config.get('Twitter_Settings','consumer_key'),
                        consumer_secret= config.get('Twitter_Settings', 'consumer_secret'),
                        access_token_key= config.get('Twitter_Settings', 'access_token_key'),
                        access_token_secret= config.get( 'Twitter_Settings', 'access_token_secret'))
  users = api.GetFriends()
  print([u.name for u in users])

# instantiate
DATA_DIR = config.get('ML_Settings','data_dir')
BATCH_SIZE = config.getint('ML_Settings', 'batch_size')
HIDDEN_DIM = config.getint('ML_Settings','hidden_dim')
SEQ_LENGTH = config.getint('ML_Settings','seq_length')
EPOCH = config.get('ML_Settings', 'epoch')
WEIGHTS = config.get('ML_Settings','weights')
GENERATE_LENGTH = config.getint('ML_Settings','gen_length')
LAYER_NUM = config.getint('ML_Settings', 'layer_num')
MODE = config.get('ML_Settings', 'mode')

# Creating training data
print('Creating Training Data')
X, y, VOCAB_SIZE, ix_to_char, char_to_ix = load_data(DATA_DIR, SEQ_LENGTH)


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


if not WEIGHTS == '':
  model.load_weights(WEIGHTS)
  #nb_epoch = int(WEIGHTS[WEIGHTS.rfind('_') + 1:WEIGHTS.find('.')])
  nb_epoch = int(EPOCH)
else:
  nb_epoch = 0

# Generate some sample before training to know how bad it is!
print('Gen samp txt')
sampleText =generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char, char_to_ix)
#status = api.PostUpdate(sampleText)
#print('Posted to twitter :: {}'.format(status.text))
print('Sample text data from before training!!!!! \n')
print(sampleText)
Logger.AddEvent(-1, sampleText)



print ('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n \n')
print ('Training started this is going to take a while \n\n\n')
# Training if there is no trained weights specified
if MODE == 'train' or WEIGHTS == '':
  while True:
    print('\n\nEpoch: {}\n'.format(nb_epoch))
    model.fit(X, y, batch_size=BATCH_SIZE, verbose=1, nb_epoch=1)
    nb_epoch += 1
    print('creating text from epoch {} \n'.format(nb_epoch))
    GenText = generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char, char_to_ix)
    print("---------------Generated Text-------------\n")
    print(GenText)
    if UseTwitter == True:
      api = twit.Api(consumer_key= config.get('Twitter_Settings','consumer_key'),
                      consumer_secret= config.get('Twitter_Settings', 'consumer_secret'),
                      access_token_key= config.get('Twitter_Settings', 'access_token_key'),
                      access_token_secret= config.get( 'Twitter_Settings', 'access_token_secret'))
      status = api.PostUpdate(GenText)
    print('\n -----------------------------------------')
    Logger.AddEvent(nb_epoch, GenText)
    print('NB Epoch {}'.format(nb_epoch))
    model.save_weights('checkpoint_layer_{}_hidden_{}_epoch_{}.hdf5'.format(LAYER_NUM, HIDDEN_DIM, nb_epoch))

# Else, loading the trained weights and performing generation only
elif WEIGHTS == '':
  # Loading the trained weights
  model.load_weights(WEIGHTS)
  generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char, char_to_ix)
  print('\n\n')
else:
  print('\n\nNothing to do!')
