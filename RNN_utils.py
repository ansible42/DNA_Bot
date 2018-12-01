from __future__ import print_function
import numpy as np
import time as time_lib
import datetime as dt_lib
import csv as lib_csv

# method for generating text
def generate_text(model, length, vocab_size, ix_to_char, char_to_ix):
	# starting with random character
	#ix = [np.random.randint(vocab_size)]
	listData = open('./SeedList.txt', 'r')
	SeedList = listData.readlines()
	for i in range(len(SeedList)):
		SeedList[i] = str(SeedList[i]).replace('\n',' ')
	#SeedList = ['Ford', 'Aurther', 'Zephod', 'Young', 'Lemurs', 'Apple', 'Whiskey', 'Actually', 'Wode', 'Mongolia', 'ZZ 9 Plural Z Alpha', 'Dirk' ]
	print("Seed List :: {0}".format(', '.join(map(str, SeedList))))	
	SeedWord = (SeedList[np.random.randint(len(SeedList)-1)])
	ix = [char_to_ix[c] for c in SeedWord]
	y_char = []
	X = np.zeros((1, (length+len(ix)) , vocab_size))
	for i in range(len(ix)):
		X[0, i, :][ix[i]] = 1
		y_char.append(ix_to_char[ix[i]])

	SeedLen = len(ix)
	#y_char = [ix_to_char[ix[-1]]]
	for i in range(length):
		Posn = SeedLen + i 
		# appending the last predicted character to sequence
		X[0, Posn, :][ix[-1]] = 1
		#print(ix_to_char[ix[-1]], end="")
		ix = np.argmax(model.predict(X[:, :Posn+1, :])[0], 1)
		y_char.append(ix_to_char[ix[-1]])
	return ('').join(y_char)

# method for preparing the training data
def load_data(data_dir, seq_length):
	data = open(data_dir, 'r').read()
	chars = list(set(data))
	
	print('Character List :: ')
	print("{0}".format(', '.join(map(str, chars))))
	VOCAB_SIZE = len(chars)

	print('Data length: {} characters'.format(len(data)))
	print('Vocabulary size: {} characters'.format(VOCAB_SIZE))

	ix_to_char = {ix:char for ix, char in enumerate(chars)}
	char_to_ix = {char:ix for ix, char in enumerate(chars)}
	shape = int(len(data)/seq_length)
	X = np.zeros((shape, seq_length, VOCAB_SIZE))
	y = np.zeros((shape, seq_length, VOCAB_SIZE))
	for i in range(0, shape):
		X_sequence = data[i*seq_length:(i+1)*seq_length]
		X_sequence_ix = [char_to_ix[value] for value in X_sequence]
		input_sequence = np.zeros((seq_length, VOCAB_SIZE))
		for j in range(seq_length):
			input_sequence[j][X_sequence_ix[j]] = 1.
			X[i] = input_sequence

		y_sequence = data[i*seq_length+1:(i+1)*seq_length+1]
		y_sequence_ix = [char_to_ix[value] for value in y_sequence]
		target_sequence = np.zeros((seq_length, VOCAB_SIZE))
		for j in range(seq_length):
			target_sequence[j][y_sequence_ix[j]] = 1.
			y[i] = target_sequence
	return X, y, VOCAB_SIZE, ix_to_char, char_to_ix


class EventType(object):
	def __init__(self, Epoch, GenText, TS):
		self.Epoch = Epoch
		self.GenText = GenText
		self.TS = TS 


class Log(object):
	def __init__(self):
		self.AllEvents = [] 
		self.LogFile = ('LogFile'+ (dt_lib.datetime.fromtimestamp(time_lib.time()).strftime('%m.%d.%Y %H%M.%S')) +'.csv')
		CSVFile = open(self.LogFile, 'w')
		fieldnames = ['Epoch', 'GeneratedText', 'Timestamp']
		writer = lib_csv.DictWriter(CSVFile, fieldnames=fieldnames)
		CSVFile.close()
	def AddEvent(self, Epoch, GenText):
		self.AllEvents.append(EventType(Epoch, GenText, time_lib.time()))
		CSVFile = open(self.LogFile ,'w')
		fieldnames = ['Epoch', 'GeneratedText', 'Timestamp']
		CSVFile.flush()
		writer = lib_csv.DictWriter(CSVFile, fieldnames=fieldnames)
		writer.writeheader()
		for Event in self.AllEvents:
			ts = dt_lib.datetime.fromtimestamp(Event.TS).strftime('%m.%d.%Y %H:%M:%S')
			writer.writerow({'Epoch': Event.Epoch, 'GeneratedText': Event.GenText, 'Timestamp' : ts })
		CSVFile.close()
