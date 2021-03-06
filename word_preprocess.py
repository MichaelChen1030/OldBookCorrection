# -*- coding: utf-8 -*-
from random import randint
import pickle
import string
import numpy as np
import argparse
import threading
import word_corpus_data
import re

################################################################
# Options
################################################################

parser = argparse.ArgumentParser(description='Data Generator')
parser.add_argument('--asterisk', type=bool, default=True,
                    help='Generate asterisked PTB corpus?')
parser.add_argument('--convert', type=bool, default=True,
                    help='Convert asterisked PTB corpus to one np array?')
parser.add_argument('--split', type=bool, default=True,
                    help='Split asterisked PTB corpus to train_data and val_data?')
parser.add_argument('--train', type=bool, default=True,
                    help='Convert train_data to np array?')
parser.add_argument('--val', type=bool, default=True,
                    help='Convert val_data to np array?')
args = parser.parse_args()

#################################################################
# Clean and asterisk PTB corpus, generate test data
#################################################################

path = "word_data"
corpus = word_corpus_data.Corpus(path)

with open(path + '/dict_array', 'wb') as handle:
    pickle.dump(corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)

test_set = corpus.test_data
test_data=open(path + '/test_data', 'wb')
pickle.dump(test_set, test_data)
test_data.close()

train_set = corpus.train_data
train_set = train_set.astype(np.int64)
train_set = train_set[len(train_set)//10:]
with open(path + '/train_data_array', 'wb') as handle:
    pickle.dump(train_set, handle, protocol=pickle.HIGHEST_PROTOCOL)

valid_set = corpus.train_data
valid_set = valid_set.astype(np.int64)
valid_set = valid_set[:len(valid_set)//10]
with open(path + '/val_data_array', 'wb') as handle:
    pickle.dump(valid_set, handle, protocol=pickle.HIGHEST_PROTOCOL)
