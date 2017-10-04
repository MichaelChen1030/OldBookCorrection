# -*- coding: utf-8 -*-
from random import randint
import pickle
import string
import numpy as np
import argparse
import threading
import word_corpus_data2_new as data
import re
import math
from random import randint

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

with open('freq_data/user_input', 'rb') as f:
	year=pickle.load(f)
year1 = year[0]

counter=0
year_round=math.floor(int(year1)/10)

with open('word_data/dict_array_'+str(year_round)+'1','rb') as f:
	corpus2=pickle.load(f)

corpus = data.Corpus('word_data')
with open('word_data/dict_array2', 'wb') as handle:
    pickle.dump(corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)

data2 = corpus.test
i = 0
asterisks2 = []
asterisks_check = []
for i in range(0, len(data2)):
    # print (data2[i][0], data2[i][1])
    asterisks2.append([data2[i][0], data2[i][1]])
    # asterisks_check.append([data2[i][0], data2[i][1], corpus.test_dictionary.idx2word[int(data2[i][1])]])

    # asterisks2.append([i+2, data2[i]])

print ("===== asterisks_check =====")
# print (asterisks_check)
print (asterisks2)
# exit()
# added
# test_data=open('word_data/test_data', 'wb')
# pickle.dump(data2, test_data)
# test_data.close()

test_data=open('word_data/test_data', 'wb')
pickle.dump(asterisks2, test_data)
test_data.close()

data3 = corpus.rare_word
data3 = data3.astype(np.int64)
data3 = data3[len(data3)//10:]

with open('word_data/train_data_array', 'wb') as handle:
    pickle.dump(data3, handle, protocol=pickle.HIGHEST_PROTOCOL)

# data4 = corpus.train
data4 = corpus.rare_word
data4 = data4.astype(np.int64)
data4 = data4[:len(data4)//10]
with open('word_data/val_data_array', 'wb') as handle:
    pickle.dump(data4, handle, protocol=pickle.HIGHEST_PROTOCOL)
