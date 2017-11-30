import os
import torch
import numpy as np
import re # added
import threading
import pickle
import math
from random import randint
import string
import xml.etree.ElementTree as ET
from xml.dom import minidom

class MyThread (threading.Thread):
    def __init__(self, id, data, corpus):
        threading.Thread.__init__(self)
        # store each document's ids
        self.id = id
        self.data = data
        self.corpus = corpus
    def run(self):
        data_array = np.array([])
        for i in range(len(self.data) // 8 * self.id, min(len(self.data) // 8 * (self.id + 1), len(self.data))):
            data_array = np.append(data_array, self.corpus.dictionary.word2idx[self.data[i]])

        if i % (len(self.data) // 50) == 0:
            print("Thread {} at {:2.1f}%".format(self.id, 100 * (i - len(self.data) // 8 * self.id) /
                  (min(len(self.data) // 8 * (self.id + 1), len(self.data)) - len(self.data) // 8 * self.id)))

        with open('word_data/data_array_{}'.format(self.id), 'wb') as handle:
            pickle.dump(data_array, handle, protocol=pickle.HIGHEST_PROTOCOL)

class MyThread2 (threading.Thread):
    def __init__(self, id, data, corpus):
        threading.Thread.__init__(self)
        self.id = id
        self.data = data
        self.corpus = corpus
    def run(self):
        data_array = np.array([])
        for i in range(len(self.data) // 8 * self.id, min(len(self.data) // 8 * (self.id + 1), len(self.data))):
            pattern = re.compile(r'●')
            text_list = (pattern.findall(self.data[i]))
            if len(text_list) >= 1:
                data_array = np.append(data_array, self.corpus.rare_dictionary.word2idx["<bd>"]) # added
            else:
                if self.data[i] in self.corpus.unique_word:
                    data_array = np.append(data_array, self.corpus.rare_dictionary.word2idx[self.data[i]]) # added
                else:
                    data_array = np.append(data_array, self.corpus.rare_dictionary.word2idx["<unk>"]) # added

            if i % (len(self.data) // 50) == 0:
                print("Thread {} at {:2.1f}%".format(self.id, 100 * (i - len(self.data) // 8 * self.id) /
                      (min(len(self.data) // 8 * (self.id + 1), len(self.data)) - len(self.data) // 8 * self.id)))

        with open('word_data/rare_data_array_{}'.format(self.id), 'wb') as handle:
            pickle.dump(data_array, handle, protocol=pickle.HIGHEST_PROTOCOL)


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):

        with open('freq_data/user_input', 'rb') as f:
            year=pickle.load(f)
        year1 = year[0]

        counter=0
        year_round=math.floor(int(year1)/10)

        with open('word_data/dict_array_'+str(year_round)+'1','rb') as handle:
            load_corpus = pickle.load(handle)

        self.dictionary = load_corpus.dictionary
        self.unique_word = load_corpus.unique_word

        self.rare_dictionary = Dictionary()
        self.test_dictionary = Dictionary()
        self.test_dictionary2 = []
        self.unique = {}
        self.test = self.generate_list_of_words(os.path.join(path, 'old_books.txt'))
        self.rare_word = self.tokenize3(os.path.join(path, 'old_books.txt'))

    def tokenize(self, path):
        assert os.path.exists(path)
        with open(path, 'r', encoding='UTF-8', newline='') as f:
            tokens = 0
            for line in f:
                words = re.split("[,.:; ]", line)
                words += ['<eos>']

                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        with open(path, 'r', encoding='UTF-8', newline='') as f:
            ids = np.array([])
            token = 0
            for line in f:
                words = re.split("[,.:; ]", line)
                words += ['<eos>']

                thread0 = MyThread(0, words, self)
                thread1 = MyThread(1, words, self)
                thread2 = MyThread(2, words, self)
                thread3 = MyThread(3, words, self)
                thread4 = MyThread(4, words, self)
                thread5 = MyThread(5, words, self)
                thread6 = MyThread(6, words, self)
                thread7 = MyThread(7, words, self)
                thread8 = MyThread(8, words, self)
                thread0.start()
                thread1.start()
                thread2.start()
                thread3.start()
                thread4.start()
                thread5.start()
                thread6.start()
                thread7.start()
                thread8.start()
                thread0.join()
                thread1.join()
                thread2.join()
                thread3.join()
                thread4.join()
                thread5.join()
                thread6.join()
                thread7.join()
                thread8.join()

        with open('word_data/data_array_0', 'rb') as handle:
            data_array_0 = pickle.load(handle)
        with open('word_data/data_array_1', 'rb') as handle:
            data_array_1 = pickle.load(handle)
        with open('word_data/data_array_2', 'rb') as handle:
            data_array_2 = pickle.load(handle)
        with open('word_data/data_array_3', 'rb') as handle:
            data_array_3 = pickle.load(handle)
        with open('word_data/data_array_4', 'rb') as handle:
            data_array_4 = pickle.load(handle)
        with open('word_data/data_array_5', 'rb') as handle:
            data_array_5 = pickle.load(handle)
        with open('word_data/data_array_6', 'rb') as handle:
            data_array_6 = pickle.load(handle)
        with open('word_data/data_array_7', 'rb') as handle:
            data_array_7 = pickle.load(handle)
        with open('word_data/data_array_8', 'rb') as handle:
            data_array_8 = pickle.load(handle)

        data_array = np.append(data_array_0, [data_array_1, data_array_2, data_array_3,
                                          data_array_4, data_array_5, data_array_6, data_array_7])
        data_array = np.append(data_array, data_array_8)
        return data_array
#pseudocorrection
    def generate_list_of_words(self, path):
        test_data_list = []
        ids = []
        i = 0
        j = 0
        # read from the xml
        tree = ET.parse('freq_data/pseudo_correction.xml')
        root = tree.getroot()

        for word in root.iter("item"):
            element_word = word.find("word").text
            element_index = int(word.find("index").text)
            element_asterisked = word.find("asterisked").text
            print (element_word, element_index, element_asterisked)
            element_unigram = word.find("unigram")

            element_confidence = float(element_unigram.find("confidence").text)
        

        #     if element_asterisked not in self.test_dictionary.word2idx:
        #         test_data_list.append(element_word)
        #         self.test_dictionary.add_word(element_asterisked)
        #         ids.append([element_index, str(self.test_dictionary.word2idx[element_asterisked])])
        #         i += 1
        #         j += 1
        #     else:
        #     	print ("#######################")
        #     	i += 1
        # print (i, j)

            if element_confidence <= .75:
                test_data_list.append(element_word)
                self.test_dictionary2.append([element_asterisked, i])

                # # self.test_dictionary2.add_word(element_asterisked)
                # # ids.append([element_index, str(self.test_dictionary2[element_asterisked])])
                ids.append([element_index, str(i)])
                i += 1


        # assert os.path.exists(path)
        # list_indices = []
    
        # test_data_list = []
        # asterisks2 = []
        # with open(path, 'r', encoding='UTF-8', newline='') as f:
        #     ids = []

        #     for line in f:
        #         words = re.split("[,.:; ]", line)
        #         words += ['<eos>']
        #         words = list(filter(None, words))

        #         i = 0
        #         while i < 100:
        #             index = randint(0, len(words) - 1)
        #             pattern = re.compile(r'●')
        #             if not pattern.match(words[index]) and words[index] not in string.digits + string.punctuation and len(words[index]) > 1:
        #                 word = words[index]
        #                 original = word
        #                 # test_data_list.append(word)
        #                 rand_num = randint(0, len(word)-1)
                        
        #                 word = word[:rand_num] + "●" + word[rand_num+1:]
        #                 if word not in self.test_dictionary.word2idx:
        #                     test_data_list.append(words[index])
        #                     self.test_dictionary.add_word(word)
        #                     ids.append([index, str(self.test_dictionary.word2idx[word])])
        #                     i += 1

                # for word in words:

                #     pattern = re.compile(r'●')
                #     text_list = (pattern.findall(word))

                #     if len(text_list) >= 1:
                #         self.test_dictionary.add_word(word)
                #         ids.append(str(self.test_dictionary.word2idx[word]))

        # asterisks2 = []
        # for i in range(0, len(ids)):
        #     asterisks2.append([i, ids[i]])

        # test_data=open('word_data/test_data', 'wb')
        # pickle.dump(asterisks2, test_data)
        # test_data.close()



        test_data2=open('word_data/test_data2', 'wb')
        pickle.dump(test_data_list, test_data2)
        test_data2.close()
        
        return ids


    def tokenize3(self, path):
        assert os.path.exists(path)
        with open(path, 'r', encoding='UTF-8', newline='') as f:
            tokens = 0
            for line in f:
                words = re.split("[,.:; ]", line)
                words += ['<eos>']
                tokens += len(words)

                for word in words:
                    pattern = re.compile(r'●')
                    text_list = (pattern.findall(word))
                    if len(text_list) >= 1:
                        self.rare_dictionary.add_word("<bd>")
                    else:
                        if word in self.unique_word:
                            self.rare_dictionary.add_word(word)
                        else:
                            self.rare_dictionary.add_word("<unk>")

        with open(path, 'r', encoding='UTF-8', newline='') as f:
            ids = np.array([])
            token = 0
            for line in f:
                words = re.split("[,.:; ]", line)
                words += ['<eos>']
                                
                thread0 = MyThread2(0, words, self)
                thread1 = MyThread2(1, words, self)
                thread2 = MyThread2(2, words, self)
                thread3 = MyThread2(3, words, self)
                thread4 = MyThread2(4, words, self)
                thread5 = MyThread2(5, words, self)
                thread6 = MyThread2(6, words, self)
                thread7 = MyThread2(7, words, self)
                thread8 = MyThread2(8, words, self)
                thread0.start()
                thread1.start()
                thread2.start()
                thread3.start()
                thread4.start()
                thread5.start()
                thread6.start()
                thread7.start()
                thread8.start()
                thread0.join()
                thread1.join()
                thread2.join()
                thread3.join()
                thread4.join()
                thread5.join()
                thread6.join()
                thread7.join()
                thread8.join()

        with open('word_data/rare_data_array_0', 'rb') as handle:
            data_array_0 = pickle.load(handle)
        with open('word_data/rare_data_array_1', 'rb') as handle:
            data_array_1 = pickle.load(handle)
        with open('word_data/rare_data_array_2', 'rb') as handle:
            data_array_2 = pickle.load(handle)
        with open('word_data/rare_data_array_3', 'rb') as handle:
            data_array_3 = pickle.load(handle)
        with open('word_data/rare_data_array_4', 'rb') as handle:
            data_array_4 = pickle.load(handle)
        with open('word_data/rare_data_array_5', 'rb') as handle:
            data_array_5 = pickle.load(handle)
        with open('word_data/rare_data_array_6', 'rb') as handle:
            data_array_6 = pickle.load(handle)
        with open('word_data/rare_data_array_7', 'rb') as handle:
            data_array_7 = pickle.load(handle)
        with open('word_data/rare_data_array_8', 'rb') as handle:
            data_array_8 = pickle.load(handle)

        data_array = np.append(data_array_0, [data_array_1, data_array_2, data_array_3,
                                          data_array_4, data_array_5, data_array_6, data_array_7])
        data_array = np.append(data_array, data_array_8)

        return data_array
