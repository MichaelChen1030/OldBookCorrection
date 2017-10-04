# -*- coding: utf-8 -*-
import argparse
import torch
import torch.cuda as cuda
from torch.autograd import Variable
from word_lstm_model import *
import time
import math
import string
import pickle
import numpy as np
import bisect
import word_corpus_data as data
import re
import sys
import time
import xml.etree.ElementTree as ET
from xml.dom import minidom
import operator

write_file = open("word_data/final.xml", "w", newline='', encoding='UTF-8')

def progress(count, total, status=''):
    bar_len = 30
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s %s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

parser = argparse.ArgumentParser(description='PyTorch LSTM Model')
parser.add_argument('--nhid', type=int, default=128,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=3,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--load_epochs', type=int, default=0,
                    help='load epoch')
parser.add_argument('--epochs', type=int, default=15,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=100,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use CUDA')
parser.add_argument('--bidirectional', action='store_true', default=False,
                    help='use Bi-LSTM')
parser.add_argument('--serialize', action='store_true', default=False,
                    help='continue training a stored model')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
args = parser.parse_args()

# added
with open('word_data/test_data2', 'rb') as f:
    test_data_list=pickle.load(f)
print ("----- test_data_list -----")
print (test_data_list)
print (len(test_data_list))

with open('word_data/dict_array2', 'rb') as handle:
    corpus2 = pickle.load(handle)

with open('freq_data/user_input', 'rb') as f:
	year=pickle.load(f)
year1 = year[0]

year_round=math.floor(int(year1)/10)
with open('word_data/dict_array_'+str(year_round)+'1','rb') as f:
	corpus=pickle.load(f)

n_categories = len(corpus.dictionary)
n_letters = len(corpus.dictionary)
# n_categories = 50000
# n_letters = 50000

temp1 = []
temp2 = []
temp3 = {}
temp4 = {}
averages=[]


with open('word_data/test_data', 'rb') as handle:
    test_data = pickle.load(handle)
total_counter=0
total_counter=len(test_data)

with open('word_data/train_data_array', 'rb') as handle:
    train_data_array = pickle.load(handle)
with open('word_data/val_data_array', 'rb') as handle:
    val_data_array = pickle.load(handle)

all_data_array = np.append(train_data_array, val_data_array)
test_data.sort(key=lambda x: x[0], reverse=False)
test_target_index = []
test_target_tensor = torch.LongTensor(len(test_data)).zero_()

for i, char in enumerate(test_data):
    test_target_index.append(char[0])
    test_target_tensor[i] = int(char[1])


def batchify(data, bsz):
    nbatch = data.size // bsz
    data = data[0: nbatch * bsz]
    data = data.reshape(bsz, -1)
    return data

# Wraps hidden states in new Variables, to detach them from their history.
def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

# for every batch you call this to get the batch of data and targets, in testing your only getting astericks for the characters you took out, 
# you have the index but the data is turned into 3D array, you have to map indices that you have, 
#you have to search for what astericks/how many are in the array, and then you go back using the indices of the astericks you already know
# to map that into indices of output array that you have 
def get_batch(source, source_target, i, evaluation=False):
    seq_len = min(args.bptt, source.size(1) - i)  # -1 so that there's data for the target of the last time step
    data = Variable(source[:, i: i + seq_len], volatile=evaluation)   # Saves memory in purely inference mode
    if args.bidirectional:
        r_source_target = np.flip(source_target[:, i - 1: i - 1 + seq_len].cpu().numpy(), 1).copy()
        target = torch.cat((Variable(source_target[:, i + 1: i + 1 + seq_len].contiguous().view(-1)),
                            Variable(torch.from_numpy(r_source_target).cuda().contiguous().view(-1))), 0)
    else:
        target = Variable(source_target[:, i : i + 1 + seq_len].contiguous().view(-1))
    return data, target


def embed(data_array, bsz):
    data_array = batchify(data_array, bsz)
    data_tensor = torch.FloatTensor(data_array.shape[0], data_array.shape[1], n_letters).zero_()

    data_array = data_array.astype(np.int64)
    for i in range(0, data_array.shape[0]):
        for j in range(0, data_array.shape[1]):
            data_tensor[i][j][data_array[i][j]] = 1

    target_tensor = torch.LongTensor(data_array)
    return data_tensor, target_tensor

val_bsz = 5

###############################################################################
# Helper functions for searching within a sorted list
###############################################################################
def find_ge(a, x):
    """Find leftmost item greater than or equal to x"""
    i = bisect.bisect_left(a, x)
    return i

def find_le(a, x):
    """Find rightmost value less than or equal to x"""
    i = bisect.bisect_right(a, x)
    return i - 1

###############################################################################
# Build the model
###############################################################################
if args.bidirectional:
    name = 'Bi-LSTM'
else:
    name = 'LSTM'

if args.serialize:
    with open(path + '/{}_Epoch{}_BatchSize{}_Dropout{}_LR{}_HiddenDim{}.pt'.format(
             name, args.load_epochs, args.batch_size, args.dropout, args.lr, args.nhid), 'rb') as f:
        model = torch.load(f)
else:
    model = MyLSTM(n_letters, args.nhid, args.nlayers, True, True, args.dropout, args.bidirectional, args.batch_size, args.cuda)
    args.load_epochs = 0

if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()
softmax = nn.Softmax()
NLL = nn.NLLLoss()

if args.cuda:
    criterion.cuda()
    softmax.cuda()
    NLL.cuda()

def test():
    model.eval()
    idx = 0
    total_loss = 0
    correct_count = 0
    not_in_fixed_list_count = 0
    total_count = 0
    high_correct = 0
    high_total = 0
    last_forward = None
    batch_length = all_data_array.size // args.batch_size
    hidden = model.init_hidden(args.batch_size)
    start_time = time.time()
    progress_counter=0

    # import unigram XML output
    tree = ET.parse('freq_data/pseudo_correction.xml')
    root = tree.getroot()

    for batch, i in enumerate(range(1, batch_length - 1, args.bptt)):
        temp_array = np.zeros(0)

        for j in range(args.batch_size):
            temp_array1 = all_data_array[batch*args.bptt*(j+1)+1 : batch*args.bptt*(j+1) + args.bptt + 1]
            temp_array = np.concatenate((temp_array,temp_array1), axis = 0)

        all_data_tensor2, all_target_tensor2 = embed(temp_array, args.batch_size)
        all_data_tensor2 = all_data_tensor2.cuda()
        all_target_tensor2 = all_target_tensor2.cuda()
        data, _ = get_batch(all_data_tensor2, all_target_tensor2, 0)
        del all_data_tensor2, all_target_tensor2

    # for batch, i in enumerate(range(1, all_data_array.size - 1, args.bptt * args.batch_size)):
    #     # single_batch_array is a 1-dimensional array of size bsz*bptt - data for one batch
    #     single_batch_array = all_data_array[batch * args.bptt * args.batch_size : (batch + 1) * args.bptt * args.batch_size]
        
    #     data_tensor, target_tensor = embed(single_batch_array, args.batch_size)
    #     data_tensor = data_tensor.cuda()
    #     target_tensor = target_tensor.cuda()
    #     data, _ = get_batch(data_tensor, target_tensor, 0)

        hidden = model.init_hidden(args.batch_size)
        output, hidden = model(data, hidden)
        output_select = Variable(torch.FloatTensor(output.size(0), n_categories).zero_())
        target_select = Variable(torch.LongTensor(output.size(0)).zero_())
        index_select = []

        count = 0
#        bptt = output.size(0) / args.batch_size

#        start = find_ge(test_target_index, i + 1)
#        end = find_le(test_target_index, i + output.size(0) + 1) + 1
#        for ii in range(start, end):
#            target = test_target_index[ii]
#            temp1.append(target)

#            output_select[count] = output[target - i - 1]
#            target_select[count] = test_target_tensor[ii]
#            index_select[count] = test_target_index[ii]
#            count += 1

        bptt = min(batch_length - i, args.bptt)

        for batch_i in range(0, args.batch_size):
            start = find_ge(test_target_index, i + 1 + batch_i * batch_length)
            end = find_le(test_target_index, i + batch_i * batch_length + bptt) + 1
            for ii in range(start, end):
                target = test_target_index[ii]
                temp1.append(target)

                if target - batch_i * (batch_length - bptt) - i - 1 >= len(output):
                    output_select[count] = output[len(output) - 1]
                else:
                    output_select[count] = output[target - batch_i * (batch_length - bptt) - i - 1]
                target_select[count] = test_target_tensor[ii]
                index_select.append(test_target_index[ii])
                count += 1

        if count != 0:
            output_select = output_select[:count, :]
            target_select = target_select[:count]

            output_prob = softmax(output_select[:, :n_categories])

            for n, target in enumerate(target_select.data):
                # top_n, top_i = output_prob[n].data.topk(1)
                # category_i is the predicted category index for each word
                # category_i = top_i.cpu().numpy()

                find = ""
                # print ("target : ", target, corpus.rare_dictionary.idx2word[target])
                # print (target)
                # target_val = corpus.rare_dictionary.idx2word[target] 
                # target_val = corpus2.test_dictionary.idx2word[target] # use this
                # target_val is the target blackdot word
                target_asterisked = corpus2.test_dictionary2[target][0]
                target_index = index_select[n]
                print ("target : ", target, target_asterisked)
                # print ("##", target_val)
                for i in target_asterisked.lower():
                    if i == '●':
                        find += "\w"
                    elif i == '[':
                        find += '\['
                    elif i == ']':
                        find += '\]'
                    elif i == '(':
                        find += '\('
                    elif i == ')':
                        find += '\)'                        
                    else:
                        find += i
                pattern = re.compile(find)
                pattern_matched = False

                num = 0
                correct = False
                result = ""
                all_candidates={}

                digit_check = 0
                for rand in target_asterisked:
                    if rand in string.digits:
                        digit_check = 1
                if digit_check == 0:

                    top_n, top_i = output_prob[n].data.topk(len(output_prob[n]))
                    category_i = top_i.cpu().numpy()
                    for i in range(0, len(output_prob[n])):
                        curr_candidate = corpus.dictionary.idx2word[category_i[i]]
                        IsTitle = False
                        if curr_candidate.istitle():
                            curr_candidate = curr_candidate.lower()
                            IsTitle = True
                        
                        if len(target_asterisked) == len(curr_candidate):
                            plausible_candidate = re.match(pattern, curr_candidate)
                            # check if candidate matches the asterisked pattern
                            if plausible_candidate:
                                # print ("target_val, check : ", target_val, check)
                                if target_asterisked[0].isupper():
                                    curr_candidate = curr_candidate.replace(curr_candidate[0], curr_candidate[0].upper())
                                curr_candidate_prob = output_prob[n].data[category_i[i]]
                                
                                if num == 0:
                                    num += 1
                                    # stores top candidate word in result
                                    result = curr_candidate.lower()
                                    if target_asterisked != curr_candidate:
                                        if curr_candidate.lower() == test_data_list[target].lower():
                                            print ("#yes", target_asterisked, curr_candidate, test_data_list[target])
                                            correct_count += 1
                                            correct = True
                                        else:
                                            print ("no", target_asterisked, curr_candidate, test_data_list[target])
                                        all_candidates[curr_candidate] = curr_candidate_prob
                                        temp2.append(target_asterisked)

                                # elif num >= 1 and num <= 4:
                                elif num >= 1:
                                    num += 1
                                    if target_asterisked!=curr_candidate:
                                        print ("#yes" + str(num), target_asterisked, curr_candidate, test_data_list[target])
                                        all_candidates[curr_candidate] = curr_candidate_prob
                                # correct += 1
                                pattern_matched = True
                if not pattern_matched:
                    print ("not matched", target_asterisked)
                    temp2.append(target_asterisked)
                    all_candidates[target_asterisked] = 0
                    not_in_fixed_list_count += 1

                # sort all_candidates by probability
                sorted_candidates = sorted(all_candidates.items(), key=operator.itemgetter(1), reverse=True)

                # output candidates, candidate probabilities, correct
                for word in root.iter("item"):
                    if int(word.find("index").text) == target_index:
                        item_lstm = ET.SubElement(word, "lstm")
                        item_prediction = ET.SubElement(item_lstm, "prediction")
                        item_prediction.text = sorted_candidates[0][0]
                        item_correct = ET.SubElement(item_lstm, "correct")
                        if correct:
                            item_correct.text = "true"
                        else:
                            item_correct.text = "false"
                        item_confidence = ET.SubElement(item_lstm, "confidence")
                        item_confidence.text = str(sorted_candidates[0][1])
                        item_candidates = ET.SubElement(item_lstm, "candidates")
                        for j in range(0, len(sorted_candidates)):
                            item_candidate = ET.SubElement(item_candidates, "candidate")
                            item_candidate_name = ET.SubElement(item_candidate, "name")
                            item_candidate_name.text = sorted_candidates[j][0]
                            item_candidate_conf = ET.SubElement(item_candidate, "conf")
                            item_candidate_conf.text = str(sorted_candidates[j][1])
                        break
                    else:
                        continue

                averages.append([target, all_candidates])
                total_count += 1
                progress_counter+=1
                # progress(progress_counter,total_counter,status="Word LSTM Progress:")
                if top_n.cpu().numpy()[0] > 0:
                    high_total += 1
                    if category_i[0] == target:
                        high_correct += 1

            output_log = torch.log(output_prob)

        # if batch % args.log_interval == 0 and batch > 0:
    elapsed = time.time() - start_time

    start_time = time.time()
    print ("total_count = ", total_count, "correct = ", correct_count, "not_in_fixed_list_count = ", not_in_fixed_list_count)

    # write to XML

    xmlstr = '\n'.join([line for line in minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ").split("\n") if line.strip()])
    XML_output = "word_data/pseudo_correction.xml"
    with open(XML_output, "w", newline='', encoding='UTF-8') as f:
        f.write(xmlstr)

    return correct_count / total_count, high_correct / high_total, high_total / total_count

lr = args.lr
best_val_loss = None

test_accuracy, high_accuracy, high_percentage = test()
print ("test_accuracy = ", test_accuracy)

# for i in range(0, len(temp1)):
#     temp3[temp1[i]] = averages[i]

# for k in sorted(temp3.keys()):
#     temp4[k] = temp3[k]

# temp5 = []
# for k in temp4:
#     temp5.append(temp4[k])

# dot_file=open('word_data/word_second_step_avgs', 'wb')
# pickle.dump(temp5, dot_file)
# dot_file.close()

print ("---------------------------")
# average_dict = {}
# for a in averages:
#     average_dict[a[0]] = a[1]

# average_array = []
# for k in sorted(average_dict.keys()):
#     average_array.append(average_dict[k])

# for i in average_array:
#     checked = False
#     if sum(i.values())!=0:
#         if not checked:
#             for j in i:
#                 if not checked:
#                     print ("confidence : ", i[j] / sum(i.values()))
#                 checked = True
#         factor=1.0/sum(i.values())
#         for k in i:
#             i[k]=i[k]*factor

# for j in average_array:
#     for k, v in j.items():
#         print (k, v)  
