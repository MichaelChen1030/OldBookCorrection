pseudo = False
training = "vertical_test"

print ("Unigram Model Start")
# from freq_get_input import *

if pseudo:
	from freq_train_pseudo import *
else:
	from freq_train import *
from freq_get_output import *

from freq_get_black_input import *
from freq_train_black import *
from freq_get_black_output import *
print ("Unigram Model End")

print ("Word-level LSTM Pseudo Correction Test Start")
from word_get_input import *
if training == "horizontal":
	# from word_preprocess import *
	from word_train_horizontal import *
elif training == "vertical":
	# from word_preprocess import *
	from word_train_vertical import *
elif training == "horizontal_pseudo":
	from word_preprocess2_new import *
	from word_test_horizontal import *
elif training == "vertical_pseudo":
	from word_preprocess2_new import *
	from word_test_vertical import *
elif training == "vertical_test":
	from word_preprocess import *
	from word_test_vertical import *
from word_test_vertical import *

# from word_preprocess2_new import * # for pseudo testing
# from word_train_new import * # for pseudo testing
# from word_train_new2 import * # for pseudo testing
# from word_train_horizontal import *

print ("Word-level LSTM Pseudo Correction Test End")
# exit()
import pickle
import math
# exit()
# A*e

# unigram -> [Aae : 0.8, Abe: 0.05, Ace: 0.15]
# lstm -> [Aae: 0.15, Abe: 0.35, Ace: 0.5]

# Average
# Model -> [Aae : 0.8 + 0.15 * 4, Abe: 0.05 + 0.35 * 4, Ace : 0.15 + 0.5 * 4]

with open('freq_data/uni_second_step_avgs','rb') as f:
     unigram_avg=pickle.load(f)
with open('word_data/word_second_step_avgs','rb') as b:
     lstm_avg=pickle.load(b)


for i in unigram_avg:
	if sum(i.values())!=0:
		factor=1.0/sum(i.values())
		for k in i:
			i[k]=i[k]*factor

for i in lstm_avg:
	if sum(i.values())!=0:
		factor=1.0/sum(i.values())
		for k in i:
			i[k]=i[k]*factor

word_prob = []
new_array = []
maximum = 0
counter=-1
temp=0
word = ""
checker=0
holdword=""
stopcount=0

for i in range(len(lstm_avg)):
	maximum = 0
	stopcount=0
	checker = 0

	for key, value in unigram_avg[i].items():
		if stopcount==0:
			holdword=key
		stopcount=1

		for key1, value1 in lstm_avg[i].items():
			if key == key1:
				temp = ((value/5) + value1) / 2
				if temp >= maximum:
					word = key1
					maximum=temp
				checker=1
	if checker==0:
		word_prob.append(0)
		new_array.append(holdword)
	else:
		word_prob.append(maximum)
		new_array.append(word)

dot_file=open('freq_data/predicted', 'wb')
pickle.dump(new_array, dot_file)
dot_file.close()
prob_file=open('freq_data/probabilities', 'wb')
pickle.dump(word_prob, prob_file)
prob_file.close()

from word_get_output import *