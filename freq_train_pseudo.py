import string
from random import randint
import re
import numpy
import operator
import pickle
import sys
import time
import xml.etree.ElementTree as ET
from xml.dom import minidom

pseudo_correction = True
XML_output = "freq_data/pseudo_correction.xml"

def progress(count, total, status=''):
    bar_len = 30
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s %s\r' % (bar, percents, '%', status))
    sys.stdout.flush()  

total_counter=0
actual_word_list = []

# extract black dot words into output_filename
def star_unique_file(input_filename, output_filename):
	input_file = open(input_filename, 'r', newline='', encoding='UTF-8')
	file_contents = input_file.read()
	input_file.close()
	countercheck=0
	duplicates = []
	word_list = file_contents.split()
	file = open(output_filename, 'w', newline='', encoding='UTF-8')
	for word in word_list:
		if "●" in word:
			countercheck+=1
			file.write(str(word) + " ")
	file.close()


# randomly assign one asterisks to 200 random non-black-dot word in input file
def generate_pseudo_correction_task(input, output, XML_output):
    # build an XML tree
    doc = ET.Element('document')
    input_file = open(input, 'r', newline='', encoding='UTF-8')
    file_contents = input_file.read()
    input_file.close()
    words = file_contents.split()
    file = open(output, 'w', newline='', encoding='UTF-8')
    i = 0
    while i < 200:
        index = randint(0, len(words) - 1)
        if (not re.search('●', words[index])) and (not re.search('[0-9]+', words[index])) and len(words[index]) > 1:
            item = ET.SubElement(doc, 'item')
            item_word = ET.SubElement(item, 'word')
            item_word.text = words[index]
            item_index = ET.SubElement(item, 'index')
            item_index.text = str(index)
            word = words[index]
            rand_num = randint(0, len(word) - 1)
            word = word[:rand_num] + "●" + word[rand_num + 1:]
            item_asterisked = ET.SubElement(item, 'asterisked')
            item_asterisked.text = word
            print("test_word : ", word, ", actual_word : ", words[index])
            actual_word_list.append(words[index])
            file.write(str(word) + " ")
            i += 1
    file.close()
    xmlstr = minidom.parseString(ET.tostring(doc)).toprettyxml(indent="    ")
    with open(XML_output, "w", newline='', encoding='UTF-8') as f:
        f.write(xmlstr)
    #tree = ET.ElementTree(doc)
    #tree.write(XML_output)


total_counter=0
if pseudo_correction:
	generate_pseudo_correction_task("freq_data/letters_with_dots.txt", "freq_data/test_words.txt", XML_output)
else:
	star_unique_file("freq_data/letters_with_dots.txt", "freq_data/test_words.txt")

deltas=[0.2,0.33,0.26,0.27,0.21,0.09,0.09,0.04,0.04,0.01,0,0,0,0,0,0,0,0,0,0]

dot_words=[]
with open('freq_data/test_words.txt', 'r', newline='', encoding='UTF-8') as myfile:
	data=myfile.read()
	dot_words = data.split()
total_counter=len(dot_words)
year = input('Enter the year: ')
if int(year) <= 1640:
	with open('freq_data/hash_table_1600','rb') as f:
		# wlist1: <word, occurrences> hash table
		wlist1=pickle.load(f)
else:
	with open('freq_data/hash_table_1640','rb') as f:
		wlist1=pickle.load(f)

li = []
li.append(year)
user_input=open('freq_data/user_input', 'wb')
pickle.dump(li, user_input)
user_input.close()


def test_word_model(star_words, delta,wlist):
    idx = 0
    temp1=""
    temp2=""
    temp4=[]
    average_dots=[]
    total_average_dots=[]
    checker=0
    index=0
    correct=0
    One_hundred_percent=0
    confidentcounter1=0
    confidentcounter2=0
    confidentcounter3=0
    confidentcounter4=0
    confidentcounter5=0
    confidentcounter6=0
    confidentcounter7=0
    confidentcounter8=0
    confidentcounter85=0
    confidentcounter75=0
    confidentcounter65=0
    confidentcounter55=0
    confidence_correct = 0
    confidence_total_counter=0

    dots=[]
    deltasums=[]
    freq_average95=0
    freq_average9=0
    freq_average85=0
    freq_average8=0
    freq_average75=0
    freq_average7=0
    freq_average65=0
    freq_average6=0
    freq_average55=0
    freq_average5=0
    freq_average4=0
    freq_average3=0
    probabilities=[]
    check_letter = 0
    progress_counter=0

    tree = ET.parse(XML_output)
    doc = tree.getroot()
    item_iter = doc.iter('item')

    for i in star_words:
        # defaults to empty <item> element to prevent StopIteration exception
        curr_item = next(item_iter, ET.Element('item'))
        item_unigram = ET.SubElement(curr_item, 'unigram')

        progress(progress_counter,total_counter,status="Unigram Progress:")
        progress_counter+=1
        checker=0
        predictions={}
        changecheck=0
        check_one=0
        score=0
        wordlength=len(i)

        sel = ''
        min = -1

        isNum=0
        for numbb in string.digits:
            if numbb in i:
                isNum=1
                break
        if isNum==0:
            counter1=0
            for j in wlist:
                counter=0
                if len(i)==len(j):
                    for x in range(len(i)):
                        if i[x]==j[x]:
                            check_one=1
                            break
                    if(len(i)==1):
                        check_one=1
                    if check_one==1:
                        dots=[m.start() for m in re.finditer('●', i)]
                        temp1=i.replace("●","")
                        temp2=j
                        temp3=""
                        for x in dots:
                            temp2 = temp2[:(x-counter)] + temp2[(x + 1-counter):]
                            counter+=1
                        counter1=counter
                        if temp1==temp2:
                            checker=1
                            predictions[i+" "+j+" "]=wlist[j]
        if checker==1 and check_letter == 0:
            # maximum is a string in the format of "Dotted_word Best_prediction"
            maximum=max(predictions, key=predictions.get)

            # sort predictions by normalized (delta?) confidence
            sorted_predictions = sorted(predictions.items(), reverse=True, key=operator.itemgetter(1))

            # append to XML <unigram> element
            item_prediction = ET.SubElement(item_unigram, 'prediction')
            item_prediction.text = maximum.split(" ")[1]
            item_correct = ET.SubElement(item_unigram, 'correct')
            item_confidence = ET.SubElement(item_unigram, 'confidence')
            item_confidence.text = str(predictions[maximum]/sum(predictions.values()))
            item_candidates = ET.SubElement(item_unigram, 'candidates')
            n = 0
            while n < 5 and n < len(sorted_predictions):
                item_candidate = ET.SubElement(item_candidates, 'candidate')
                item_name = ET.SubElement(item_candidate, 'name')
                item_name.text = sorted_predictions[n][0].split(" ")[1]
                item_conf = ET.SubElement(item_candidate, 'conf')
                item_conf.text = str(sorted_predictions[n][1]/sum(predictions.values()))
                n += 1

            # delta experiment: more precise confidence of predictions
            maximum_confidence = (predictions[maximum]-delta[wordlength])/sum(predictions.values())
            probabilities.append(maximum_confidence)
            if maximum_confidence>0.3:
                confidentcounter1+=1
                freq_average3+=(predictions[maximum]-delta[wordlength])/sum(predictions.values())
            if maximum_confidence>.4:
                confidentcounter2+=1
                freq_average4+=(predictions[maximum]-delta[wordlength])/sum(predictions.values())
            if maximum_confidence>.5:
                confidentcounter3+=1
                freq_average5+=(predictions[maximum]-delta[wordlength])/sum(predictions.values())
            if maximum_confidence>.6:
                confidentcounter4+=1
                freq_average6+=(predictions[maximum]-delta[wordlength])/sum(predictions.values())
            if maximum_confidence>.7:
                confidentcounter5+=1
                freq_average7+=(predictions[maximum]-delta[wordlength])/sum(predictions.values())
            if maximum_confidence>.8:
                confidentcounter6+=1
                freq_average8+=(predictions[maximum]-delta[wordlength])/sum(predictions.values())
            if maximum_confidence>.9:
                confidentcounter7+=1
                freq_average9+=(predictions[maximum]-delta[wordlength])/sum(predictions.values())
            if maximum_confidence>.95:
                confidentcounter8+=1
                freq_average95+=(predictions[maximum]-delta[wordlength])/sum(predictions.values())
            if maximum_confidence==1:
                One_hundred_percent+=1
            if maximum_confidence>.85:
                confidentcounter85+=1
                freq_average85+=(predictions[maximum]-delta[wordlength])/sum(predictions.values())
            if maximum_confidence>.75:
                confidentcounter75+=1

                temp4.append(maximum.split(" ")[1])
                changecheck=1
                freq_average75 += maximum_confidence

                if maximum.split(" ")[1] == actual_word_list[idx]:
                    print ("#yes", "Test Word:", maximum.split(" ")[0], ", Prediction:", maximum.split(" ")[1], ", Actual Word:", actual_word_list[idx])
                    confidence_correct += 1
                else:
                    print ("no", "Test Word:", maximum.split(" ")[0], ", Prediction:", maximum.split(" ")[1], ", Actual Word:", actual_word_list[idx])
                confidence_total_counter += 1

            if (predictions[maximum]-delta[wordlength])/sum(predictions.values())>.65:
                confidentcounter65+=1
                freq_average65+=(predictions[maximum]-delta[wordlength])/sum(predictions.values())
            if (predictions[maximum]-delta[wordlength])/sum(predictions.values())>.55:
                confidentcounter55+=1
                freq_average55+=(predictions[maximum]-delta[wordlength])/sum(predictions.values())

        if maximum.split(" ")[1] == actual_word_list[idx]:
            print("#yes", "Test Word:", maximum.split(" ")[0], ", Prediction:", maximum.split(" ")[1], ", Actual Word:",
                  actual_word_list[idx])
            correct += 1
            item_correct.text = "true"

        else:
            print("no", "Test Word:", maximum.split(" ")[0], ", Prediction:", maximum.split(" ")[1], ", Actual Word:",
                  actual_word_list[idx])
            item_correct.text = "false"
        idx += 1

        if checker==0:
            probabilities.append(0)
            first_dot = False
            first_check = 0
            remove_dot = False
            two_dots = False
            for rand in i:
                if first_check == 1 and rand == "●":
                    remove_dot = True
                    first_check = 2
                    break
                if rand == "●" and i[0] == "●":
                    first_dot = True
                    first_check = 1
            if first_check == 1 and first_dot == True:
                i = i.replace("●","●●",1)

        if checker==0:
            dictt={}
            dictt[i]=0
            average_dots.append(dictt)
            total_average_dots.append(dictt)
        else:
            hold_dict={}
            max_size=0
            saved=""
            prediction_size=sum(predictions.values())
            total_sum=sum(predictions.values())
            while(prediction_size!=0 and max_size<5):
                saved=maximum.split(" ")[1].split(" "[0])[0]
                hold_dict[saved]=predictions[maximum]/total_sum
                predictions_size = prediction_size-1
                max_size+=1
                predictions[maximum]=.5
                maximum=max(predictions, key=predictions.get)
            total_average_dots.append(hold_dict)

        if changecheck==0:
            temp4.append(i)
            hold_dict={}
            max_size=0
            saved=""
            if checker!=0:
                prediction_size=sum(predictions.values())
                total_sum=sum(predictions.values())
                while(prediction_size!=0 and max_size<5):
                    saved=maximum.split(" ")[1].split(" "[0])[0]
                    hold_dict[saved]=predictions[maximum]/total_sum
                    predictions_size = prediction_size-1
                    max_size+=1
                    predictions[maximum]=.5
                    maximum=max(predictions, key=predictions.get)
                average_dots.append(hold_dict)

    print ("total_count = ", total_counter, ", correct = ", correct)
    print ("accuracy = ", correct / total_counter)

    print("confidence_total_count = ", confidence_total_counter, ", confidence_correct = ", confidence_correct)
    print("accuracy = ", confidence_correct / confidence_total_counter)

    dot_file=open('freq_data/predicted', 'wb')
    pickle.dump(temp4, dot_file)
    dot_file.close()

    second_dot_file=open('freq_data/uni_second_step_avgs', 'wb')
    pickle.dump(average_dots, second_dot_file)
    second_dot_file.close()

    prob=open('freq_data/probabilities', 'wb')
    pickle.dump(probabilities, prob)
    prob.close()

    supp_probs=open('freq_data/sup_probs', 'wb')
    pickle.dump(total_average_dots, supp_probs)
    supp_probs.close()

    xmlstr = '\n'.join([line for line in minidom.parseString(ET.tostring(doc)).toprettyxml(indent="    ").split('\n') if line.strip()])
    with open(XML_output, "w", newline='', encoding='UTF-8') as f:
        f.write(xmlstr)

test_word_model(dot_words, deltas,wlist1)
