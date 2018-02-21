import glob
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import codecs
import sys

import math


print (" Number of arguments: ", len(sys.argv))
print (" Training path : ", sys.argv[1])
print (" Test path : ", sys.argv[2])

print("\n")

stopWords = set(stopwords.words('english'))
file_count=0
vocabulary={}
uniqueVocabWords=set()
for name in glob.glob(sys.argv[1]+"/*/*"):
    file_count+=1
    file = codecs.open(name, "r",encoding='utf-8', errors='ignore')
    data = file.read()
    replaced_data = re.sub('[^a-zA-Z\n]', ' ', data)
    words = word_tokenize(replaced_data)
    wordsFiltered = [w for w in words if w not in stopWords]
    for x in wordsFiltered:
        uniqueVocabWords.add(x)

print("Total files: ",file_count)
#print(uniqueVocabWords)
total_words=len(uniqueVocabWords)
print("Total words in the files: ",total_words)
hockey_filecount=0
wordcount1 = Counter()
wordcount2 = Counter()
wordcount3 = Counter()
wordcount4 = Counter()
wordcount5 = Counter()

for name in glob.glob(sys.argv[1]+"/rec.sport.hockey/*"):
      hockey_filecount+=1
      file = codecs.open(name, "r",encoding='utf-8', errors='ignore')
      data = file.read()
      replaced_data = re.sub('[^a-zA-Z\n]', ' ',data)
      words = word_tokenize(replaced_data)
      wordsFiltered = [w for w in words if w not in stopWords]
      wordcount1.update(wordsFiltered)

classA_words=len(wordcount1)
print("Words in class A: Hockey: ",classA_words)

vocab_dict={}
for word in uniqueVocabWords:
    if word in wordcount1.keys():
       prob_a = (wordcount1[word] + 1) / (classA_words + total_words)
    else:
       prob_a=(0 + 1) / (classA_words + total_words)
    prob_a_dict={'prob_a':prob_a}
    vocab_dict[word]=prob_a_dict

#finding prior of class A
prior_A=hockey_filecount/file_count

motorcycles_filecount=0
for name in glob.glob(sys.argv[1]+"/rec.motorcycles/*"):
      motorcycles_filecount+=1
      file = codecs.open(name, "r",encoding='utf-8', errors='ignore')
      data = file.read()
      replaced_data = re.sub('[^a-zA-Z\n]', ' ',data)
      words = word_tokenize(replaced_data)
      wordsFiltered = [w for w in words if w not in stopWords]
      wordcount2.update(wordsFiltered)

classB_words=len(wordcount2)
print("Words in class B: motercycle: ",classB_words)
#finding prior of class B
prior_B=motorcycles_filecount/file_count

for word in uniqueVocabWords:
    if word in wordcount2.keys():
       prob_b = (wordcount2[word] + 1) / (classB_words + total_words)
    else:
       prob_b=(0 + 1) / (classB_words + total_words)

    vocab_dict[word]['prob_b']=prob_b

#print(vocab_dict)


#### C ####

politics_filecount=0
for name in glob.glob(sys.argv[1]+"/talk.politics.guns/*"):
      politics_filecount+=1
      file = codecs.open(name, "r",encoding='utf-8', errors='ignore')
      data = file.read()
      replaced_data = re.sub('[^a-zA-Z\n]', ' ',data)
      words = word_tokenize(replaced_data)
      wordsFiltered = [w for w in words if w not in stopWords]
      wordcount3.update(wordsFiltered)

classC_words=len(wordcount3)
print("Words in class C: Politics: ",classC_words)
#finding prior of class B
prior_C=politics_filecount/file_count

for word in uniqueVocabWords:
    if word in wordcount3.keys():
       prob_c = (wordcount3[word] + 1) / (classC_words + total_words)
    else:
       prob_c=(0 + 1) / (classC_words + total_words)

    vocab_dict[word]['prob_c']=prob_c


#### D ####


space_filecount=0
for name in glob.glob(sys.argv[1]+"/sci.space/*"):
      space_filecount+=1
      file = codecs.open(name, "r",encoding='utf-8', errors='ignore')
      data = file.read()
      replaced_data = re.sub('[^a-zA-Z\n]', ' ',data)
      words = word_tokenize(replaced_data)
      wordsFiltered = [w for w in words if w not in stopWords]
      wordcount4.update(wordsFiltered)

classD_words=len(wordcount4)
print("Words in class D: space: ",classD_words)
#finding prior of class B
prior_D=space_filecount/file_count

for word in uniqueVocabWords:
    if word in wordcount4.keys():
       prob_d = (wordcount4[word] + 1) / (classD_words + total_words)
    else:
       prob_d=(0 + 1) / (classD_words + total_words)

    vocab_dict[word]['prob_d']=prob_d


#### E ####



religion_filecount=0
for name in glob.glob(sys.argv[1]+"/soc.religion.christian/*"):
      religion_filecount+=1
      file = codecs.open(name, "r",encoding='utf-8', errors='ignore')
      data = file.read()
      replaced_data = re.sub('[^a-zA-Z\n]', ' ',data)
      words = word_tokenize(replaced_data)
      wordsFiltered = [w for w in words if w not in stopWords]
      wordcount5.update(wordsFiltered)

classE_words=len(wordcount5)
print("Words in class E: religion: ",classE_words)
#finding prior of class B
prior_E=religion_filecount/file_count

for word in uniqueVocabWords:
    if word in wordcount5.keys():
       prob_e = (wordcount5[word] + 1) / (classE_words + total_words)
    else:
       prob_e=(0 + 1) / (classE_words + total_words)

    vocab_dict[word]['prob_e']=prob_e


#### * ####

correct_prediction_hockey=0
hockey_test_filecount=0
#testing
for name in glob.glob(sys.argv[2]+"/rec.sport.hockey/*"):
        hockey_test_filecount+=1
        file = codecs.open(name, "r",encoding='utf-8', errors='ignore')
        data = file.read()
        replaced_data = re.sub('[^a-zA-Z\n]', ' ', data)
        words = word_tokenize(replaced_data)
        wordsFiltered = [w for w in words if w not in stopWords]
        classA = math.log(prior_A)
        classB=math.log(prior_B)
        classC=math.log(prior_C)
        classD=math.log(prior_D)
        classE=math.log(prior_E)

        for eachword in wordsFiltered:
            if eachword in vocab_dict.keys():
               classA+=math.log(vocab_dict[eachword]['prob_a'])
               classB+=math.log(vocab_dict[eachword]['prob_b'])
               classC+=math.log(vocab_dict[eachword]['prob_c'])
               classD+=math.log(vocab_dict[eachword]['prob_d'])
               classE+=math.log(vocab_dict[eachword]['prob_e'])


        max_class=max(classA,classB,classC,classD,classE)
        if max_class==classA:
            correct_prediction_hockey+=1

accuracy=correct_prediction_hockey/hockey_test_filecount
#print("\nAccuracy for hockey class:",accuracy)

#### * ####

correct_prediction_motercycle=0
motercycle_test_filecount=0
#testing
for name in glob.glob(sys.argv[2]+"/rec.motorcycles/*"):
        motercycle_test_filecount+=1
        file = codecs.open(name, "r",encoding='utf-8', errors='ignore')
        data = file.read()
        replaced_data = re.sub('[^a-zA-Z\n]', ' ', data)
        words = word_tokenize(replaced_data)
        wordsFiltered = [w for w in words if w not in stopWords]
        classA = math.log(prior_A)
        classB=math.log(prior_B)
        classC=math.log(prior_C)
        classD=math.log(prior_D)
        classE=math.log(prior_E)

        for eachword in wordsFiltered:
            if eachword in vocab_dict.keys():
               classA+=math.log(vocab_dict[eachword]['prob_a'])
               classB+=math.log(vocab_dict[eachword]['prob_b'])
               classC+=math.log(vocab_dict[eachword]['prob_c'])
               classD+=math.log(vocab_dict[eachword]['prob_d'])
               classE+=math.log(vocab_dict[eachword]['prob_e'])


        max_class=max(classA,classB,classC,classD,classE)
        if max_class==classB:
            correct_prediction_motercycle+=1

accuracy=correct_prediction_motercycle/motercycle_test_filecount
#print("\nAccuracy for motercycle class:",accuracy)

#### * ####

correct_prediction_politics=0
politics_test_filecount=0
#testing
for name in glob.glob(sys.argv[2]+"/talk.politics.guns/*"):
        politics_test_filecount+=1
        file = codecs.open(name, "r",encoding='utf-8', errors='ignore')
        data = file.read()
        replaced_data = re.sub('[^a-zA-Z\n]', ' ', data)
        words = word_tokenize(replaced_data)
        wordsFiltered = [w for w in words if w not in stopWords]
        classA = math.log(prior_A)
        classB=math.log(prior_B)
        classC=math.log(prior_C)
        classD=math.log(prior_D)
        classE=math.log(prior_E)

        for eachword in wordsFiltered:
            if eachword in vocab_dict.keys():
               classA+=math.log(vocab_dict[eachword]['prob_a'])
               classB+=math.log(vocab_dict[eachword]['prob_b'])
               classC+=math.log(vocab_dict[eachword]['prob_c'])
               classD+=math.log(vocab_dict[eachword]['prob_d'])
               classE+=math.log(vocab_dict[eachword]['prob_e'])


        max_class=max(classA,classB,classC,classD,classE)
        if max_class==classC:
            correct_prediction_politics+=1

accuracy=correct_prediction_politics/politics_test_filecount
#print("\nAccuracy for politics class:",accuracy)

#### * ####

correct_prediction_space=0
space_test_filecount=0
#testing
for name in glob.glob(sys.argv[2]+"/sci.space/*"):
        space_test_filecount+=1
        file = codecs.open(name, "r",encoding='utf-8', errors='ignore')
        data = file.read()
        replaced_data = re.sub('[^a-zA-Z\n]', ' ', data)
        words = word_tokenize(replaced_data)
        wordsFiltered = [w for w in words if w not in stopWords]
        classA = math.log(prior_A)
        classB=math.log(prior_B)
        classC=math.log(prior_C)
        classD=math.log(prior_D)
        classE=math.log(prior_E)

        for eachword in wordsFiltered:
            if eachword in vocab_dict.keys():
               classA+=math.log(vocab_dict[eachword]['prob_a'])
               classB+=math.log(vocab_dict[eachword]['prob_b'])
               classC+=math.log(vocab_dict[eachword]['prob_c'])
               classD+=math.log(vocab_dict[eachword]['prob_d'])
               classE+=math.log(vocab_dict[eachword]['prob_e'])


        max_class=max(classA,classB,classC,classD,classE)
        if max_class==classD:
            correct_prediction_space+=1

accuracy=correct_prediction_space/space_test_filecount
#print("\nAccuracy for space class:",accuracy)

#### * ####

correct_prediction_religion=0
christian_test_filecount=0
#testing
for name in glob.glob(sys.argv[2]+"/soc.religion.christian/*"):
        christian_test_filecount+=1
        file = codecs.open(name, "r",encoding='utf-8', errors='ignore')
        data = file.read()
        replaced_data = re.sub('[^a-zA-Z\n]', ' ', data)
        words = word_tokenize(replaced_data)
        wordsFiltered = [w for w in words if w not in stopWords]
        classA = math.log(prior_A)
        classB=math.log(prior_B)
        classC=math.log(prior_C)
        classD=math.log(prior_D)
        classE=math.log(prior_E)

        for eachword in wordsFiltered:
            if eachword in vocab_dict.keys():
               classA+=math.log(vocab_dict[eachword]['prob_a'])
               classB+=math.log(vocab_dict[eachword]['prob_b'])
               classC+=math.log(vocab_dict[eachword]['prob_c'])
               classD+=math.log(vocab_dict[eachword]['prob_d'])
               classE+=math.log(vocab_dict[eachword]['prob_e'])


        max_class=max(classA,classB,classC,classD,classE)
        if max_class==classE:
            correct_prediction_religion+=1

accuracy=correct_prediction_religion/christian_test_filecount
#print("\nAccuracy for christian class:",accuracy)

total_filecount_test = christian_test_filecount+space_test_filecount+politics_test_filecount+motercycle_test_filecount+hockey_test_filecount
total_correct_count=correct_prediction_religion+correct_prediction_space+correct_prediction_politics+correct_prediction_motercycle+correct_prediction_hockey

final_accuracy=total_correct_count/total_filecount_test

print("\nfinal accuracy: ",final_accuracy*100 ,"%")