#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 11:20:08 2018

@author: gabrielefrattaroli - R00144141
"""

#################################################
####### Stage	1–Vocabulary	Composition and Word Frequency Calculations
#################################################

import pandas as pd 
import numpy as np

import string

import nltk

#nltk.download('all')

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer #### Added after research
from nltk.stem.porter import PorterStemmer #### Added after research
from nltk.stem.lancaster import LancasterStemmer #### Added after research
from nltk.stem import WordNetLemmatizer #### Added after research




train_pos = pd.read_csv('trainPos.txt', error_bad_lines=False, header=None)
train_pos['sentiment'] = 'Positive'
train_pos.columns = ['text','sentiment']

stopwords_set = set(stopwords.words("english")) #### Added after research
snowball_stemmer = SnowballStemmer("english")  #### Added after research
porter_stemmer = PorterStemmer() #### Added after research
lancaster_stemmer = LancasterStemmer() #### Added after research
wordnet_lemmatizer = WordNetLemmatizer() #### Added after research
stopwords_set = set(stopwords.words("english")) #### Added after research


pos_tweets=[]
for index, row in train_pos.iterrows():
    words_filtered = [e.lower() for e in row.text.split()]  
    words_cleaned= [word for word in words_filtered
        if 'http' not in word
        and not word.startswith('@')
        and not word.startswith('#')
        and word.translate(str.maketrans('','',string.punctuation))]
    words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set] #### Added after research
    pos_tweets.append((words_without_stopwords,row.sentiment))

poswords = []
for (words, sentiment) in pos_tweets:
    poswords.extend(words)
    
#poswords = [snowball_stemmer.stem(word) for word in poswords]   #### Added after research USE for Snowball
#poswords = [porter_stemmer.stem(word) for word in poswords]   #### Added after research USE for Porter
#poswords = [lancaster_stemmer.stem(word) for word in poswords]   #### Added after research USE for Lancaster
poswords = [wordnet_lemmatizer.lemmatize(word) for word in poswords]   #### Added after research USE for Lemmatizer
        
from collections import Counter
poscounts = dict(Counter(poswords))

posdf = pd.DataFrame.from_dict(poscounts, orient='index', dtype=None)
posdf['text'] = posdf.index
posdf.columns = ['frequency','text']

train_neg = pd.read_csv('trainNeg.txt', error_bad_lines=False, header=None)
train_neg['sentiment'] = 'Negative'
train_neg.columns = ['text','sentiment']


neg_tweets=[]
for index, row in train_neg.iterrows():
    words_filtered = [e.lower() for e in row.text.split()]  
    words_cleaned= [word for word in words_filtered
        if 'http' not in word
        and not word.startswith('@')
        and not word.startswith('#')
        and word.translate(str.maketrans('','',string.punctuation))]
    words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set] #### Added after research
    neg_tweets.append((words_cleaned,row.sentiment))


negwords = []
for (words, sentiment) in neg_tweets:
    negwords.extend(words)
    
#negwords = [snowball_stemmer.stem(word) for word in negwords]   #### Added after research USE for Snowball
#negwords = [porter_stemmer.stem(word) for word in negwords]   #### Added after research USE for Porter
#negwords = [lancaster_stemmer.stem(word) for word in negwords]   #### Added after research USE for Lancaster
negwords = [wordnet_lemmatizer.lemmatize(word) for word in negwords]   #### Added after research USE for Lemmatizer    
    
from collections import Counter
negcounts = dict(Counter(negwords))

negdf = pd.DataFrame.from_dict(negcounts, orient='index', dtype=None)
negdf['text'] = negdf.index
negdf.columns = ['frequency','text']

######### This is the frequency for each words in both negative and positive sets

print (posdf)
print (negdf)

#################################################
####### Stage	2 – Calculating Word Probability Calculations
#################################################

total = len(negdf)+len(posdf)

posdf['condPrb'] = ((posdf['frequency']+1) / (len(posdf) + total) ) 
posdf['label'] = 'Positive'

negdf['condPrb'] = ((negdf['frequency']+1) / (len(negdf) + total) ) 
negdf['label'] = 'Negative'


posdfdict = zip(posdf['text'],posdf['condPrb'])
posdfdict = dict(posdfdict)

negdfdict = zip(negdf['text'],negdf['condPrb'])
negdfdict = dict(negdfdict)


#################################################
####### Stage	3 – Classifying Unseen Tweets and Performing Basic	 Evaluation
#################################################

#tweet = 'having. @test #test dogs'  #### tweet test to see if lemmatization / stemming works
#
#valuestorepos = []
#valuestoreneg = []
#words_filtered = [e.lower() for e in tweet.split()]  
#words_cleaned= [word for word in words_filtered
#    if 'http' not in word
#    and not word.startswith('@')
#    and not word.startswith('#')
#    and word.translate(str.maketrans('','',string.punctuation))]
#words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set] #### Added after research
#words_advanced_cleaning = [wordnet_lemmatizer.lemmatize(word) for word in words_without_stopwords]
#for x in range(len(words_advanced_cleaning)):
#    a = words_advanced_cleaning[x]
#    valuestorepos.append(posdfdict.get(a, 0.000001))
#    valuestoreneg.append(negdfdict.get(a, 0.000001))
#PcPos = np.log(len(pos_tweets)/ (len(pos_tweets)+len(neg_tweets)))
#PcNeg = np.log(len(neg_tweets)/ (len(pos_tweets)+len(neg_tweets)))
#totalpos = (PcPos + sum(np.log(valuestorepos)))
#totalneg = (PcNeg + sum(np.log(valuestoreneg)))
#if totalpos>totalneg:
#    print ('positive')
#else:
#    print ('negative')



test_pos = pd.read_csv('testPos.txt', error_bad_lines=False, header=None)
test_pos['sentiment'] = 'Positive'
test_pos.columns = ['text','sentiment']

test_neg = pd.read_csv('testNeg.txt', error_bad_lines=False, header=None)
test_neg['sentiment'] = 'Negative'
test_neg.columns = ['text','sentiment']
frames = [test_pos, test_neg]
testdf = pd.concat(frames)
testdf.columns = ['text','sentiment']

Predictions = []
for index, row in testdf.iterrows():
    words_filtered = [e.lower() for e in row.text.split()]  
    words_cleaned= [word for word in words_filtered
        if 'http' not in word
        and not word.startswith('@')
        and not word.startswith('#')
        and word.translate(str.maketrans('','',string.punctuation))]
    words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set] #### Added after research
    words_advanced_cleaning = [wordnet_lemmatizer.lemmatize(word) for word in words_without_stopwords]
    valuestorepos = []
    valuestoreneg = []
    for x in range(len(words_advanced_cleaning)):
        a = words_advanced_cleaning[x]
        valuestorepos.append(posdfdict.get(a, 0.000001))
        valuestoreneg.append(negdfdict.get(a, 0.000001))
    PcPos = np.log(len(pos_tweets)/ (len(pos_tweets)+len(neg_tweets)))
    PcNeg = np.log(len(neg_tweets)/ (len(pos_tweets)+len(neg_tweets)))
    totalpos = (PcPos + sum(np.log(valuestorepos)))
    totalneg = (PcNeg + sum(np.log(valuestoreneg)))
    Psentiment = 0
    if totalpos>totalneg:
        Psentiment = 'Positive'
    else:
        Psentiment = 'Negative'
    Predictions.append((words_advanced_cleaning,Psentiment))
    
PredictionDF = pd.DataFrame(list(Predictions), columns=['text','sentiment'])

######## Imporing metrics for evaluation
from sklearn import metrics
print (metrics.accuracy_score(testdf['sentiment'], PredictionDF['sentiment'])*100, 'percent accuracy')

print (metrics.confusion_matrix(testdf['sentiment'], PredictionDF['sentiment']))
