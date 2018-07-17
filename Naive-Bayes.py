# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 14:19:08 2018

@author: babit
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer

import scipy
import string

train_pos = pd.read_csv('trainPos.txt', error_bad_lines=False, header=None)
train_pos['sentiment'] = 'Positive'
train_pos.columns = ['text','sentiment']
train_pos = train_pos.head(10000)


tweets=[]
stopwords_set = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")
for index, row in train_pos.iterrows():
    words_filtered = [e.lower() for e in row.text.split() if len(e) >= 3]
    words_cleaned= [word for word in words_filtered
        if 'http' not in word
        and not word.startswith('@')
        and not word.startswith('#')
        and word.translate(str.maketrans('','',string.punctuation))]
    words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
    tweets.append((words_without_stopwords,row.sentiment))
    
poswords = []
for (words, sentiment) in tweets:
    poswords.extend(words)

poswords = [stemmer.stem(word) for word in poswords]

    
from collections import Counter
counts = dict(Counter(poswords))

posdf = pd.DataFrame.from_dict(counts, orient='index', dtype=None)
posdf['text'] = posdf.index
posdf['condPrb'] = (posdf[0] / len(posdf))

poswordsdf = pd.DataFrame(poswords, dtype=None)
poswordsdf['label'] = 'Positive'

################

train_neg = pd.read_csv('trainNeg.txt', error_bad_lines=False, header=None)
train_neg['sentiment'] = 'Negative'
train_neg.columns = ['text','sentiment']
train_neg = train_neg.head(10000)


tweets=[]
for index, row in train_neg.iterrows():
    words_filtered = [e.lower() for e in row.text.split() if len(e) >= 3]
    words_cleaned= [word for word in words_filtered
        if 'http' not in word
        and not word.startswith('@')
        and not word.startswith('#')
        and word.translate(str.maketrans('','',string.punctuation))]
    words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
    tweets.append((words_without_stopwords,row.sentiment))

negwords = []
for (words, sentiment) in tweets:
    negwords.extend(words)
    
negwords = [stemmer.stem(word) for word in negwords]
    
from collections import Counter
counts = dict(Counter(negwords))

negdf = pd.DataFrame.from_dict(counts, orient='index', dtype=None)
negdf['text'] = negdf.index
total = len(negdf)+len(posdf)
negdf['condPrb'] = ((negdf[0]+1) / (len(negdf) + total) ) 
negdf['label'] = 'Negative'

negwordsdf = pd.DataFrame(negwords, dtype=None)
negwordsdf['label'] = 'Negative'

##############################


frames = [poswordsdf, negwordsdf]
trainingsetwords = pd.concat(frames)
trainingsetwords.columns = ['features','labels']

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()

X_train_dtm = vect.fit_transform(trainingsetwords['features'])
X_train_dtm

nb = MultinomialNB()
nb.fit(X_train_dtm, trainingsetwords['labels'])

test_pos = pd.read_csv('testPos.txt', error_bad_lines=False, header=None)
test_pos['sentiment'] = 'Positive'
test_pos.columns = ['text','sentiment']

test_neg = pd.read_csv('testNeg.txt', error_bad_lines=False, header=None)
test_neg['sentiment'] = 'Negative'
test_neg.columns = ['text','sentiment']
frames = [test_pos, test_neg]
testdf = pd.concat(frames)
testdf.columns = ['text','sentiment']

X_test_dtm = vect.transform(testdf['text'])
X_test_dtm

y_pred_class = nb.predict(X_test_dtm)

from sklearn import metrics
metrics.accuracy_score(testdf['sentiment'], y_pred_class)


metrics.confusion_matrix(testdf['sentiment'], y_pred_class)

##########
X_train_dtm = scipy.sparse.csr_matrix.todense(X_train_dtm)
X_test_dtm = scipy.sparse.csr_matrix.todense(X_test_dtm)

nb = GaussianNB()
nb.fit(X_train_dtm, trainingsetwords['labels'])

y_pred_class = nb.predict(X_test_dtm)

from sklearn import metrics
metrics.accuracy_score(testdf['sentiment'], y_pred_class)


metrics.confusion_matrix(testdf['sentiment'], y_pred_class)
