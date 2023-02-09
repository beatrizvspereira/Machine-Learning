#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Alexandre Silva 90004 e Beatriz Pereira 90029

Usage: Lab AA Bayes classifiers


"""
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.metrics import accuracy_score 
from sklearn.feature_extraction.text import CountVectorizer 

# --------- Pergunta 3. LANGUAGE RECOGNIZER --------- 

### -- 3.2.1 Training --

##load data
PT = pd.read_csv('data_lab4/pt_trigram_count.tsv', sep='\t', header=None)
EN = pd.read_csv('data_lab4/en_trigram_count.tsv', sep='\t', header=None)
ES = pd.read_csv('data_lab4/es_trigram_count.tsv', sep='\t', header=None)
FR = pd.read_csv('data_lab4/fr_trigram_count.tsv', sep='\t', header=None)

## shape, head 
print('PT\n shape-', PT.shape, '\nhead', PT.head(1))
print('ES\n shape-', ES.shape, '\nhead', ES.head(1))
print('EN\n shape-', EN.shape, '\nhead', EN.head(1))
print('FR\n shape-', FR.shape, '\nhead', FR.head(1))

## X_train, y_train
X_train = np.zeros((4, PT.shape[0]))

#vetores auxiliares
PT_aux = PT.to_numpy()
EN_aux = EN.to_numpy()
ES_aux = ES.to_numpy()
FR_aux = FR.to_numpy()

trigrams = np.transpose(PT_aux[:,1]); # read all trigrams

X_train[0,:] = np.transpose(PT_aux[:,2])
X_train[1,:] = np.transpose(EN_aux[:,2])
X_train[2,:] = np.transpose(ES_aux[:,2])
X_train[3,:] = np.transpose(FR_aux[:,2])

y_train = ['pt','en','es','fr']

# Multinomial Naive Bayes model
NB_model = MNB(alpha=1.0, fit_prior = False) #alpha=1 para laplace smoothing, equal priors
# Fitting the model
NB_model.fit(X_train, y_train)

### -- 3.2.1 Testing --

# predictions and accurancy of the model
y_prediction = NB_model.predict(X_train)
NB_accuracy = accuracy_score(y_prediction, y_train)

# sentences matrix
sentences = [
    'Que fácil es comer peras.',
    'Que fácil é comer peras.',
    'Today is a great day for sightseeing.',
    'Je vais au cinéma demain soir.',
    'Ana es inteligente y simpática.',
    'Tu vais à escola hoje.',
    'Tu não vais à escola hoje.']

# vectorizer
vectorizer = CountVectorizer(vocabulary=trigrams, analyzer='char', ngram_range=(3,3))

# data trigram counts for the given sentences: X_test matrix
X_test = vectorizer.fit_transform(sentences)
X_test_count = X_test.toarray()
y_test =  ['es','pt','en','fr','es','pt','pt']

# predictions
y_test_prediction = NB_model.predict(X_test)
print('\nRecognized Language:\n',y_test_prediction)
print('Accuracy score: ',  accuracy_score(y_test, y_test_prediction), '\n')

# Classification margin
score = NB_model.predict_proba(X_test) 

score_sort = np.sort(score)
print('\nScore:\n', score_sort[:,3])

classific_margin = score_sort[:,3]-score_sort[:,2]
print('\nClassification Margin:', classific_margin)
