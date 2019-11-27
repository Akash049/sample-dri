from django.shortcuts import render
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from django.shortcuts import redirect, render
from django.views.decorators.csrf import csrf_exempt
from django.core import serializers as django_serializers
from django.db.models import Q
from django.utils.encoding import smart_text

from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
import pandas as pd

import tensorflow as tf
import os
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Input, GRU, Dense, Concatenate, TimeDistributed
from attention_keras_thushv import AttentionLayer
from tensorflow.python.keras.models import Model

# Fow language detection
from langdetect import detect

import tensorflow.keras as keras
from tensorflow.python.keras.utils import to_categorical
import numpy as np
import os, sys
import pickle
import json
import time

from datetime import timedelta
from datetime import datetime

import requests
import re


#Loading the model
# bd_rnn_model = load_model('Bd_RNN_Model_v3.h5')

# def tokenize(x):
#     """
#     Tokenize x
#     :param x: List of sentences/strings to be tokenized
#     :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
#     """
#     # TODO: Implement
#     tokenizer = Tokenizer()
#     tokenizer.fit_on_texts(x)
#     return tokenizer

# def pad(x, length=None):
#     """
#     Pad x
#     :param x: List of sequences.
#     :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
#     :return: Padded numpy array of sequences
#     """
#     # TODO: Implement
#     return pad_sequences(x, maxlen=50, padding='post')

# def logits_to_text(logits, tokenizer):
#     """
#     Turn logits from a neural network into text using the tokenizer
#     :param logits: Logits from a neural network
#     :param tokenizer: Keras Tokenizer fit on the labels
#     :return: String that represents the text of the logits
#     """
#     index_to_words = {id: word for word, id in tokenizer.word_index.items()}
#     index_to_words[0] = '<PAD>'

#     return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

# # loading
# with open('english_tokenizer_2.pickle', 'rb') as handle:
#     en_tokenizer = pickle.load(handle)
# with open('spanish_tokenizer_2.pickle', 'rb') as handle:
#     sp_tokenizer = pickle.load(handle)
#----------------------------


#Loading the model
bd_rnn_model = load_model('Bd_RNN_Model_v3.h5')

#reading data
spa = pd.read_csv('spa.txt', sep='\n', header = None)

#Preprocessing
spa.columns=['Content']
text = spa['Content'].apply(lambda x: x[: x.find('CC-BY 2.0')])
text = text.str.strip()
spa['English'] = text.apply(lambda x: x.split('\t')[0])
spa['Spanish'] = text.apply(lambda x: x.split('\t')[1])
spa = spa[['English','Spanish']]
spa['English'] = spa['English'].str.lower()
spa['Spanish'] = spa['Spanish'].str.lower()

#Removing dots from starting and ending and numbers
spa['English'] = spa['English'].apply(lambda x: re.sub('[<>;+:!¡/\|?¿,.0-9@#$%^&*"]+' , '' , x))
spa['Spanish'] = spa['Spanish'].apply(lambda x: re.sub('[<>;+:!¡/\|?¿,.0-9@#$%^&*"]+' , '' , x))

#replacing hypen with a space
spa['English'] = spa['English'].apply(lambda x: re.sub('[-]+' , ' ' , x))
spa['Spanish'] = spa['Spanish'].apply(lambda x: re.sub('[-]+' , ' ' , x))
del(text)

#Shuffling dataset
spa = spa.sample(frac = 1)

def tokenize(x):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    # TODO: Implement
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)
    return tokenizer

def pad(x, length=None):
    """
    Pad x
    :param x: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    :return: Padded numpy array of sequences
    """
    # TODO: Implement
    return pad_sequences(x, maxlen=50, padding='post')


def logits_to_text(logits, tokenizer):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])
    
en_tokenizer, sp_tokenizer = Tokenizer(), Tokenizer()
en_tokenizer.fit_on_texts(spa['English'].tolist())
sp_tokenizer.fit_on_texts(spa['Spanish'].tolist())

# Create your views here.
class SpaToEng(APIView):

    def  post(self, request):
        try:
            entered_text = request.POST.get('spanish','').lower()
            entered_text = entered_text.lower()
            entered_text = re.sub('[<>;+:!¡/\|?¿,.0-9@#$%^&*"]+' , '' , entered_text)
            entered_text = re.sub('[-]+' , ' ' , entered_text)

            eng_seq = list()
            eng_seq.append(entered_text)

            #Converting into vectors
            input_en = sp_tokenizer.texts_to_sequences(eng_seq)
            #Putting padding
            input_en = pad(input_en)
            #Formatting shape
            input_en = input_en.reshape(1,50,1)

            input_en = input_en.astype(float)

            #Translated Seq
            translated_seq = logits_to_text(bd_rnn_model.predict(input_en)[0], en_tokenizer)
            #Replacing any <PAD> with spaces
            translated_seq = translated_seq.replace('<PAD>', '')
            print(translated_seq)

            return Response({'status':1, 'spanish':translated_seq})
        except Exception as e:
            return Response({'status':0,'error':repr(e)})

# Create your views here.
class EngToSpanish(APIView):

    def  post(self, request):
        try:
            entered_text = request.POST.get('english','').lower()
            entered_text = entered_text.lower()
            entered_text = re.sub('[<>;+:!¡/\|?¿,.0-9@#$%^&*"]+' , '' , entered_text)
            entered_text = re.sub('[-]+' , ' ' , entered_text)

            eng_seq = list()
            eng_seq.append(entered_text)

            #Converting into vectors
            input_en = en_tokenizer.texts_to_sequences(eng_seq)
            #Putting padding
            input_en = pad(input_en)
            #Formatting shape
            input_en = input_en.reshape(1,50,1)

            input_en = input_en.astype(float)

            #predictions = bd_rnn_model.predict(input_en)

            #print(predictions.shape)

            #Translated Seq
            translated_seq = logits_to_text(bd_rnn_model.predict(input_en)[0], sp_tokenizer)
            #Replacing any <PAD> with spaces
            print(translated_seq)
            translated_seq = translated_seq.replace('<PAD>', '')
            print(translated_seq)

            return Response({'status':1, 'spanish':translated_seq})
        except Exception as e:
            return Response({'status':0,'error':repr(e)})

class LanguageDetect(APIView):

    def  post(self, request):
        try:
            entered_text = request.POST.get('data','').lower()
            language = detect(entered_text)

            return Response({'status':1, 'lang':language})
        except Exception as e:
            return Response({'status':0,'error':repr(e)})
