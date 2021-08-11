# -*- coding: utf-8 -*-

import pandas as pd
import os
import argparse
import time
import pickle
import re
import tensorflow as tf
import numpy as np
import sys, getopt
from subprocess import check_output
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import SeparableConv1D
from tensorflow.python.keras.layers import MaxPooling1D
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import GlobalAveragePooling1D
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from tensorflow.python.keras.preprocessing import text
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn


MAX_VOCAB_SIZE = 10000
embed_dim = 16
lstm_out = 100
batch_size = 32

class model(object):
    """ Example of valid model """

    def __init__(self):
        """ Initialization for model
        :param metadata: a dict formed like:
            {"class_num": 10,
             "language": ZH,
             "num_train_instances": 10000,
             "num_test_instances": 1000,
             "time_budget": 300}
        """
        # self.metadata = metadata
        self.train_output_path = './'
        self.test_input_path = './'
        self.model = LogisticRegression()
        self.label_num = 0

    def fit(self, x_train, y_train, remaining_time_budget=None):
        """Train this algorithm on the NLP task.

         This method will be called REPEATEDLY during the whole training/predicting
         process. So your `train` method should be able to handle repeated calls and
         hopefully improve your model performance after each call.

         ****************************************************************************
         ****************************************************************************
         IMPORTANT: the loop of calling `train` and `test` will only run if
             self.done_training = False
           (the corresponding code can be found in ingestion.py, search
           'M.done_training')
           Otherwise, the loop will go on until the time budget is used up. Please
           pay attention to set self.done_training = True when you think the model is
           converged or when there is not enough time for next round of training.
         ****************************************************************************
         ****************************************************************************
    
        :param train_dataset: tuple, (x_train, y_train)
            x_train: list of str, input training sentence.
            y_train: A `numpy.ndarray` matrix of shape (sample_count, class_num).
                     here `sample_count` is the number of examples in this dataset as train
                     set and `class_num` is the same as the class_num in metadata. The
                     values should be binary.
        :param remaining_time_budget:

        :return: None
        """
        self.label_num = y_train.shape[1]
        x_train = pd.DataFrame(x_train)

        for i in range(x_train.shape[1]):
            try:
                x_train[i] = np.array(x_train[i].values, dtype=np.float)
            except ValueError:
                X = x_train[i].values.reshape(-1,1)
                enc = OneHotEncoder(categories='auto').fit(X) 
                result = enc.transform(X).toarray()
                x_train = x_train.drop([i], axis=1)
                for j in range(result.shape[1]):
                    x_train['a'*(i+1)+'b'*(j+1)] = result[:,j]
        x_train = np.array(x_train)
        
        min_max_scaler = MinMaxScaler(feature_range=(0,1))
        x_train = min_max_scaler.fit_transform(x_train)
        
        y = []
        for i in range(y_train.shape[0]):
            y.append(list(y_train[i]).index('1'))
        
        self.model.fit(x_train,y)
        # print(type(x_train))
        # print(y_train)
        # print(type(y_train))


    def predict(self, x_test, remaining_time_budget=None):
        """
        :param x_test: list of str, input test sentence.
        :param remaining_time_budget:
        :return: A `numpy.ndarray` matrix of shape (sample_count, class_num).
                 here `sample_count` is the number of examples in this dataset as test
                 set and `class_num` is the same as the class_num in metadata. The
                 values should be binary or in the interval [0,1].
        """
        x_test = pd.DataFrame(x_test)

        for i in range(x_test.shape[1]):
            try:
                x_test[i] = np.array(x_test[i].values, dtype=np.float)
            except ValueError:
                X = x_test[i].values.reshape(-1,1)
                enc = OneHotEncoder(categories='auto').fit(X) 
                result = enc.transform(X).toarray()
                x_test = x_test.drop([i], axis=1)
                for j in range(result.shape[1]):
                    x_test['a'*(i+1)+'b'*(j+1)] = result[:,j]
        x_test = np.array(x_test)

        min_max_scaler = MinMaxScaler(feature_range=(0,1))
        x_test = min_max_scaler.fit_transform(x_test)
        y_pred = self.model.predict(x_test)
        y = []
        for i in range(y_pred.shape[0]):
            y.append([0]*y_pred[i]+[1]+[0]*(self.label_num-1-y_pred[i]))
        return np.array(y)
    
    def save(self, path="./"):
        '''
        Save a trained model.
        '''
        pass

    def load(self, path="./"):
        '''
        Load a trained model.
        '''
        pass


