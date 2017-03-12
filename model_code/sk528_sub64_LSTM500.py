# -*- coding: utf-8 -*-
"""
Created on Thu Nov 03 09:52:00 2016

@author: zhangzhi
"""
import numpy as np
from keras.models import Sequential, model_from_json
from keras.models import Model
from keras.layers import Input,Dense, Dropout, LSTM, merge
#from utils import pre_data
from keras.layers.wrappers import TimeDistributed
import keras.wrappers
n_feat = 528
n_class = 21
def build_model():
    n_hidden = 500
    
    input = Input(shape = (None,n_feat), name = 'inputs')
    lstm = LSTM(output_dim = n_hidden, return_sequences = True)(input)
    output = TimeDistributed(Dense(output_dim = n_class, activation='softmax'))(lstm)
    model = Model(input = input, output = output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
    return model
