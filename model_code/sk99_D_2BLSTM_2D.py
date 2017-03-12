# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 09:11:25 2016

@author: zhangzhi
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 10:07:30 2016

@author: zhangzhi
"""
from keras.models import Sequential, model_from_json
from keras.models import Model
from keras.layers import Input,Dense, Dropout, LSTM, merge, GRU
#from utils import pre_data
from keras.layers.wrappers import TimeDistributed

def build_model(n_feat = 528, n_class = 21):
    n_hidden = 99
    input = Input(shape = (None,n_feat), name = 'inputs')
    
    dense1 = TimeDistributed(Dense(output_dim = 198, activation='relu', W_regularizer = l1(0.01)))(input)    
    
    f_lstm1 = LSTM(output_dim = n_hidden, return_sequences = True)(dense1)
    b_lstm1 = LSTM(output_dim = n_hidden,return_sequences = True,go_backwards = True)(dense1)
    lstm1 = merge([f_lstm1, b_lstm1], mode='concat')    
#    dense1 = TimeDistributed(Dense(output_dim = 99, activation='relu'))(lstm1)
    f_lstm2 = LSTM(output_dim = n_hidden, return_sequences = True)(lstm1)
    b_lstm2 = LSTM(output_dim = n_hidden,return_sequences = True,go_backwards = True)(lstm1)
    lstm2 = merge([f_lstm2, b_lstm2], mode='concat')      
    dense2 = TimeDistributed(Dense(output_dim = 198, activation='relu'))(lstm2)  
    
    output = TimeDistributed(Dense(output_dim = n_class, activation='softmax'))(dense2)
    
    model = Model(input = input, output = output)
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
#    lstm = LSTM(output_dim = n_hidden, return_sequences = True)(input)
#    output = TimeDistributed(Dense(output_dim = n_class, activation='softmax'))(lstm)
#    model = Model(input = input, output = output)
#    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
    print model.summary()
    return model