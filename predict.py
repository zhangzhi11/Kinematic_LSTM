# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 20:48:34 2016

@author: zhangzhi
"""

import numpy as np
from keras.models import Sequential, model_from_json
from keras.models import Model
from keras.layers import Input,Dense, Dropout, LSTM, merge
#from utils import pre_data
from keras.layers.wrappers import TimeDistributed
import keras.wrappers
from keras.utils import np_utils
import data_preproc
import os
import time
from glob import glob
from utils import eval_jaccard
import importlib
from model import model
import matplotlib.pyplot as plt


test_x = np.load('sk_feature/0000.npy')
test_y = np.load('data/labels_raw.npy')[0][:-1]
test_y_cate = np.zeros(shape = (test_y.shape[0],21))
for i in range(test_y.shape[0]):
    test_y_cate[i,test_y[i]] = 1
    


test_x = test_x.reshape(1,test_x.shape[0],test_x.shape[1])


clf = model()
clf.load_model(model_name = 'sk528_LSTM500')
c = clf.model.predict(test_x)[0]
startFrame = 0
endFrame = c.shape[0]-1
frames_count = np.array(range(startFrame,endFrame+1))
f, (a1,a2) = plt.subplots(2,1,figsize=(15,4))

for i in range(c.shape[1]):

    if not i ==0:       
        pred_label_temp = c[:,i]
        a1.plot(frames_count, pred_label_temp,  linewidth=2.0)
    
for i in range(c.shape[1]):
    if not i ==0:       
        pred_label_temp = test_y_cate[:,i]
        a2.plot(frames_count, pred_label_temp,  linewidth=2.0)
plt.show()    
    
#        if True:
#            im  = imdisplay(global_score)
#            plt.clf()
#            plt.imshow(im, cmap='gray')
#            plt.plot(range(global_score.shape[-1]), path, color='c',linewidth=2.0)
#            plt.xlim((0, global_score.shape[-1]))
#            # plot ground truth
#            for gesture in gesturesList:
#            # Get the gesture ID, and start and end frames for the gesture
#                gestureID,startFrame,endFrame=gesture
#                frames_count = numpy.array(range(startFrame, endFrame+1))
#                pred_label_temp = ((gestureID-1) *10 +5) * numpy.ones(len(frames_count))
#                plt.plot(frames_count, pred_label_temp, color='r', linewidth=5.0)
#            
#            # plot clean path
#            for i in range(len(begin_frame)):
#                frames_count = numpy.array(range(begin_frame[i], end_frame[i]+1))
#                pred_label_temp = ((pred_label[i]-1) *10 +5) * numpy.ones(len(frames_count))
#                plt.plot(frames_count, pred_label_temp, color='#ffff00', linewidth=2.0)
#
#            plt.show()