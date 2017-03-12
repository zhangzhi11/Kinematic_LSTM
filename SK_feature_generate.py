#-------------------------------------------------------------------------------
# Name:        Starting Kit for ChaLearn LAP 2014 Track3
# Purpose:     Show basic functionality of provided code
#
# Author:      Xavier Baro
# Author:      Di Wu: stevenwudi@gmail.com
# Created:     24/03/2014
# Copyright:   (c) Chalearn LAP 2014
# Licence:     GPL3
#-------------------------------------------------------------------------------
import sys, os,random,zipfile
from shutil import copyfile
import matplotlib.pyplot as plt

from ChalearnLAPSample import GestureSample
from utils import IsLeftDominant
from utils import Extract_feature_Realtime
from utils import Extract_feature_UNnormalized
import numpy as np

# Data folder (Training data)
print("Extracting the training files")
data=os.path.join("E:\\program\\Chalearn\\rawdata\\train\\")  
# Get the list of training samples
samples=os.listdir(data)
used_joints = ['ElbowLeft', 'WristLeft', 'ShoulderLeft','HandLeft',
                'ElbowRight', 'WristRight','ShoulderRight','HandRight',
                'Head','Spine','HipCenter']
njoints = len(used_joints)
count = 0
target_category = 21
# pre-allocating the memory
#Feature_all =  numpy.zeros(shape=(400000, (njoints*(njoints-1)/2 + njoints**2)*3),dtype=numpy.float32)
Feature_all = []
Target_all = []
#Targets = numpy.zeros( shape=(400000, target_category), dtype=numpy.uint8)
# Access to each sample
for file_count, file in enumerate(samples):
    #if not file.endswith(".zip"):
    #    continue;        
    if file_count<800: 
        print("\t Processing file " + file)
        # Create the object to access the sample
        smp=GestureSample(os.path.join(data,file))
        # ###############################################
        # USE Ground Truth information to learn the model
        # ###############################################
        # Get the list of actions for this frame
        gesturesList=smp.getGestures()
        frame_num = smp.getNumFrames()
        Feature_Array = np.zeros(shape = (frame_num , (njoints*(njoints-1)/2 + njoints**2)*3),dtype=np.float32)
#        Target = np.zeros( shape=(frame_num, target_category), dtype=np.uint8)
        
        #feature generate
        Skeleton_matrix, valid_skel = Extract_feature_UNnormalized(smp,used_joints, 1, frame_num)
        Feature_Array = Extract_feature_Realtime(Skeleton_matrix, njoints)
        Feature_all.append(Feature_Array)
        
        #target generate
        
        labels = np.zeros(frame_num, np.uint8)
        for row in gesturesList:
            labels[int(row[1])-1:int(row[2])-1] = int(row[0])
        Target_all.append(labels)
        del smp

# save the skeleton file:


import cPickle as pickle
f = open('Feature_train_realtime.pkl','wb')
pickle.dump( {"Feature_all": Feature_all, "Targets_all": Target_all},f)
f.close()


#    f = file('Feature_train_realtime.pkl','rb' )
#    Feature_train = cPickle.load(f)
#    f.close()
#
#    f = file('Feature_all_neutral_realtime.pkl','rb' )
#    Feature_train_neural = cPickle.load(f)
#    f.close()
#import scipy.io as sio
#sio.savemat('Feature_all_train__realtime.mat', { "Feature_all": Feature_all[0:end_frame, :], "Targets_all": Targets[0:end_frame, :] })
#



