# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 14:39:31 2016

@author: zhangzhi
"""


import sys, os,random,zipfile
from shutil import copyfile
import matplotlib.pyplot as plt
from glob import glob
from ChalearnLAPSample import GestureSample
from utils import IsLeftDominant
from utils import Extract_feature_Realtime,Extract_feature_pva
from utils import Extract_feature_UNnormalized,normalize,Extract_feature_normalized_ALL, Smooth_Skeleton
import numpy as np
import cPickle as pickle

def generate_position(feature_name = 'sk_position_33',labels_name = 'labels_raw'):
    print("Extracting the training set of position")
    data=os.path.join("E:\\program\\Chalearn\\rawdata\\train\\")  
    # Get the list of training samples
    samples=os.listdir(data)
    target_dir = 'E:\\program\\Chalearn\\Chalearn_LSTM\\target\\'
    output_dir = 'E:\\program\\Chalearn\\Chalearn_LSTM\\feature\\'+feature_name
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)
    used_joints = ['ElbowLeft', 'WristLeft', 'ShoulderLeft','HandLeft',
                    'ElbowRight', 'WristRight','ShoulderRight','HandRight',
                    'Head','Spine','HipCenter']
    njoints = len(used_joints)
    
#    f = open('SK_normalization.pkl','r')
#    normal_params = pickle.load(f)
#    f.close()
#    Mean = normal_params['Mean1']
#    Std = normal_params['Std1']
    
    count = 0
#    target_category = 21
    Target_all = []
    #Feature_all =  numpy.zeros(shape=(400000, (njoints*(njoints-1)/2 + njoints**2)*3),dtype=numpy.float32)
    for file_count, file in enumerate(samples):
        if int(file[-8:-4])!=417 and int(file[-8:-4])!=675: 
            print("\t Processing file " + file)
            # Create the object to access the sample
            smp=GestureSample(os.path.join(data,file))
            # ###############################################
            # USE Ground Truth information to learn the model
            # ###############################################
            # Get the list of actions for this frame
            gesturesList=smp.getGestures()
            frame_num = smp.getNumFrames()
            Feature_Array = np.zeros(shape = (frame_num , 3*3*njoints),dtype=np.float32)
    #        Target = np.zeros( shape=(frame_num, target_category), dtype=np.uint8)
            
            #feature generate
            Feature_Array, valid_skel = Extract_feature_normalized_ALL(smp,used_joints, 1, frame_num)
#            Feature_Array = Extract_feature_Realtime(Skeleton_matrix, njoints)
#            Skeleton_matrix = Smooth_Skeleton(Skeleton_matrix, window_len = 5, smooth_mode = 'gaussian')            
#            Feature_Array = Extract_feature_pva(Skeleton_matrix,njoints)     
            
            Mean = np.mean(Feature_Array,axis = 0)
            Std = np.std(Feature_Array,axis = 0)
            
            Feature_Array = normalize(Feature_Array,Mean,Std)
            
            
            #save sample sk features
            output_name = '%04d.npy'%count
#            output_name = file[-8:-4]+'.npy'
            np.save(os.path.join(output_dir,output_name), Feature_Array)
            
            count += 1
            #target generate
            
            labels = np.zeros(frame_num, np.uint8)
            for row in gesturesList:
                labels[int(row[1])-1:int(row[2])-1] = int(row[0])
            Target_all.append(labels)
            del smp        
    np.save(target_dir+'%s.npy'%labels_name,Target_all)
def generate_pva(feature_name = 'sk_pva_99',labels_name = 'labels_raw'):
    print("Extracting the training set of position, velocity and acceleration")
    data=os.path.join("E:\\program\\Chalearn\\rawdata\\train\\")  
    # Get the list of training samples
    samples=os.listdir(data)
    target_dir = 'E:\\program\\Chalearn\\Chalearn_LSTM\\target\\'
    output_dir = 'E:\\program\\Chalearn\\Chalearn_LSTM\\feature\\'+feature_name
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)
    used_joints = ['ElbowLeft', 'WristLeft', 'ShoulderLeft','HandLeft',
                    'ElbowRight', 'WristRight','ShoulderRight','HandRight',
                    'Head','Spine','HipCenter']
    njoints = len(used_joints)
    
#    f = open('SK_normalization.pkl','r')
#    normal_params = pickle.load(f)
#    f.close()
#    Mean = normal_params['Mean1']
#    Std = normal_params['Std1']
    
    count = 0
#    target_category = 21
    Target_all = []
    #Feature_all =  numpy.zeros(shape=(400000, (njoints*(njoints-1)/2 + njoints**2)*3),dtype=numpy.float32)
    for file_count, file in enumerate(samples):
        if int(file[-8:-4])!=417 and int(file[-8:-4])!=675: 
            print("\t Processing file " + file)
            # Create the object to access the sample
            smp=GestureSample(os.path.join(data,file))
            # ###############################################
            # USE Ground Truth information to learn the model
            # ###############################################
            # Get the list of actions for this frame
            gesturesList=smp.getGestures()
            frame_num = smp.getNumFrames()
            Feature_Array = np.zeros(shape = (frame_num , 3*3*njoints),dtype=np.float32)
    #        Target = np.zeros( shape=(frame_num, target_category), dtype=np.uint8)
            
            #feature generate
            Skeleton_matrix, valid_skel = Extract_feature_normalized_ALL(smp,used_joints, 1, frame_num)
#            Feature_Array = Extract_feature_Realtime(Skeleton_matrix, njoints)
#            Skeleton_matrix = Smooth_Skeleton(Skeleton_matrix, window_len = 5, smooth_mode = 'gaussian')            
            Feature_Array = Extract_feature_pva(Skeleton_matrix,njoints)     
            
            Mean = np.mean(Feature_Array,axis = 0)
            Std = np.std(Feature_Array,axis = 0)
            
            Feature_Array = normalize(Feature_Array,Mean,Std)
            
            
            #save sample sk features
            output_name = '%04d.npy'%count
#            output_name = file[-8:-4]+'.npy'
            np.save(os.path.join(output_dir,output_name), Feature_Array)
            
            count += 1
            #target generate
            
            labels = np.zeros(frame_num, np.uint8)
            for row in gesturesList:
                labels[int(row[1])-1:int(row[2])-1] = int(row[0])
            Target_all.append(labels)
            del smp        
    np.save(target_dir+'%s.npy'%labels_name,Target_all)
#    f = open('Targets.pkl','wb')
#    pickle.dump( { "Targets_all": Target_all},f)
#    f.close()
def generate_eigenjoint(feature_name = 'sk_eigenjoint_nor_528',labels_name = 'labels_raw'):
     # Data folder (Training data)
    print("Extracting the training files")
    data=os.path.join("E:\\program\\Chalearn\\rawdata\\train\\")  
    target_dir = 'E:\\program\\Chalearn\\Chalearn_LSTM\\target\\'
    # Get the list of training samples
    samples=os.listdir(data)
    output_dir = 'E:\\program\\Chalearn\\Chalearn_LSTM\\feature\\'+feature_name
    
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)
    used_joints = ['ElbowLeft', 'WristLeft', 'ShoulderLeft','HandLeft',
                    'ElbowRight', 'WristRight','ShoulderRight','HandRight',
                    'Head','Spine','HipCenter']
    njoints = len(used_joints)
    
    f = open('SK_normalization.pkl','r')
    normal_params = pickle.load(f)
    f.close()
    Mean = normal_params['Mean1']
    Std = normal_params['Std1']
    
    count = 0
#    target_category = 21
    Target_all = []
    #Feature_all =  numpy.zeros(shape=(400000, (njoints*(njoints-1)/2 + njoints**2)*3),dtype=numpy.float32)
    for file_count, file in enumerate(samples):
        if int(file[-8:-4])!=417 and int(file[-8:-4])!=675: 
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
                        
            Feature_Array = normalize(Feature_Array,Mean,Std)
            add_ = Feature_Array[-1].reshape((1,Feature_Array.shape[1]))
            Feature_Array = np.concatenate((Feature_Array,add_),axis = 0)
            
            #save sample sk features
            output_name = '%04d.npy'%count
            
            count += 1
            np.save(os.path.join(output_dir,output_name), Feature_Array)
            
            
            #target generate
            
            labels = np.zeros(frame_num, np.uint8)
            for row in gesturesList:
                labels[int(row[1])-1:int(row[2])-1] = int(row[0])
            Target_all.append(labels)
            del smp        
    
    np.save(target_dir+'%s.npy'%labels_name,Target_all)

def generate_map_file(sub_sequence_length = 128, valid_segment_idx = 650, data_name = 'sk_eigenjoint_nor_528'):
    
    map_dir = 'map_file'
    data_dir = 'feature/'+data_name+'/'
    
    if not os.path.exists(map_dir): 
        os.makedirs(map_dir)
        
        
    file_paths = glob(data_dir+"/*.npy")
    file_paths.sort()    
    for index, file_path in enumerate(file_paths):
        if index < valid_segment_idx:
            sequence = np.load(file_path)
            sequence_length = sequence.shape[0]
            if sequence_length<sub_sequence_length:
                print('The %d file is not long enough!'%index)
            n_sub_sequence = sequence_length - sub_sequence_length + 1
            file_index = int(file_path[-8:-4])
            file_map = np.zeros(shape = (n_sub_sequence,2), dtype = np.int)
            file_index_array = np.array([file_index]*n_sub_sequence).T
            time_index_array = np.array(range(n_sub_sequence)).T  
            file_map[:,0] = file_index_array
            file_map[:,1] = time_index_array
            if index == 0 :
                maps = file_map
            else:
                maps = np.concatenate((maps,file_map))
            
    maps_name = 'map_seg_%d_sub_%d.npy'%(valid_segment_idx,sub_sequence_length)
    np.save(os.path.join(map_dir,maps_name),maps)
    return maps

def extract_order_batch(batch_idx = 0, batch_size = 258, sub_sequence_length = 128, n_feat = 528):
    #open map_file
    map_dir = 'map_file'
    data_dir = 'sk_feature_pva'
    maps = np.load(os.path.join(map_dir,'shmmap_%04d.npy'%sub_sequence_length))
    
    labels = np.load('data/labels_raw.npy')
    import cPickle as pickle
    idx_batch = maps[batch_idx*batch_size:(batch_idx+1)*batch_size]
    unique_file_idx, index_starts , counts = np.unique(idx_batch[:,0],
                                                       return_index=True, 
                                                       return_inverse=False, 
                                                       return_counts=True)
    
#    batch_feat = np.zeros(shape = (batch_size, sub_sequence_length, n_feat))
    batch_label = np.zeros(shape = (batch_size, sub_sequence_length))
    
    for idx,file_idx in enumerate(unique_file_idx):
        print file_idx
        #feature
        features=np.load(os.path.join(data_dir,'%04d.npy'%file_idx))
        file_idx_batch = idx_batch[index_starts[idx]:index_starts[idx]+counts[idx],1]
        tmp = np.array([features[i:i+sub_sequence_length,:] for i in file_idx_batch])
        if idx==0:
            batch_feat = tmp
        else:
            batch_feat = np.concatenate((batch_feat,tmp))
        #label
        file_label = labels[file_idx][:-1]
        label_tmp = np.array([file_label[i:i+sub_sequence_length] for i in file_idx_batch])
        if idx==0:
            batch_label = label_tmp
        else:
            batch_label = np.concatenate((batch_label,label_tmp))
        
    return batch_feat, batch_label
     
def extract_random_batch(batch_size = 258, map_name = 'map_seg_650_sub_64', feature_name = 'sk_eigenjoint_nor_528'):
    sub_sequence_length = int(map_name.split('_')[-1])
    n_feat = int(feature_name.split('_')[-1])
    
    
    map_dir = 'map_file'
    data_dir = 'feature/'+feature_name
    maps = np.load(os.path.join(map_dir,map_name+'.npy'))
    
    labels = np.load('target/labels_raw.npy')
    
    idx_batch = maps[np.random.randint(0, maps.shape[0], size=(batch_size,))]
    #idx_batch sort
    idx_batch = idx_batch[np.argsort(idx_batch[:,0],axis = 0)]
    unique_file_idx, index_starts , counts = np.unique(idx_batch[:,0],
                                                       return_index=True, 
                                                       return_inverse=False, 
                                                       return_counts=True)
    
    batch_feat = np.zeros(shape = (batch_size, sub_sequence_length, n_feat))
    batch_label = np.zeros(shape = (batch_size, sub_sequence_length))
    
    for idx,file_idx in enumerate(unique_file_idx):
#        print file_idx
        features=np.load(os.path.join(data_dir,'%04d.npy'%file_idx))
        file_idx_batch = idx_batch[index_starts[idx]:index_starts[idx]+counts[idx],1]
        tmp = np.array([features[i:i+sub_sequence_length,:] for i in file_idx_batch])
        if idx==0:
            batch_feat = tmp
        else:
            batch_feat = np.concatenate((batch_feat,tmp))
            
#        file_label = labels[file_idx][:-1]
        file_label = labels[file_idx]
        label_tmp = np.array([file_label[i:i+sub_sequence_length] for i in file_idx_batch])
        if idx==0:
            batch_label = label_tmp
        else:
            batch_label = np.concatenate((batch_label,label_tmp))
                                                     
    return batch_feat, batch_label
    
def extract_random_batch_RAM(Feature_all ,maps, labels, batch_size = 258):
    sub_sequence_length = int(map_name.split('_')[-1])
#    n_feat = int(feature_name.split('_')[-1])
    n_feat = Feature_all[0].shape[-1]
    

    
    idx_batch = maps[np.random.randint(0, maps.shape[0], size=(batch_size,))]
    #idx_batch sort
    idx_batch = idx_batch[np.argsort(idx_batch[:,0],axis = 0)]
    unique_file_idx, index_starts , counts = np.unique(idx_batch[:,0],
                                                       return_index=True, 
                                                       return_inverse=False, 
                                                       return_counts=True)
    
    batch_feat = np.zeros(shape = (batch_size, sub_sequence_length, n_feat))
    batch_label = np.zeros(shape = (batch_size, sub_sequence_length))
    
    for idx,file_idx in enumerate(unique_file_idx):
#        print file_idx
#        features=np.load(os.path.join(data_dir,'%04d.npy'%file_idx))
        features = Feature_all[file_idx]
        file_idx_batch = idx_batch[index_starts[idx]:index_starts[idx]+counts[idx],1]
        tmp = np.array([features[i:i+sub_sequence_length,:] for i in file_idx_batch])
        if idx==0:
            batch_feat = tmp
        else:
            batch_feat = np.concatenate((batch_feat,tmp))
            
#        file_label = labels[file_idx][:-1]
        file_label = labels[file_idx]
        label_tmp = np.array([file_label[i:i+sub_sequence_length] for i in file_idx_batch])
        if idx==0:
            batch_label = label_tmp
        else:
            batch_label = np.concatenate((batch_label,label_tmp))
                                                     
    return batch_feat, batch_label
if __name__ == '__main__':
#    map_dir = 'map_file'
#    batch_size = 258
#    sub_sequence_length = 64
#    maps = np.load(os.path.join(map_dir,'map_seg_650_sub_64.npy'))   
#    n_sample  = maps.shape[0]
#    n_batch = n_sample/batch_size
#    
#    import time
#    start = time.time()    
#    
#    Feature_all = np.load('Feature_all.npy')
#    
#    map_name = 'map_seg_650_sub_64'
#    map_dir = 'map_file'
#    maps = np.load(os.path.join(map_dir,map_name+'.npy'))
#    
#    labels = np.load('target/labels_raw.npy')
#
#    for i_batch in range(n_batch):
#        batch_feat, batch_label = extract_random_batch_RAM(Feature_all, maps, labels, batch_size = batch_size)
#        print i_batch
#    end = time.time()
#    print 'a epoch spend %f s'%(end-start)

#    generate_pva(feature_name = 'sk_pva_99',labels_name = 'labels_raw')
#    generate_position()
#    generate_map_file(sub_sequence_length = 128,
#                      valid_segment_idx=697,data_name='sk_pva_99')
    generate_eigenjoint()

    

