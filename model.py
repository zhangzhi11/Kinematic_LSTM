# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 09:45:32 2016

@author: zhangzhi
"""

import numpy as np
from keras.models import Sequential, model_from_json
from keras.models import Model
from keras.layers import Input,Dense, Dropout, LSTM, merge, GRU, MaxPooling1D, AveragePooling1D
#from utils import pre_data
from keras.layers.wrappers import TimeDistributed
from keras.layers.noise import GaussianNoise
import keras.wrappers
from keras.utils import np_utils
import os
import time
from glob import glob
from utils import eval_jaccard,plot_confusion_matrix,plot_fig
import importlib
import data_preproc
from sklearn.metrics import classification_report,confusion_matrix 
import matplotlib.pyplot as plt
import itertools
from keras.regularizers import activity_l1,l1

batch_size = 258 
sub_sequence_length = 64 
n_feat = 528
n_class = 21

model_dir = 'model'
data_dir = 'feature'
taget_dir = 'target'
map_dir = 'map_file'

def batch_generator(maps, labels, sub_sequence_length = 64,  batch_size = 258,  feature_name = 'sk_eigenjoint_nor_528'):
     
    n_feat = int(feature_name.split('_')[-1])
    
    
    
    data = data_dir+'/'+feature_name
    
    
    n_samples = maps.shape[0]
#    n_batchs = int(n_samples/batch_size)
#    b_idx = 0
    
    
    
    while True:
        
        idx_batch = maps[np.random.randint(0, n_samples, size=(batch_size,))]
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
            features=np.load(os.path.join(data,'%04d.npy'%file_idx),mmap_mode = 'r')
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
        batch_label_cate = np.zeros(shape=[batch_label.shape[0], batch_label.shape[1], n_class], dtype = np.uint8)
        for i,y_line in enumerate(batch_label):
                batch_label_cate[i,:,:] = np_utils.to_categorical(y_line,nb_classes = n_class)                                                        
        yield batch_feat, batch_label_cate
        
#        b_idx += 1
#        if b_idx>=n_batchs:
#            break





def build_model_from_code(model_name = 'LSTM500'):     
    print 'Model:', model_name
    cfg = importlib.import_module("model_code.%s" % model_name)
    print 'Building'
    model = cfg.build_model()
    print model.summary()
    return model
        
def build_model(n_feat = 528, n_class = 21):
    n_hidden = 200
    input = Input(shape = (None,n_feat), name = 'inputs')
#    noise_in = GaussianNoise(sigma = 0.1)(input)
#    ap = AveragePooling1D(pool_length=3, stride=1, border_mode='same')(input)
    dense1 = TimeDistributed(Dense(output_dim = n_hidden, activation='relu'))(input)    
    
#    drop_dense1 = Dropout(0.2)(dense1)
    
    f_lstm1 = LSTM(output_dim = n_hidden, return_sequences = True)(dense1)
    b_lstm1 = LSTM(output_dim = n_hidden,return_sequences = True,go_backwards = True)(dense1)
    lstm1 = merge([f_lstm1, b_lstm1], mode='concat')    

    f_lstm2 = LSTM(output_dim = n_hidden, return_sequences = True)(lstm1)
    b_lstm2 = LSTM(output_dim = n_hidden,return_sequences = True,go_backwards = True)(lstm1)
    lstm2 = merge([f_lstm2, b_lstm2], mode='concat')  
    
#    f_lstm3 = LSTM(output_dim = n_hidden, return_sequences = True)(lstm2)
#    b_lstm3 = LSTM(output_dim = n_hidden,return_sequences = True,go_backwards = True)(lstm2)
#    lstm3 = merge([f_lstm3, b_lstm3], mode='concat')  
     
    dense2 = TimeDistributed(Dense(output_dim = 500, activation='relu'))(lstm2)  
    
    output = TimeDistributed(Dense(output_dim = n_class, activation='softmax'))(dense2)
    
    model = Model(input = input, output = output)
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
#    lstm = LSTM(output_dim = n_hidden, return_sequences = True)(input)
#    output = TimeDistributed(Dense(output_dim = n_class, activation='softmax'))(lstm)
#    model = Model(input = input, output = output)
#    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
    print model.summary()
    return model
        
        
def save_model(model, name = 'zz'):
    if not os.path.exists(model_dir): 
        os.mkdir(model_dir)
    out_model_dir = os.path.join(model_dir,name)
    if not os.path.exists(out_model_dir): 
        os.mkdir(out_model_dir)
    json_string = model.to_json()  #等价于 json_string = model.get_config()  
    open(os.path.join(out_model_dir,name+'.json'),'w').write(json_string)    
    model.save_weights(os.path.join(out_model_dir, name+'.h5'))   
        
def load_model( model_name = 'zz'):
    out_model_dir = os.path.join(model_dir,model_name)
    model = model_from_json(open(os.path.join(out_model_dir,model_name+'.json')).read())
    model.load_weights(os.path.join(out_model_dir,model_name+'.h5'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])      
    return model
    
def valid_model(model, labels , data_name = 'sk_eigenjoint_nor_528',  valid_segment_idx= 650):
    data = data_dir+'/'+data_name
    file_paths = glob(data+"/*.npy")
    losses = []    
    for index, file_path in enumerate(file_paths) :
        if index >= valid_segment_idx:
            valid_x = np.load(file_path)
            valid_x = np.reshape(valid_x, newshape = (1,valid_x.shape[0],valid_x.shape[1]))
            valid_y = labels[index]
#                valid_y = valid_y[:-1]
            pred_y = model.predict_on_batch(valid_x)
            pred_y = np.argmax(pred_y,axis = 2)
            loss = eval_jaccard(valid_y,pred_y.ravel())
            losses.append(loss)
    return np.mean(losses)   
    
def train_model(model, maps, labels, model_name, data_name,
                n_epoch = 10,
                valid_segment_idx = 650, 
                sub_sequence_length = 64,
                batch_size = 256):    
    best_loss = -1.0
    print('Train....')
    
#    class_weight = []
    for epoch_idx in range(n_epoch):
        print('Start train %d epoch!'%epoch_idx)
        
        start = time.time()
        model.fit_generator(generator = batch_generator(maps, labels,
                                                        sub_sequence_length = sub_sequence_length,
                                                        batch_size = batch_size, 
                                                        feature_name = data_name),
                            samples_per_epoch = maps.shape[0],
                            verbose = 1,
                            nb_epoch = 1)
        valid_loss = valid_model(model, labels,data_name = data_name, valid_segment_idx = valid_segment_idx)
        if valid_loss>=best_loss:
            save_model(model,name = data_name+'_'+model_name)
        print 'valid jaccard index: %.4f'%valid_loss
        end = time.time()
        print 'this epoch training spend %.2f s'%(end-start)
    return model
    
def test_subseq_model(model, labels,data_name = 'sk_eigenjoint_nor_528',  valid_segment_idx= 650):
    sub_seq = 64    
    data = data_dir+'/'+data_name
    file_paths = glob(data+"/*.npy")
    losses = []  
    ground_ys = []
    pred_ys = []
    time_lens =[]
    for index, file_path in enumerate(file_paths) :
        if index >= valid_segment_idx:
            print 'Predict '+file_path
            valid_x = np.load(file_path)
            
            valid_batch_x = np.array([valid_x[i:i+sub_seq] for i in range(valid_x.shape[0]-sub_seq)])
            
        
#            valid_x = np.reshape(valid_x, newshape = (1,valid_x.shape[0],valid_x.shape[1]))
            valid_y = labels[index]
##                valid_y = valid_y[:-1]
            pred_y = model.predict_on_batch(valid_batch_x) 
            
            pred_matrix = np.zeros(shape = (pred_y.shape[0] , pred_y.shape[0]+64, 21))
            for i in range(pred_y.shape[0]):
                pred_matrix[i,i:i+64,:] = pred_y[i,:,:]
            pred_pro_list = np.zeros(shape = (valid_x.shape[0]))
            pred_y_list = np.zeros(shape = (valid_x.shape[0]))
            for i in range(valid_x.shape[0]):
                a = pred_matrix[:,i,:]
                pred_pro = np.max(a,axis = 1)
    
                pred_pro_id = np.argmax(pred_pro)
                pred_y_id = np.argmax(a[pred_pro_id])
                pred_max_pro = np.max(pred_pro)
                pred_pro_list[i] = pred_max_pro
                pred_y_list[i] = pred_y_id
                
#            for i in range(valid_x.shape[0]):
#                pred_time = np.zeros((64,21))
#                if i<64:
#                    for j in range(i+1):
#                        pred_time[j,:] = pred_y[j,i-j,:]
#                elif i>=64 and i<pred_y.shape[0]:
#                    for j in range(64):
#                        pred_time[j,:] = pred_y[]
                    
##            file_name = str(index-valid_segment_idx)
##            plot_fig(pred_y,valid_y,file_name,save_flag = True)            
#            
#            pred_y = np.argmax(pred_y,axis = 2)
#            pred_y = pred_y.ravel()
            ground_ys.append(valid_y)
            pred_ys.append(pred_y_list) 
##            pred_y = clear_pred(pred_y)  
#
#
            loss = eval_jaccard(valid_y,pred_y_list)
            losses.append(loss)
            time_lens.append(valid_x.shape[1])
    ground_ys = np.concatenate(ground_ys)
    pred_ys = np.concatenate(pred_ys)
    
    print(classification_report(ground_ys, pred_ys))
    
    cnf_matrix = confusion_matrix(ground_ys, pred_ys)
    np.set_printoptions(precision=2)
    cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=range(21), normalize=True,
              title='Normalized confusion matrix')
              
    plt.show()
#    print cnf_matrix

    return losses,cnf_matrix,time_lens
    
def test_model(model, labels,data_name = 'sk_eigenjoint_nor_528',  valid_segment_idx= 650):
    data = data_dir+'/'+data_name
    file_paths = glob(data+"/*.npy")
    losses = []  
    ground_ys = []
    pred_ys = []
    for index, file_path in enumerate(file_paths) :
        if index >= valid_segment_idx:
            print 'Predict '+file_path
            valid_x = np.load(file_path)
            valid_x = np.reshape(valid_x, newshape = (1,valid_x.shape[0],valid_x.shape[1]))
            valid_y = labels[index]
#                valid_y = valid_y[:-1]
            pred_y = model.predict_on_batch(valid_x)
            file_name = str(index-valid_segment_idx)
            plot_fig(pred_y,valid_y,file_name,save_flag = True)            
            
            pred_y = np.argmax(pred_y,axis = 2)
            pred_y = pred_y.ravel()
            ground_ys.append(valid_y)
            pred_ys.append(pred_y) 
#            pred_y = clear_pred(pred_y)  


            loss = eval_jaccard(valid_y,pred_y)
            losses.append(loss)
    ground_ys = np.concatenate(ground_ys)
    pred_ys = np.concatenate(pred_ys)
    
    print(classification_report(ground_ys, pred_ys))
    
    cnf_matrix = confusion_matrix(ground_ys, pred_ys)
    np.set_printoptions(precision=2)
    cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=range(21), normalize=True,
              title='Normalized confusion matrix')
              
    plt.show()
#    print cnf_matrix

    return losses,cnf_matrix

#print 'Pridect and Veterbi decode...'
#labels = np.load('target/labels_raw.npy')
#model = load_model('sk_pva_99_4BLSTM')
#avg_jaccard1,avg_jaccard2 = test_model(model,labels,data_name = 'sk_pva_99')
#print 'The average jaccard index of veterbi decode is %.4f'%avg_jaccard1
#print 'The average jaccard index of raw LSTM is %.4f'%avg_jaccard2
            

if __name__ == '__main__':
    
    
    model_name = 'D200_2BLSTM200_2D500'
    data_name = 'sk_position_33'
    
    print 'Loading map file...'
    map_dir = 'map_file'
    maps = np.load(os.path.join(map_dir,'map_seg_697_sub_64.npy'))   
    sub_sequence_length = 64
    n_sample  = maps.shape[0]
    n_epoch = 10
    
    print 'Loading label file...'
    labels = np.load(taget_dir+'/labels_raw.npy')
    
    print 'Build model...'
    model = build_model(n_feat = 33, n_class = 21)
    

#    print 'Loading Model...'
#    model = load_model(model_name = data_name+'_'+model_name)
#    print model.summary()
    print 'Train model...'
    model = train_model(model, maps, labels,model_name, data_name, n_epoch = 10, 
                        valid_segment_idx = 697, 
                        sub_sequence_length = sub_sequence_length,
                        batch_size = 256)
    
#    print 'Test...'
#    jaccards,cnf_matrix = test_model(model, labels, data_name = 'sk_eigenjoint_nor_528',valid_segment_idx= 697)   
#    avg_jaccard = np.mean(jaccards)

#model = model()
#model.build_model_from_code(model_name = 'LSTM500')
#model.train_model(batch_mode = batch_mode, n_sample = n_sample)
#    save_model(model)      