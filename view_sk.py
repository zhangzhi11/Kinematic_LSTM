# -*- coding: utf-8 -*-
"""
Created on Tue Dec 06 09:18:44 2016

@author: zhangzhi
"""

from ChalearnLAPSample import GestureSample,Skeleton
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


used_joints = ['ElbowLeft', 'WristLeft', 'ShoulderLeft','HandLeft',
                'ElbowRight', 'WristRight','ShoulderRight','HandRight',
                'Head','Spine','HipCenter']
def show_sk_image(gesture_id = 100):
    data=os.path.join("E:\\program\\Chalearn\\rawdata\\train\\")  
    # Get the list of training samples
    
    smp = GestureSample(os.path.join(data,'Sample%04d.zip'%gesture_id))
    frame_num = smp.getNumFrames()
    fps = 30
    cv2.namedWindow("sk_image")
    for i in range(1,frame_num+1):
        sk = smp.getSkeletonImage(i)
    
        cv2.imshow("sk_image",sk)
        cv2.waitKey(int(1000/fps))
    cv2.destroyAllWindows() 
    del smp   

def plot_position_time(gesture_id = 100):
    labels = np.load('target/labels_raw.npy')
    data_dir = os.path.join('feature','sk_pva_99')
    data = np.load(os.path.join(data_dir,'%04d.npy'%gesture_id))
    #data的格式为[position,velocity,acceleration]
    position = data[:,:33]
    label = labels[gesture_id]
    test_y_cate = np.zeros(shape = (label.shape[0],21))
    for i in range(label.shape[0]):
        test_y_cate[i,label[i]] = 1
        
    fig, ax = plt.subplots(nrows = 9, sharex=True)
    time_axis = np.arange(position.shape[0])
    for i in range(8):
        for j in range(3):
            ax[i].plot(time_axis,position[:,3*i+j].ravel())
        ax[i].set_title(used_joints[i])
        ax[i].set_xlabel('time')
        ax[i].set_ylabel(used_joints[i])
        
        
    for s in range(test_y_cate.shape[1]):
        if not s ==0:       
            pred_label_temp = test_y_cate[:,s]
            ax[8].plot(time_axis, pred_label_temp)    
    plt.show()
    
    print '0'
if __name__ == '__main__':
    plot_position_time(gesture_id =658)
    show_sk_image(gesture_id = 660)