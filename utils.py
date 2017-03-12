""" This file contains different utility functions that are not connected
in anyway to the networks presented in the tutorials, but rather help in
processing the outputs into a more understandable way.

For example ``tile_raster_images`` helps in generating a easy to grasp
image from a set of samples or weights.
"""


import numpy
import numpy.random as random
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix 
import matplotlib.pyplot as plt
import itertools
import cv2

def Smooth_Skeleton(Skeleton_Matrix, window_len = 5, smooth_mode = 'average'):
    #smooth_mode is average ,gaussian,median
    if smooth_mode == 'average':
        Skeleton = cv2.blur(Skeleton_Matrix,(window_len,1))
    elif smooth_mode == 'gaussian':
        Skeleton = cv2.GaussianBlur(Skeleton_Matrix,(window_len,1),0)
    elif smooth_mode == 'median':
        Skeleton = cv2.medianBlur(Skeleton_Matrix,(window_len,1))
    return Skeleton
            
def clear_pred(pred_y, mini_frame = 15):
    begin_frame = []
    end_frame = []
    pred_label = []
    i = 1
    status=True
    while(i< pred_y.shape[-1]-1):
        if pred_y[i-1] == 0 and not pred_y[i] ==0 and status: 
            begin_frame.append(i)
            # python integer divsion will do the floor for us :)
            pred_label .append(pred_y[i])
            i += 1
            status=False
        elif pred_y[i+1] == 0 and not pred_y[i] ==0 and not status:
            end_frame.append(i)
            i += 1
            status=True
        i += 1
    end_frame = np.array(end_frame)
    begin_frame = np.array(begin_frame)
    pred_label= np.array(pred_label)
    if len(begin_frame)> len(end_frame):
        begin_frame = begin_frame[:-1]
    elif len(begin_frame)< len(end_frame):# risky hack! just for validation file 668
        end_frame = end_frame[1:] 
        
    frame_length = end_frame - begin_frame
    mask = frame_length > mini_frame
    
    begin_frame = begin_frame[mask]
    end_frame = end_frame[mask]
    pred_label = pred_label[mask]
    
    cleared_pred = np.zeros_like(pred_y)
    for i,begin in enumerate(begin_frame):
        cleared_pred[begin:end_frame[i]+1] = pred_label[i]
        
    return cleared_pred

def plot_fig(pred_y , label, file_name ='zz', save_flag = False):
    pred_y = pred_y[0]
    test_y_cate = np.zeros(shape = (label.shape[0],21))
    for i in range(label.shape[0]):
        test_y_cate[i,label[i]] = 1
    startFrame = 0
    
    endFrame = pred_y.shape[0]
    frames_count = np.array(range(startFrame,endFrame))
    f, (a1,a2) = plt.subplots(2,1,figsize=(15,4))

    for i in range(pred_y.shape[1]):

        if not i ==0:       
            pred_label_temp = pred_y[:,i]
            a1.plot(frames_count, pred_label_temp,  linewidth=2.0)
    
    for i in range(pred_y.shape[1]):
        if not i ==0:       
            pred_label_temp = test_y_cate[:,i]
            a2.plot(frames_count, pred_label_temp,  linewidth=2.0)
    if save_flag == True:
        print 'save...'
        plt.savefig('fig/'+file_name+'.jpg')
    plt.show()  
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '%.02f'%cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def  IsLeftDominant ( Skeleton_matrix ):
    """
    Check wether the motion is left dominant or right dominant
    """

    elbowDiffLeft = Skeleton_matrix[1:, 0:12] - Skeleton_matrix[0:-1, 0:12]
    elbowDiffRigh = Skeleton_matrix[1:, 12:24] - Skeleton_matrix[0:-1, 12:24]

    motionLeft = numpy.sum( numpy.sqrt( numpy.sum(elbowDiffLeft**2)))
    motionRigh = numpy.sum( numpy.sqrt( numpy.sum(elbowDiffRigh**2)))

    if motionLeft > motionRigh:
        leftDominantFlag = True
    else:
        leftDominantFlag = False
    return leftDominantFlag

def Extract_feature_UNnormalized(smp, used_joints, startFrame, endFrame):
    """
    Extract original features
    """
    frame_num = 0 
    Skeleton_matrix  = numpy.zeros(shape=(endFrame-startFrame+1, len(used_joints)*3))

    for numFrame in range(startFrame,endFrame+1):                    
        # Get the Skeleton object for this frame
        skel=smp.getSkeleton(numFrame)
        for joints in range(len(used_joints)):
            Skeleton_matrix[frame_num, joints*3: (joints+1)*3] = skel.joins[used_joints[joints]][0]
        frame_num += 1

    
    if numpy.allclose(sum(sum(numpy.abs(Skeleton_matrix))),0):
        valid_skel = False
    else:
        valid_skel = True

    return Skeleton_matrix, valid_skel

def eval_jaccard(gt, pred):
    if not gt.shape == pred.shape:
        print 'Predict sequence length is wrong!'
        return 0
    overlap_sum = 0
    gt_sum = 0
    for time_idx, label in enumerate(gt):
        if gt[time_idx]!=0 or pred[time_idx]!=0:
            overlap_sum +=1
            if gt[time_idx]==pred[time_idx]:
                gt_sum +=1
    if overlap_sum == 0:
        print 'This sequence don\'t have labels'
        return 0
    return gt_sum/float(overlap_sum)


def Extract_feature_normalized(smp, used_joints, startFrame, endFrame):
    """
    Extract normalized features
    """
    frame_num = 0 
    Skeleton_matrix  = numpy.zeros(shape=(endFrame-startFrame+1, len(used_joints)*3))
    normalized_joints = ['HipCenter', 'Spine', 'HipLeft', 'HipRight']
    HipCentre_matrix = numpy.zeros(shape=(endFrame-startFrame+1, len(normalized_joints)*3))

    for numFrame in range(startFrame,endFrame+1):                    
        # Get the Skeleton object for this frame
        skel=smp.getSkeleton(numFrame)
        for joints in range(len(used_joints)):
            Skeleton_matrix[frame_num, joints*3: (joints+1)*3] = skel.joins[used_joints[joints]][0]
        for joints in range(len(normalized_joints)):
            HipCentre_matrix[frame_num, joints*3: (joints+1)*3] = skel.joins[normalized_joints[joints]][0]             

        frame_num += 1

    xCentLst = HipCentre_matrix[:, range(0,10,3)]
    xCentVal = sum(sum(xCentLst)) / (xCentLst.shape[0]*xCentLst.shape[1])

    yCentLst = HipCentre_matrix[:, range(1,11,3)]
    yCentVal = sum(sum(yCentLst)) / (yCentLst.shape[0]*yCentLst.shape[1])

    zCentLst = HipCentre_matrix[:, range(2,12,3)]
    zCentVal = sum(sum(zCentLst)) / (zCentLst.shape[0]*zCentLst.shape[1])

    Skeleton_matrix[:, range(0,10,3)] = Skeleton_matrix[:, range(0,10,3)] - xCentVal
    Skeleton_matrix[:, range(1,11,3)] = Skeleton_matrix[:, range(1,11,3)] - yCentVal
    Skeleton_matrix[:, range(2,12,3)] = Skeleton_matrix[:, range(2,12,3)] - zCentVal

    xCentLst -= xCentVal
    yCentLst -= yCentVal
    zCentLst -= zCentVal

    coordHip = [xCentLst[:,0], yCentLst[:,0], zCentLst[:,0]]
    coordHip = numpy.mean(coordHip, axis=1)

    coordShou = [xCentLst[:,1], yCentLst[:,1], zCentLst[:,1]]
    coordShou = numpy.mean(coordShou, axis=1)

    scaleRatio = (sum(coordHip - coordShou)**2)**0.5

    Skeleton_matrix = Skeleton_matrix / scaleRatio

    if scaleRatio==0:
        valid_skel = False
    else:
        valid_skel = True
    return Skeleton_matrix, valid_skel

def Extract_feature_normalized_ALL(smp, used_joints, startFrame, endFrame):
    """
    Extract normalized features, but we replicate the first undetected frames as the 
    last detected frames
    """
    frame_num = 0 
    Skeleton_matrix  = numpy.zeros(shape=(endFrame-startFrame+1, len(used_joints)*3))
    normalized_joints = ['HipCenter', 'Spine', 'HipLeft', 'HipRight']
    HipCentre_matrix = numpy.zeros(shape=(endFrame-startFrame+1, len(normalized_joints)*3))


    Start_frame = 0
    ### first detect initial frames are valid:
    for numFrame in range(startFrame,endFrame):                    
    # Get the Skeleton object for this frame
        skel=smp.getSkeleton(numFrame)
        for joints in range(len(used_joints)):
            Skeleton_matrix[frame_num, joints*3: (joints+1)*3] = skel.joins[used_joints[joints]][0]
        if sum(Skeleton_matrix[frame_num, :])==0:
            Start_frame = numFrame
            skel=smp.getSkeleton(numFrame+1)
            Skeleton_matrix[frame_num, joints*3: (joints+1)*3] = skel.joins[used_joints[joints]][0]
            if sum(Skeleton_matrix[frame_num, :])!=0:
                break

    Take_Frame = endFrame
    while(1):
        skel=smp.getSkeleton(Take_Frame)
        Skeleton_matrix_temp  = numpy.zeros(shape=(1, len(used_joints)*3))
        for joints in range(len(used_joints)):
            Skeleton_matrix_temp[:, joints*3: (joints+1)*3] = skel.joins[used_joints[joints]][0]
        if sum(sum(Skeleton_matrix_temp))!=0:
                break
        else:
            Take_Frame -= 1
            print "missing frame"+str(Take_Frame)


    for numFrame in range(0,Start_frame):                    
        # Get the Skeleton object for this frame
        skel=smp.getSkeleton(Take_Frame)
        for joints in range(len(used_joints)):
            Skeleton_matrix[numFrame, joints*3: (joints+1)*3] = skel.joins[used_joints[joints]][0]
        for joints in range(len(normalized_joints)):
            HipCentre_matrix[numFrame, joints*3: (joints+1)*3] = skel.joins[normalized_joints[joints]][0]  



    for numFrame in range(Start_frame,endFrame):                    
        # Get the Skeleton object for this frame
        skel=smp.getSkeleton(numFrame+1)
        for joints in range(len(used_joints)):
            Skeleton_matrix[numFrame, joints*3: (joints+1)*3] = skel.joins[used_joints[joints]][0]
        for joints in range(len(normalized_joints)):
            HipCentre_matrix[numFrame, joints*3: (joints+1)*3] = skel.joins[normalized_joints[joints]][0]             

    xCentLst = HipCentre_matrix[:, range(0,10,3)]
    xCentVal = sum(sum(xCentLst)) / (xCentLst.shape[0]*xCentLst.shape[1])

    yCentLst = HipCentre_matrix[:, range(1,11,3)]
    yCentVal = sum(sum(yCentLst)) / (yCentLst.shape[0]*yCentLst.shape[1])

    zCentLst = HipCentre_matrix[:, range(2,12,3)]
    zCentVal = sum(sum(zCentLst)) / (zCentLst.shape[0]*zCentLst.shape[1])

    Skeleton_matrix[:, range(0,10,3)] = Skeleton_matrix[:, range(0,10,3)] - xCentVal
    Skeleton_matrix[:, range(1,11,3)] = Skeleton_matrix[:, range(1,11,3)] - yCentVal
    Skeleton_matrix[:, range(2,12,3)] = Skeleton_matrix[:, range(2,12,3)] - zCentVal

    xCentLst -= xCentVal
    yCentLst -= yCentVal
    zCentLst -= zCentVal

    coordHip = [xCentLst[:,0], yCentLst[:,0], zCentLst[:,0]]
    coordHip = numpy.mean(coordHip, axis=1)

    coordShou = [xCentLst[:,1], yCentLst[:,1], zCentLst[:,1]]
    coordShou = numpy.mean(coordShou, axis=1)

    scaleRatio = (sum(coordHip - coordShou)**2)**0.5

    Skeleton_matrix = Skeleton_matrix / scaleRatio

    if scaleRatio==0:
        valid_skel = False
    else:
        valid_skel = True
    return Skeleton_matrix, valid_skel


def Extract_feature(Pose, njoints):
    #Fcc
    FeatureNum = 0
    Fcc =  numpy.zeros(shape=(Pose.shape[0], njoints * (njoints-1)/2*3))
    for joints1 in range(njoints-1):
        for joints2 in range(joints1+1,njoints):
            Fcc[:, FeatureNum*3:(FeatureNum+1)*3] = Pose[:, joints1*3:(joints1+1)*3]-Pose[:, joints2*3:(joints2+1)*3];
            FeatureNum += 1
            
    #F_cp
    FeatureNum = 0
    Fcp = numpy.zeros(shape=(Pose.shape[0]-1, njoints **2*3))
    for joints1 in range(njoints):
        for joints2 in range(njoints):
            Fcp[:, FeatureNum*3: (FeatureNum+1)*3] = Pose[1:,joints1*3:(joints1+1)*3]-Pose[0:-1,joints2*3:(joints2+1)*3]
            FeatureNum += 1
              
    #Instead of initial frame as in the paper Eigenjoints-based action recognition using
    #naive-bayes-nearest-neighbor, we use final frame because it's better initiated
    # F_cf
    FeatureNum = 0
    Pose_final = numpy.tile(Pose [-1 , :] , [Pose.shape[0], 1])
    Fcf = numpy.zeros(shape=(Pose.shape[0]-1, njoints **2*3))
    for joints1 in range(njoints):
        for joints2 in range(njoints):
                Fcf[:, FeatureNum*3: (FeatureNum+1)*3] = Pose[0:-1, joints1*3:(joints1+1)*3] - Pose_final[0:-1,joints2*3:(joints2+1)*3]
                FeatureNum=FeatureNum+1

    Features = numpy.concatenate( (Fcc[0:-1, :], Fcp, Fcf), axis = 1)
    return Features
    
def Extract_feature_pva(Pose, njoints):
    Velocity = numpy.zeros(shape = (Pose.shape[0],3*njoints))
    a = Pose[1:]-Pose[0:-1]
    Velocity[1:] = a    
    
    Acc = numpy.zeros(shape = (Pose.shape[0],3*njoints))
    b = a[1:]-a[:-1]
    Acc[2:] = b
    Features = numpy.concatenate((Pose,Velocity,Acc),axis = 1)
    return Features
    
def Extract_feature_Realtime(Pose, njoints):
    #Fcc
    FeatureNum = 0
    Fcc =  numpy.zeros(shape=(Pose.shape[0], njoints * (njoints-1)/2*3))
    for joints1 in range(njoints-1):
        for joints2 in range(joints1+1,njoints):
            Fcc[:, FeatureNum*3:(FeatureNum+1)*3] = Pose[:, joints1*3:(joints1+1)*3]-Pose[:, joints2*3:(joints2+1)*3];
            FeatureNum += 1
            
    #F_cp
    FeatureNum = 0
    Fcp = numpy.zeros(shape=(Pose.shape[0]-1, njoints **2*3))
    for joints1 in range(njoints):
        for joints2 in range(njoints):
            Fcp[:, FeatureNum*3: (FeatureNum+1)*3] = Pose[1:,joints1*3:(joints1+1)*3]-Pose[0:-1,joints2*3:(joints2+1)*3]
            FeatureNum += 1
              
    #Instead of initial frame as in the paper Eigenjoints-based action recognition using
    #naive-bayes-nearest-neighbor, we use final frame because it's better initiated
    # F_cf

    Features = numpy.concatenate( (Fcc[0:-1, :], Fcp), axis = 1)
    return Features

def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def zero_mean_unit_variance(Data):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    Mean = numpy.mean(Data, axis=0)
    Data  -=  Mean

    Std = numpy.std(Data, axis = 0)
    index = (numpy.abs(Std<10**-5))
    Std[index] = 1
    Data /= Std
    return [Data, Mean, Std]


def normalize(Data, Mean, Std):
    Data -= Mean
    Data /= Std
    return Data

class DataLoader():
    def __init__(self, src, batch_size):
        self.batch_size = batch_size
        import h5py
        file = h5py.File(src+"/data%d.hdf5", "r", driver="family", memb_size=2**32-1)
        self.x_train = file["x_train"]
        self.x_valid = file["x_valid"]
        self.y_train = file["y_train"]
        self.y_valid = file["y_valid"]

        self.n_iter_train = int(numpy.floor(self.x_train.shape[0]/float(batch_size)))
        self.n_iter_valid = int(numpy.floor(self.x_valid.shape[0]/float(batch_size)))

        self.shuffle_train()
        self.shuffle_valid()

    def next_train_batch(self, x_, y_):
        if len(self.pos_train) == 0: self.shuffle_train()
        pos = self.pos_train.pop()
        x_.set_value(self.x_train[pos:pos+self.batch_size] , borrow=True)
        y_.set_value(self.y_train[pos:pos+self.batch_size] , borrow=True)

    def next_valid_batch(self, x_, y_):
        if len(self.pos_valid) == 0: self.shuffle_valid()
        pos = self.pos_valid.pop()
        x_.set_value(self.x_valid[pos:pos+self.batch_size] , borrow=True)
        y_.set_value(self.y_valid[pos:pos+self.batch_size] , borrow=True)

    def shuffle_train(self):
        self.pos_train = list(random.permutation(self.n_iter_train)*self.batch_size)

    def shuffle_valid(self):
        self.pos_valid = list(random.permutation(self.n_iter_valid)*self.batch_size)



class DataLoader_with_skeleton_normalisation():
    def __init__(self, src, batch_size, Mean_CNN=0, Std_CNN=1, Mean1=0, Std1=1, load_path=""):
        self.batch_size = batch_size
        import h5py
        import os
        file = h5py.File(src+"/data%d.hdf5", "r", driver="family", memb_size=2**32-1)
        self.x_train = file["x_train"]
        self.x_valid = file["x_valid"]
        self.y_train = file["y_train"]
        self.y_valid = file["y_valid"]

        # we used only the first 1000 frames
        #from dbn.utils import zero_mean_unit_variance
        #[train_set_feature_normalized, Mean_CNN, Std_CNN]  = zero_mean_unit_variance(self.x_train[:5000,:] / 255.)
        #### we need to load the pre-store normalization constant
        #import cPickle as pickle
        #f = open('CNN_normalization.pkl','wb')
        #pickle.dump( {"Mean_CNN": Mean_CNN, "Std_CNN": Std_CNN },f)
        #f.close()
        self.Mean_CNN = Mean_CNN
        self.Std_CNN = Std_CNN
        self.Mean1 = Mean1
        self.Std1 = Std1
        self.x_train_skeleton_feature = file["x_train_skeleton_feature"]
        self.x_valid_skeleton_feature = file["x_valid_skeleton_feature"]

        self.n_iter_train = int(floor(self.x_train.shape[0]/float(batch_size)))
        self.n_iter_valid = int(floor(self.x_valid.shape[0]/float(batch_size)))

        self.shuffle_train()
        self.shuffle_valid()

    def next_train_batch(self, x_, y_, x_skeleton_):
        if len(self.pos_train) == 0: self.shuffle_train()
        pos = self.pos_train.pop()
        x_.set_value(normalize(self.x_train[pos:pos+self.batch_size], self.Mean_CNN, self.Std_CNN) , borrow=True)
        y_.set_value(self.y_train[pos:pos+self.batch_size] , borrow=True)
        x_skeleton_.set_value( normalize(self.x_train_skeleton_feature[pos:pos+self.batch_size], self.Mean1, self.Std1), borrow=True)


    def next_valid_batch(self, x_, y_, x_skeleton_):
        if len(self.pos_valid) == 0: self.shuffle_valid()
        pos = self.pos_valid.pop()
        x_.set_value(normalize(self.x_valid[pos:pos+self.batch_size], self.Mean_CNN, self.Std_CNN) , borrow=True)
        y_.set_value(self.y_valid[pos:pos+self.batch_size] , borrow=True)
        x_skeleton_.set_value( normalize(self.x_valid_skeleton_feature[pos:pos+self.batch_size], self.Mean1, self.Std1) , borrow=True)

    def shuffle_train(self):
        self.pos_train = list(random.permutation(self.n_iter_train)*self.batch_size)

    def shuffle_valid(self):
        self.pos_valid = list(random.permutation(self.n_iter_valid)*self.batch_size)
def viterbi_path(prior, transmat, observ_likelihood):
    """ Viterbi path decoding
    Wudi first implement the forward pass.
    Future works include forward-backward encoding
    input: prior probability 1*N...
    transmat: N*N
    observ_likelihood: N*T
    """
    T = observ_likelihood.shape[-1]
    N = observ_likelihood.shape[0]

    path = numpy.zeros(T, dtype=numpy.int32)
    global_score = numpy.zeros(shape=(N,T))
    predecessor_state_index = numpy.zeros(shape=(N,T), dtype=numpy.int32)

    t = 1
    global_score[:, 0] = prior * observ_likelihood[:, 0]
    # need to  normalize the data
    global_score[:, 0] = global_score[:, 0] /sum(global_score[:, 0] )
    
    for t in range(1, T):
        for j in range(N):
            temp = global_score[:, t-1] * transmat[:, j] * observ_likelihood[j, t]
            global_score[j, t] = max(temp)
            predecessor_state_index[j, t] = temp.argmax()

        global_score[:, t] = global_score[:, t] / sum(global_score[:, t])

    path[T-1] = global_score[:, T-1].argmax()
    
    for t in range(T-2, -1, -1):
        path[t] = predecessor_state_index[ path[t+1], t+1]

    return [path, predecessor_state_index, global_score]


def viterbi_path_log(prior, transmat, observ_likelihood):
    """ Viterbi path decoding
    Wudi first implement the forward pass.
    Future works include forward-backward encoding
    input: prior probability 1*N...
    transmat: N*N
    observ_likelihood: N*T
    """
    T = observ_likelihood.shape[-1]
    N = observ_likelihood.shape[0]

    path = numpy.zeros(T, dtype=numpy.int32)
    global_score = numpy.zeros(shape=(N,T))
    predecessor_state_index = numpy.zeros(shape=(N,T), dtype=numpy.int32)

    t = 1
    global_score[:, 0] = prior + observ_likelihood[:, 0]
    # need to  normalize the data
    
    for t in range(1, T):
        for j in range(N):
            temp = global_score[:, t-1] + transmat[:, j] + observ_likelihood[j, t]
            global_score[j, t] = max(temp)
            predecessor_state_index[j, t] = temp.argmax()

    path[T-1] = global_score[:, T-1].argmax()
    
    for t in range(T-2, -1, -1):
        path[t] = predecessor_state_index[ path[t+1], t+1]

    return [path, predecessor_state_index, global_score]

def createSubmisionFile(predictionsPath,submisionPath):
    """ Create the submission file, ready to be submited to Codalab. """
    import os, zipfile
    # Create the output path and remove any old file
    if os.path.exists(submisionPath):
        oldFileList = os.listdir(submisionPath);
        for file in oldFileList:
            os.remove(os.path.join(submisionPath,file));
    else:
        os.makedirs(submisionPath);

    # Create a ZIP with all files in the predictions path
    zipf = zipfile.ZipFile(os.path.join(submisionPath,'Submission.zip'), 'w');
    for root, dirs, files in os.walk(predictionsPath):
        for file in files:
            zipf.write(os.path.join(root, file), file, zipfile.ZIP_DEFLATED);
    zipf.close()