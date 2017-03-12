# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 16:03:40 2016

@author: zhangzhi
"""

import utils
import numpy as np
import matplotlib.pyplot as plt
a = np.load('confusionMatrix.npy')

name_list = ['non-gesture','vattene', 'vieniqui', 'perfetto', 'furbo', 'cheduepalle', 
             'chevuoi', 'daccordo', 'seipazzo', 'combinato',
             'freganiente', 'ok', 'cosatifarei', 'basta', 'prendere', 'noncenepiu',
             'fame', 'tantotempo', 'buonissimo', 'messidaccordo', 'sonostufo']

utils.plot_confusion_matrix(a, name_list,
                          normalize=True,
                          title='',
                          cmap=plt.cm.Blues)