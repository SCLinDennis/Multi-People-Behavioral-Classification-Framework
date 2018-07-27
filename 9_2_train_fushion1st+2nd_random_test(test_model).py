# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 14:16:03 2017

@author: dennis60512
"""

import os, glob, sys
import multiprocessing  as mp
import pandas as pd
import collections
import joblib as ib
from collections import defaultdict
import numpy as np
from sklearn.svm import SVC
from scipy.stats import spearmanr
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectPercentile
from sklearn.ensemble import RandomForestClassifier as RFC
from scipy import stats
import scipy as sp
from sklearn.metrics import recall_score as UAR
from sklearn.metrics import confusion_matrix as cm
from numpy import matlib as matlib
import warnings
#%%
def zsco(arg,axis):
    lenn,widd=arg.shape
    uper=(arg-matlib.repmat(np.mean(arg,axis),lenn,1)) 
    lower=matlib.repmat((np.std(arg,axis)),lenn,1)
    return uper/lower,np.mean(arg,axis),np.std(arg,axis)
def passzsco(arg,axis,mean,std):
    lenn,widd=arg.shape
    uper=(arg-matlib.repmat(mean,lenn,1))    
    lower=matlib.repmat(std,lenn,1)
    return uper/lower


#%%
os.chdir('/home/cdh/video_SCL/Disk_tmp/Lab/Gamania/Script/')
#WORKDIR = '../Script/VideoFeature/Combine/'
WORKDIR = './VideoFeatureTest/Combine/'
PARAMDIR = '/home/cdh/video_SCL/Disk_tmp/Lab/Gamania/Script/VideoFeatureNewCut/Combine/'
ROOT = os.getcwd()
LABELTYPE = ['Act']
CTYPE = [ 0.001]
#%%
feaAudio_lab = []
feaVideo_lab = []
feaAudio = collections.defaultdict(list)
feaVideo = collections.defaultdict(list)
for label in LABELTYPE:
    feaAudio = ib.load(WORKDIR + label+ '_feaAudio.pkl')
    feaVideo = ib.load(WORKDIR + label+ '_feaVideo.pkl') 
    parameters = ib.load(PARAMDIR + label + '_param.pkl')
    fesLength2 = ib.load(WORKDIR + label + '_fesLength2.pkl')

xx = 80
yy = 100
#yy = 80
#yy2 = 70
final = collections.defaultdict(list)
for C_number in CTYPE:           
    for xx2 in [30]:
#            for yy in range(10, 110, 10):
        for yy2 in [70]:
            tru=[]
            pree=[]
            for n in feaAudio.keys(): 
                fesLength1st = collections.defaultdict(list)
                fesLength2nd = collections.defaultdict(list)
                '''
                tralis=list(set(feaAudio.keys())-{n})
                train_data_v=[]
                train_data_a=[]
                train_label=[]
                train_data =[]
                train_data2nd_v=[]
                train_data2nd_a=[]
                train_label2nd=[]
                train_data2nd =[]
                for i in tralis:
                    #collect 1st train_data
                    data = np.array(feaAudio[i])
                    data2= np.array(feaVideo[i])
                    train_data_a.extend(data[:,:-2].tolist())
                    train_data_v.extend(data2[:, :-2].tolist())
                    train_label.extend(data2[:, -2].tolist())

                    #collect 2nd train_data
                    index2 = np.where(np.array(feaVideo[i])[:, -2] != 0)[0]
                    data = np.array(feaAudio[i])[index2]
                    data2= np.array(feaVideo[i])[index2]
                    train_data2nd_a.extend(data[:,:-2].tolist())
                    train_data2nd_v.extend(data2[:, :-2].tolist())
                    train_label2nd.extend(data2[:, -1].tolist())    
                
                #1st train data pre-processing
                [train_data_v, mean, std] = zsco(np.array(train_data_v), 0) 
                train_data_v = np.nan_to_num(train_data_v)       

                #2nd train data pre-processing    
                [train_data2nd_v, mean2nd, std2nd] = zsco(np.array(train_data2nd_v), 0)
                train_data2nd_v = np.nan_to_num(train_data2nd_v)
                '''
                data = np.array(feaAudio[n])
                data2= np.array(feaVideo[n])
                test_data_v = data2[:, :-2].tolist()
                test_data_a = data[:, :-2].tolist()  
                test_data_v = passzsco(np.array(test_data_v), 0, parameters['mean_1'], parameters['std_1'])
                test_data_v = np.nan_to_num(test_data_v)
                test_label= data2[:, -2].tolist()
                                  
                
                '''
                ps_v = SelectPercentile(f_classif, percentile = xx)
                train_data_v = ps_v.fit_transform(train_data_v, train_label)
                '''
                ps_v = parameters['Percentile_v1']
                test_data_v = ps_v.transform(test_data_v)
                #yy = 100
                '''
                ps_a = SelectPercentile(f_classif, percentile = yy)
                train_data_a = ps_a.fit_transform(np.array(train_data_a), train_label)
                '''
                ps_a = parameters['Percentile_a1']
                test_data_a = ps_a.transform(test_data_a)
                '''
                train_data = np.dot(train_data_v, train_data_v.T)+np.dot(train_data_a, train_data_a.T)
                '''
                test_data = np.dot(test_data_v, parameters['train_data_v'].T)+np.dot(test_data_a, parameters['train_data_a'].T)
                '''
                clf=SVC(kernel='precomputed',class_weight='balanced',C=C_number)
                clf.fit(train_data, train_label)
                '''
                clf = parameters['Classifier_1'] 
                pre=clf.predict(test_data)

                #take out the index of prediction are 1
                proceed_index = [ii for ii, pre_num in enumerate(pre) if pre_num == 1]
                stay_index = [ii for ii, pre_num in enumerate(pre) if pre_num == 0]
                for kk in fesLength2.keys():
                    if (kk[0:-1]) == n:
                        proceed_num = 0
                        stay_num = 0
                        for i2, pre_num in enumerate(pre):
                            if i2 == fesLength2[kk][0]:
                                fesLength1st[kk].append(stay_num)
                                fesLength2nd[kk].append(proceed_num)

                            if pre_num == 0:
                                stay_num += 1
                            elif pre_num == 1:
                                proceed_num += 1

                            if i2 == fesLength2[kk][1]-1:
                                fesLength1st[kk].append(stay_num)
                                fesLength2nd[kk].append(proceed_num)
                '''
                tru.extend(list(data2[stay_index, -1]))
                '''
                pree.extend(list(np.array(pre)[stay_index]))                    

                #2nd test data collect
                test_data2nd_v = data2[proceed_index, :-2].tolist()   
                test_data2nd_a = data[proceed_index, :-2].tolist()
                test_data2nd_v = passzsco(np.array(test_data2nd_v), 0, parameters['mean_2'], parameters['std_2'] )
                test_data2nd_v = np.nan_to_num(test_data2nd_v)
                test_label2nd = data2[proceed_index, -1].tolist()
                '''
                ps_v2nd = SelectPercentile(f_classif, percentile = xx2)
                train_data2nd_v = ps_v2nd.fit_transform(train_data2nd_v, train_label2nd)
                '''
                ps_v2nd = parameters['Percentile_v2nd']
                test_data2nd_v = ps_v2nd.transform(test_data2nd_v)
                #yy2 = 90
                '''
                ps_a2nd = SelectPercentile(f_classif, percentile = yy2)
                train_data2nd_a = ps_a2nd.fit_transform(np.array(train_data2nd_a), train_label2nd)
                '''
                ps_a2nd = parameters['Percentile_a2nd']
                test_data2nd_a = ps_a2nd.transform(test_data2nd_a)
                '''
                train_data2nd = np.dot(train_data2nd_v, train_data2nd_v.T)+np.dot(train_data2nd_a, train_data2nd_a.T)
                '''
                test_data2nd = np.dot(test_data2nd_v, parameters['train_data_v2nd'].T)+np.dot(test_data2nd_a, parameters['train_data_a2nd'].T)
                '''
                clf=SVC(kernel='precomputed',class_weight='balanced',C=C_number)
                clf.fit(train_data2nd, train_label2nd)
                '''
                clf = parameters['Classifier_2']
                pre2nd=clf.predict(test_data2nd)  
                '''
                tru.extend(test_label2nd)
                '''
                pree.extend(pre2nd)  
                for jj in fesLength2.keys():
                    if (jj[0:-1]) == n:
                        trueone = list(data2[fesLength2[jj][0]:fesLength2[jj][1], -1]).count(1) + list(data2[fesLength2[jj][0]:fesLength2[jj][1], -1]).count(2)*2
                        preone = list(pre2nd[fesLength2nd[jj][0]:fesLength2nd[jj][1]]).count(1) + list(pre2nd[fesLength2nd[jj][0]:fesLength2nd[jj][1]]).count(2)*2
                        testall = trueone + list(data2[fesLength2[jj][0]:fesLength2[jj][1], -2]).count(0)
                        final[jj] = [trueone, preone, testall]
ib.dump(final, WORKDIR+label+'_final.pkl')
'''       
            print(cm(tru,pree))
        #        print(UAR(tru,pree))
            accu = np.int(UAR(tru,pree, average='macro')*1000)/10.0
            accu2 = np.int(UAR(tru,pree, average='weighted')*1000)/10.0
            print('Label:',label,'C:',C_number,'per_v:',xx,'per_a:', yy,'per_v2:',xx2,'per_a2:', yy2, 'UAR:', accu)
            print('Label:',label,'C:',C_number,'per_v:',xx,'per_a:', yy,'per_v2:',xx2,'per_a2:', yy2, 'Accu:',accu2)
'''

