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
os.chdir('/home/cdh/video_SCL/Gamania/Gamania/Script/')
#WORKDIR = '../Script/VideoFeature/Combine/'
WORKDIR = '/home/cdh/video_SCL/Disk_tmp/Lab/Gamania/Script/VideoFeatureNewCut/Combine/'
ROOT = os.getcwd()
LABELTYPE = ['Act']
CTYPE = [ 0.001]
#%%
feaAudio_lab = []
feaVideo_lab = []
feaAudio = collections.defaultdict(list)
feaVideo = collections.defaultdict(list)
for label in LABELTYPE:
    feaAudio_tmp = ib.load(WORKDIR + label+ '_feaAudio.pkl')
    data = feaAudio
    feaVideo_tmp = ib.load(WORKDIR + label+ '_feaVideo.pkl') 
    for i, n in enumerate(sorted(feaAudio_tmp.keys())):
    	data = np.array(feaAudio_tmp[n])
    	data2 = np.array(feaVideo_tmp[n])
    	feaAudio_lab = (data[:, -2].tolist())
    	zero_index = [ii for ii, lab in enumerate(feaAudio_lab) if lab == 0]
    	others_index = [ii for ii, lab in enumerate(feaAudio_lab) if lab == 1]
    	np.random.seed(i)
    	keep_index = np.random.permutation(zero_index)[:int(len(zero_index)*0.45)]
    	feaAudio[n].extend(data[keep_index, :].tolist())
    	feaAudio[n].extend(data[others_index, :].tolist())
    	feaVideo[n].extend(data2[keep_index, :].tolist())
    	feaVideo[n].extend(data2[others_index,:].tolist())

xx = 60
yy = 70
#yy = 80
#yy2 = 70
parameters = collections.defaultdict()
for C_number in CTYPE:           
    for xx2 in [40]:
#            for yy in range(10, 110, 10):
        for yy2 in [90]:
            tru=[]
            pree=[]
            #for n in feaAudio.keys():
            for n in  ['0524']:
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
                parameters['mean_1'] = mean
                parameters['std_1'] =std    

                #2nd train data pre-processing    
                [train_data2nd_v, mean2nd, std2nd] = zsco(np.array(train_data2nd_v), 0)
                train_data2nd_v = np.nan_to_num(train_data2nd_v)
                parameters['mean_2'] = mean2nd
                parameters['std_2'] = std2nd
                '''
                data = np.array(feaAudio[n])
                data2= np.array(feaVideo[n])
                test_data_v = data2[:, :-2].tolist()
                test_data_a = data[:, :-2].tolist()  
                test_data_v = passzsco(np.array(test_data_v), 0, mean, std)
                test_data_v = np.nan_to_num(test_data_v)
                test_label= data2[:, -2].tolist()
                '''                  
                
                
                ps_v = SelectPercentile(f_classif, percentile = xx)
                train_data_v = ps_v.fit_transform(train_data_v, train_label)
                parameters['Percentile_v1'] = ps_v
                '''
                test_data_v = ps_v.transform(test_data_v)
                '''

                #yy = 100
                ps_a = SelectPercentile(f_classif, percentile = yy)
                train_data_a = ps_a.fit_transform(np.array(train_data_a), train_label)
                parameters['Percentile_a1'] = ps_a
                '''
                test_data_a = ps_a.transform(test_data_a)
                '''
                train_data = np.dot(train_data_v, train_data_v.T)+np.dot(train_data_a, train_data_a.T)
                parameters['train_data_v'] = train_data_v
                parameters['train_data_a'] = train_data_a
                '''
                test_data = np.dot(test_data_v, train_data_v.T)+np.dot(test_data_a, train_data_a.T)
                '''
                clf=SVC(kernel='precomputed',class_weight='balanced',C=C_number)
                clf.fit(train_data, train_label)
                parameters['Classifier_1'] = clf
                '''
                pre=clf.predict(test_data)

                #take out the index of prediction are 1
                proceed_index = [ii for ii, pre_num in enumerate(pre) if pre_num == 1]
                stay_index = [ii for ii, pre_num in enumerate(pre) if pre_num == 0]
                tru.extend(list(data2[stay_index, -1]))
                pree.extend(list(np.array(pre)[stay_index]))                    

                #2nd test data collect
                test_data2nd_v = data2[proceed_index, :-2].tolist()   
                test_data2nd_a = data[proceed_index, :-2].tolist()
                test_data2nd_v = passzsco(np.array(test_data2nd_v), 0, mean2nd, std2nd)
                test_data2nd_v = np.nan_to_num(test_data2nd_v)
                test_label2nd = data2[proceed_index, -1].tolist()
                '''
                ps_v2nd = SelectPercentile(f_classif, percentile = xx2)
                train_data2nd_v = ps_v2nd.fit_transform(train_data2nd_v, train_label2nd)
                parameters['Percentile_v2nd'] = ps_v2nd
                '''
                test_data2nd_v = ps_v2nd.transform(test_data2nd_v)
                '''
                #yy2 = 90
                ps_a2nd = SelectPercentile(f_classif, percentile = yy2)
                train_data2nd_a = ps_a2nd.fit_transform(np.array(train_data2nd_a), train_label2nd)
                parameters['Percentile_a2nd'] = ps_a2nd
                '''
                test_data2nd_a = ps_a2nd.transform(test_data2nd_a)
                '''
                train_data2nd = np.dot(train_data2nd_v, train_data2nd_v.T)+np.dot(train_data2nd_a, train_data2nd_a.T)
                parameters['train_data_v2nd'] = train_data2nd_v
                parameters['train_data_a2nd'] = train_data2nd_a
                '''
                test_data2nd = np.dot(test_data2nd_v, train_data2nd_v.T)+np.dot(test_data2nd_a, train_data2nd_a.T)
                '''
                clf=SVC(kernel='precomputed',class_weight='balanced',C=C_number)
                clf.fit(train_data2nd, train_label2nd)
                parameters['Classifier_2'] = clf
                '''
                pre=clf.predict(test_data2nd)  
                tru.extend(test_label2nd)
                pree.extend(pre)  
                '''
                ib.dump(parameters, WORKDIR+label+'_param.pkl')
            '''       
            print(cm(tru,pree))
        #        print(UAR(tru,pree))
            accu = np.int(UAR(tru,pree, average='macro')*1000)/10.0
            accu2 = np.int(UAR(tru,pree, average='weighted')*1000)/10.0
            print('Label:',label,'C:',C_number,'per_v:',xx,'per_a:', yy,'per_v2:',xx2,'per_a2:', yy2, 'UAR:', accu)
            print('Label:',label,'C:',C_number,'per_v:',xx,'per_a:', yy,'per_v2:',xx2,'per_a2:', yy2, 'Accu:',accu2)
            '''            

