# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 10:41:10 2017

@author: dennis60512
"""

import os, glob, sys
import subprocess as sp
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
import scipy
from sklearn.metrics import recall_score as UAR
from sklearn.metrics import confusion_matrix as cm
from numpy import matlib as matlib
import operator
from tqdm import tqdm
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

def score_analyse(score, limit):
    score_statistic = collections.defaultdict(int)
    score_array = np.zeros([score.shape[0], limit])
    feature_names = []
    for i in range(score.shape[1]):
        feature_names.append(str(i))    
    for y in range(score.shape[0]):
        idx = 0
        for k in np.argsort(score[y])[::-1]:
            if idx < limit:
                score_array[y, idx] =  feature_names[k]
                idx += 1
                if feature_names[k] in score_statistic.keys():
                    score_statistic[feature_names[k]] += 1
                else:
                    score_statistic[feature_names[k]] == 0
#    score_out = sorted(score_statistic.items(), key=operator.itemgetter(1), reverse = True)
    return score_statistic

def functional_name(sun):
    num = sun%15
    num2 = sun//15
    if num == 1:
        name = 'max'
    elif num == 2:
        name = 'min'
    elif num == 3:
        name = 'mean'
    elif num == 4:
        name = 'median'
    elif num == 5:
        name = 'std'
    elif num == 6:
        name = 'first_percentile'
    elif num == 7:
        name = 'last_percentile'
    elif num == 8:
        name = 'range_percentile'
    elif num == 9:
        name = 'skewness'
    elif num == 10:
        name = 'kurtosis'
    elif num == 11:
        name = 'minimun_position'
    elif num == 12:
        name = 'maximun_position'
    elif num == 13:
        name = 'lower_quartile'
    elif num == 14:
        name = 'upper_quartile'
    elif num == 0:
        name = 'interquartile_range'
    if num2 != 0:
        name = name + str(num2)
    return name

def deep_analyse(score_statistic):
    deep_score = collections.defaultdict(int)
    for i in score_statistic.keys():
        if int(i) <= (58*2*15):
            sun = ((int(i)+1)//116)+1
            fish = (int(i)+1)%116
            name = functional_name(sun) 
            if fish == 0:
                ty = 'gaze_others_'
            elif fish > 0 and fish <= 6:
                ty = 'pose_self_' + str(fish)
            elif fish > 6 and fish <= 46:
                ty = 'param_self_' + str(fish-6)
            elif fish > 46 and fish <= 58:
                ty = 'gaze_self_' + str(fish - 46)
            elif fish > 58 and fish <= 64:
                ty = 'pose_others_' + str(fish-58)
            elif fish > 64 and fish <= 104:
                ty = 'param_others_' + str(fish-64)
            elif fish > 104 and fish <= 115:
                ty = 'gaze_others_' + str(fish - 104)
        
        elif int(i) > (58*2*15):
            i2 = int(i) - (58*2*15)
            sun = ((i2+1)//58)+1
            fish = (i2+1)%58
            name = functional_name(sun) 
            if fish == 0:
                ty = 'gaze_delta_'
            elif fish > 0 and fish <= 6:
                ty = 'pose_delta_' + str(fish)
            elif fish > 6 and fish <= 46:
                ty = 'param_delta_' + str(fish-6)
            elif fish > 46 and fish <= 57:
                ty = 'gaze_delta_' + str(fish - 46)            
        deep_score[ty+name] = score_statistic[i]
    return deep_score        
def deep_analyse_au(score_statistic):
    deep_score = collections.defaultdict(int)
    for i in score_statistic.keys():
        if int(i) <= (70*15):
            sun = ((int(i)+1)//70)+1
            fish = (int(i)+1)%70
            name = functional_name(sun)
            if fish == 0:
                ty = 'presenceAU45_others_'
            elif fish > 0 and fish <= 17:
                ty = 'Intensity_self_' + str(fish)
            elif fish > 17 and fish <= 35:
                ty = 'Presence_self_' + str(fish-17)
            elif fish > 35 and fish <= 51:
                ty = 'Intensity_other_' + str(fish-35)
            elif fish > 51 and fish <= 69:
                ty = 'Presence_others_' + str(fish- 51)
        elif int(i) > (70*15):
            i2 = int(i) -(70*15)
            sun = ((int(i2)+1)//35)+1
            fish = (int(i2)+1)%35 
            name = functional_name(sun)
            if fish == 0:
                ty = 'AU45presence_delta_'
            elif fish > 0 and fish <= 17:
                ty = 'Intensity_delta_' + str(fish)
            elif fish > 17 and fish <= 34:
                ty = 'Presence_delta' + str(fish-17)
        deep_score[ty+name] = score_statistic[i]
    return deep_score   

def score_initial(fea_Com):
    length = len(fea_Com.keys())
    check = 0
    for key in fea_Com.keys():  
        if check == 0:     
            width = len(fea_Com[key][0])
            check += 1
    score= np.zeros([length, width-1])
    return score    
#%%
os.chdir('D:\\Lab\\Dennis\\Gamania\\Script\\')
WORKDIR = '.\\VideoFeature\\Pose\\'
#WORKDIR = '..\\Script\\VideoFeature\\Info_traj\\'
#WORKDIR = '..\\Script\\VideoFeature\\NewFeature3\\Pos_param_gaze\\Sum\\'
A = ['0609','0613','0626_1','0628','0630',  '0728_1']
B= ['0703_1', '0703_2', '0705','0706_1','0706_2', '0728_2']
C = ['0707','0711_1','0711_2','0712_1','0712_2', '0728_3']
D =['0712_3','0713_1','0713_2','0714_1','0714_2']
E =['0718','0719_1','0719_2','0719_3','0720_1']
F=['0720_3','0721_1','0721_2','0721_3','0724_1']
G = ['0724_2','0725_2','0726','0727_2','0727_3']
test_name= [A, B, C, D, E, F, G]
ROOT = os.getcwd()
LABELTYPE = ['Act']
CTYPE = [  0.001]
#%%
for label in LABELTYPE:
    feaVideo = ib.load(WORKDIR + label+ '_feaVideo.pkl')
    feaCom = feaVideo
    for C_number in CTYPE:              
        for xx in range(10,110,10):
#        for xx in [50]:
            tru=[]
            pree=[]
#            score= score_initial(feaCom)
            for idx, n in tqdm(enumerate(feaCom.keys())):
                if 1!= 0 :
                    tralis=list(set(feaCom.keys())-{n})
                    train_data=[]
                    train_label=[]    
                    for i in tralis:
                        if int(i.split('_')[0])>0:
                            data2=np.array(feaCom[i])
                            train_data.extend(data2[:, :-1].tolist())
                            train_label.extend(data2[:, -1].tolist())   
                    [train_data, mean, std] = zsco(np.array(train_data), 0)
                    train_data = np.nan_to_num(train_data)
                    data1=np.array(feaCom[n])
                    test_data = data1[:, :-1].tolist()
                    test_label= data1[:, -1].tolist()
                    test_data = passzsco(np.array(test_data), 0, mean, std)
                    test_data = np.nan_to_num(test_data)
                    ps = SelectPercentile(f_classif, percentile = xx)
                    
                    train_data = ps.fit_transform(train_data, train_label)
                    test_data=ps.transform(test_data)
#                    score[idx] = ps.scores_
                    clf=SVC(kernel='linear',class_weight='balanced',C=C_number)
                    clf.fit(train_data,train_label)
                    pre=clf.predict(test_data)
                    tru.extend(test_label)
                    pree.extend(pre)
                   
#            print(cm(tru,pree))
            accu = np.int(UAR(tru,pree, average='macro')*1000)/10.0
            print('Label:',label,'C:',C_number,'per:',xx,'Accu:',accu)
#score_statistic = score_analyse(score, 630)
#new_score = deep_analyse_au(score_statistic)
#sort_score = sorted(new_score.items(), key=operator.itemgetter(1), reverse = True)      

     
