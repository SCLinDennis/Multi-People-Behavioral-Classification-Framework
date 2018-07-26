# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 14:13:54 2017

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
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from tqdm import tqdm
import operator

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

def score_analyse(score, percentile):
    score_statistic = collections.defaultdict(int)
    limit = int(score.shape[1]*percentile/100)
    score_array = np.zeros([score.shape[0], limit])
    feature_names = []
    dim = []
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
                    score_statistic[feature_names[k]] = 1
#    score_out = sorted(score_statistic.items(), key=operator.itemgetter(1), reverse = True)
    for key, value in score_statistic.items():
        if value > 10:
            dim.append(int(key))
    
    return score_statistic, sorted(dim)
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
    score_accum = collections.defaultdict(int)
    score_accum2 = collections.defaultdict(int)
    score_accum3 = collections.defaultdict(int)
    for i in score_statistic.keys():
        if int(i) <= (58*2*15):
            sun = ((int(i)+1)//116)+1
            fish = (int(i)+1)%116
            name = functional_name(sun) 
            if fish == 0:
                ty = 'gaze_delta_'
                score_accum2['delta'] += score_statistic[i]
                score_accum3['gaze'] += score_statistic[i]
            elif fish > 0 and fish <= 6:
                ty = 'pose_self_' + str(fish)
                score_accum2['self'] += score_statistic[i]
                score_accum3['pose'] += score_statistic[i]
            elif fish > 6 and fish <= 46:
                ty = 'param_self_' + str(fish - 6)
                score_accum2['self'] += score_statistic[i]
                score_accum3['param'] += score_statistic[i]
            elif fish > 46 and fish <= 58:
                ty = 'gaze_self_' + str(fish - 46)
                score_accum2['self'] += score_statistic[i]
                score_accum3['gaze'] += score_statistic[i]
            elif fish > 58 and fish <= 64:
                ty = 'pose_delta_' + str(fish - 58)
                score_accum2['delta'] += score_statistic[i]
                score_accum3['pose'] += score_statistic[i]
            elif fish > 64 and fish <= 104:
                ty = 'param_delta_' + str(fish - 64)
                score_accum2['delta'] += score_statistic[i]
                score_accum3['param'] += score_statistic[i]
            elif fish > 104 and fish <= 115:
                ty = 'gaze_delta_' + str(fish - 104)
                score_accum2['delta'] += score_statistic[i]
                score_accum3['gaze'] += score_statistic[i]
        
        elif int(i) > (58*2*15):
            i2 = int(i) - (58*2*15)
            sun = ((i2+1)//58)+1
            fish = (i2+1)%58
            name = functional_name(sun) 
            if fish == 0:
                ty = 'gaze_others_'
                score_accum2['others'] += score_statistic[i]
                score_accum3['gaze'] += score_statistic[i]
            elif fish > 0 and fish <= 6:
                ty = 'pose_others_' + str(fish)
                score_accum2['others'] += score_statistic[i]
                score_accum3['pose'] += score_statistic[i]
            elif fish > 6 and fish <= 46:
                ty = 'param_others_' + str(fish - 6)
                score_accum2['others'] += score_statistic[i]
                score_accum3['param'] += score_statistic[i]
            elif fish > 46 and fish <= 57:
                ty = 'gaze_others_' + str(fish - 46)
                score_accum2['others'] += score_statistic[i]
                score_accum3['gaze'] += score_statistic[i]
        score_accum[ty] += score_statistic[i]
        deep_score[ty+name] = score_statistic[i]
    return score_accum, score_accum2, score_accum3, deep_score        

def deep_analyse_au(score_statistic):
    deep_score = collections.defaultdict(int)
    score_accum = collections.defaultdict(int)
    score_accum2 = collections.defaultdict(int)
    score_accum3 = collections.defaultdict(int)    
    for i in score_statistic.keys():
        if int(i) <= (70*15):
            sun = ((int(i)+1)//70)+1
            fish = (int(i)+1)%70
            name = functional_name(sun)
            if fish == 0:
                ty = 'PresenceAU45_delta_'
                score_accum2['delta'] += score_statistic[i]
                score_accum3['Presence'] += score_statistic[i]
            elif fish > 0 and fish <= 17:
                ty = 'Intensity_self_' + str(fish)
                score_accum2['self'] += score_statistic[i]
                score_accum3['Intensity'] += score_statistic[i]
            elif fish > 17 and fish <= 35:
                ty = 'Presence_self_' + str(fish-17)
                score_accum2['self'] += score_statistic[i]
                score_accum3['Presence'] += score_statistic[i]
            elif fish > 35 and fish <= 51:
                ty = 'Intensity_delta_' + str(fish-35)
                score_accum2['delta'] += score_statistic[i]
                score_accum3['Intensity'] += score_statistic[i]
            elif fish > 51 and fish <= 69:
                ty = 'Presence_delta_' + str(fish- 51)
                score_accum2['delta'] += score_statistic[i]
                score_accum3['Presence'] += score_statistic[i]
        elif int(i) > (70*15):
            i2 = int(i) -(70*15)
            sun = ((int(i2)+1)//35)+1
            fish = (int(i2)+1)%35 
            name = functional_name(sun)
            if fish == 0:
                ty = 'Presence_others_'
                score_accum2['others'] += score_statistic[i]
                score_accum3['Presence'] += score_statistic[i]
            elif fish > 0 and fish <= 17:
                ty = 'Intensity_others_' + str(fish)
                score_accum2['others'] += score_statistic[i]
                score_accum3['Intensity'] += score_statistic[i]
            elif fish > 17 and fish <= 34:
                ty = 'Presence_others' + str(fish-17)
                score_accum2['others'] += score_statistic[i]
                score_accum3['Presence'] += score_statistic[i]
        score_accum[ty] += score_statistic[i]
        deep_score[ty+name] = score_statistic[i]
    return score_accum, score_accum2, score_accum3, deep_score       
def deep_analyse_DT(score_statistic):
    deep_score = collections.defaultdict(int)
    score_accum = collections.defaultdict(int)
    score_accum2 = collections.defaultdict(int)
    score_accum3 = collections.defaultdict(int)    
    for i in score_statistic.keys():
        if int(i) <= (60*15):
            sun = ((int(i)+1)//60)+1
            fish = (int(i)+1)%60
            name = functional_name(sun)
            if fish == 0:
                ty = 'Traject_delta_'
                score_accum2['delta'] += score_statistic[i]
                score_accum3['Traject'] += score_statistic[i]
            elif fish > 0 and fish <= 15:
                ty = 'Info_self_' + str(fish)
                score_accum2['self'] += score_statistic[i]
                score_accum3['Info'] += score_statistic[i]
            elif fish > 15 and fish <= 30:
                ty = 'Traject_self_' + str(fish-15)
                score_accum2['self'] += score_statistic[i]
                score_accum3['Traject'] += score_statistic[i]
            elif fish > 30 and fish <= 45:
                ty = 'Info_delta_' + str(fish-30)
                score_accum2['delta'] += score_statistic[i]
                score_accum3['Info'] += score_statistic[i]
            elif fish > 45 and fish <= 59:
                ty = 'Traject_delta_' + str(fish- 45)
                score_accum2['delta'] += score_statistic[i]
                score_accum3['Traject'] += score_statistic[i]
#        elif int(i) > (60*15):
#            i2 = int(i) -(60*15)
#            sun = ((int(i2)+1)//30)+1
#            fish = (int(i2)+1)%30 
#            name = functional_name(sun)
#            if fish == 0:
#                ty = 'Traject_others_'
#                score_accum2['others'] += score_statistic[i]
#                score_accum3['Traject'] += score_statistic[i]
#            elif fish > 0 and fish <= 15:
#                ty = 'Info_others_' + str(fish)
#                score_accum2['others'] += score_statistic[i]
#                score_accum3['Info'] += score_statistic[i]
#            elif fish > 15 and fish <= 30:
#                ty = 'Traject_others' + str(fish-15)
#                score_accum2['others'] += score_statistic[i]
#                score_accum3['Traject'] += score_statistic[i]
        score_accum[ty] += score_statistic[i]
        deep_score[ty+name] = score_statistic[i]
    return score_accum, score_accum2, score_accum3, deep_score       
    

def score_initial(fea_Com):
    length = len(fea_Com.keys())
    check = 0
    for key in fea_Com.keys():  
        if check == 0:     
            width = len(fea_Com[key][0])
            check += 1
    score= np.zeros([length, width-2])
    return score    

#%%
os.chdir('D:\\Lab\\Dennis\\Gamania\\Script')
WORKDIR = 'C:\\Users\\dennis60512\\Desktop\\'

#WORKDIR = '..\\Script\\VideoFeature\\NewFeature2\\Delta\\'
ROOT = os.getcwd()
LABELTYPE = [ 'Act']
CTYPE = [ 0.001]
#PATH = os.path.join(ROOT, WORKDIR, WORKDIR.split('\\')[2]+'-crop')   
#%%
test_dict = collections.defaultdict(list)
#fesLength2 = collections.defaultdict(list)
final = collections.defaultdict(list)
final2 = collections.defaultdict(list)
#fesLength2 = ib.load(WORKDIR + 'fesLength2.pkl')
for label in LABELTYPE:
    feaVideo = ib.load(WORKDIR + label+ '_feaVideo.pkl')
    feaCom = feaVideo
#    fesLength2 = ib.load(WORKDIR + label + '_fesLength2.pkl')
    for C_number in CTYPE:
        for xx in range(20, 110, 10):
            if xx == 30:
                tru=[]
                pree=[]
                score= score_initial(feaCom)
                for idx, n in tqdm(enumerate(feaCom.keys())):
                    if int(n.split('_')[0])>0:
                        tralis=list(set(feaCom.keys())-{n})
                        train_data=[]
                        train_label=[]    
                        for i in tralis:
                            if int(i.split('_')[0])>0:
                                data2=np.array(feaCom[i])
                                train_data.extend(data2[:, :-2].tolist())
                                train_label.extend(data2[:, -2].tolist())   
                        [train_data, mean, std] = zsco(np.array(train_data), 0)
                        train_data = np.nan_to_num(train_data)
                        data1=np.array(feaCom[n])
                        test_data = data1[:, :-2].tolist()
                        test_label= data1[:, -2].tolist()
                    
                        test_data = passzsco(np.array(test_data), 0, mean, std)
                        test_data = np.nan_to_num(test_data)
                        ps = SelectPercentile(f_classif, percentile = xx)
                        train_data = ps.fit_transform(train_data, train_label)
                        test_data=ps.transform(test_data)
                        score[idx] = ps.scores_
                        clf=SVC(kernel='linear',class_weight='balanced',C=C_number)
                        clf.fit(train_data,train_label)
                        pre=clf.predict(test_data)
#                        for kk in fesLength2.keys():
#                            if n == (kk[0:-1]):
#                                tmp = []
#                                trueone = test_label[fesLength2[kk][0]:fesLength2[kk][1]].count(1)
#                                preone= pre[fesLength2[kk][0]:fesLength2[kk][1]].tolist().count(1)
#                                testall = trueone + test_label[fesLength2[kk][0]:fesLength2[kk][1]].count(0) + test_label[fesLength2[kk][0]:fesLength2[kk][1]].count(2)
#                                tmp.append(trueone)
#                                tmp.append(preone)
#                                tmp.append(testall)
#                                final[kk] = tmp
                            
                        tru.extend(test_label)
                        pree.extend(pre)
                       
#                print(cm(tru,pree, labels= [1, 0]))
                accu = np.int(UAR(tru,pree, average='macro')*1000)/10.0
                accu2 = np.int(UAR(tru,pree, average='weighted')*1000)/10.0
                print('Label:',label,'C:',C_number,'per:',xx,'UAR:',accu)
                print('Label:',label,'C:',C_number,'per:', xx,'Accu:',accu2)
#tes = []
#pr = []
#for key,value in final.items():
#    tes.append(value[0])
#    pr.append(value[1])
#print(spearmanr(tes, pr))
#print(pearsonr(tes, pr))
#tes = []
#pr = []
#for key,value in final.items():
#    tes.append(value[0]/value[2])
#    pr.append(value[1]/value[2])
#print(spearmanr(tes, pr))
#print(pearsonr(tes, pr)) 
#print('\n')
score4 = score
#score_statistic, dim = score_analyse(score, 50)#(522/2610)
#score_accum, score_accum2, score_accum3, new_score = deep_analyse(score_statistic)
#sort_score = sorted(new_score.items(), key=operator.itemgetter(1), reverse = True)              
#sort_accum = sorted(score_accum.items(), key=operator.itemgetter(1), reverse = True)  
#################action_unit########################
#score_statistic, dim = score_analyse(score2, 40)
#score_accum, score_accum2, score_accum3, new_score = deep_analyse_au(score_statistic)
#sort_score = sorted(new_score.items(), key=operator.itemgetter(1), reverse = True)              
#sort_accum = sorted(score_accum.items(), key=operator.itemgetter(1), reverse = True)  
################info_traj############################
#score_statistic, dim = score_analyse(score3, 40)
#score_accum, score_accum2, score_accum3, new_score = deep_analyse_DT(score_statistic)
#sort_score = sorted(new_score.items(), key=operator.itemgetter(1), reverse = True)              
#sort_accum = sorted(score_accum.items(), key=operator.itemgetter(1), reverse = True)  
###############OpenPose############################
score_statistic, dim = score_analyse(score4, 40)
score_accum, score_accum2, score_accum3, new_score = deep_analyse_DT(score_statistic)
sort_score = sorted(new_score.items(), key=operator.itemgetter(1), reverse = True)              
sort_accum = sorted(score_accum.items(), key=operator.itemgetter(1), reverse = True)  
ib.dump(dim, WORKDIR + 'dim.pkl')