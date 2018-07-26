# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 17:03:21 2017

@author: dennis60512
"""


import os, glob, sys
import multiprocessing  as mp
import pandas as pd
import collections
import joblib as ib
from collections import defaultdict
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
from scipy import stats as stats
import scipy as sp
import pdb
import  scipy.stats as stats
#%%
def getFunctional(data):    
    """Functional list: ['max','min','mean','median','standard_deviation','1_percentile','99_percentile',
                         '99minus1_percentile','skewneww','kurtosis','min_pos','max_pos','low_quar',
                         'up_quar','quartile_range'] """
    Functional = []
    #0 max
    Functional.append(np.max(data, axis = 0))
    #1 min    
    Functional.append(np.min(data, axis = 0))
    #2 mean
    Functional.append(np.mean(data, axis = 0))
    #3 median    
    Functional.append(np.median(data, axis = 0))
    #4 standard deviation
    Functional.append(np.std(data, axis = 0) )         
    #5 1st_percentile   
    Functional.append(np.percentile(data, 1, axis = 0))
    #6 99th percentile
    Functional.append(np.percentile(data, 99, axis = 0))
    #7 99th percentile - 1st percentile
    Functional.append(Functional[-1]-Functional[-2])
    #8 skewness
    Functional.append(stats.skew(data, axis=0))
    #9 kurtosis
    Functional.append(stats.kurtosis(data, axis=0))
    #10 minmum position
    Functional.append((np.argmin(data, axis=0)).astype(float)/len(data))
    #11 maximum position
    Functional.append((np.argmax(data, axis=0)).astype(float)/len(data))
    #12 lower quartile
    Functional.append(np.percentile(data, 25, axis = 0))
    #13 upper quartile
    Functional.append(np.percentile(data, 75, axis = 0))
    #14 interqyartile range
    Functional.append(Functional[-1]-Functional[-2])
    #return np.asanyarray(Functional)
    return np.vstack(Functional).reshape(1, -1)
def select_index(dt, start_time, end_time, frame_rate):    
    answer = 1
    index_tmp = np.where((dt[:, 0:1] > start_time * frame_rate)& (dt[:, 0:1] < end_time*frame_rate))
    if len(index_tmp[0]) ==0:
        #print("Sorry. We cannot find feature between " +  str(start_time)+" and "+ str(end_time))
        answer = 0
    
    dt_out = dt[index_tmp[0]]
    return dt_out, answer

def delta_extract(df):
    df_tmp = np.diff(df, axis = 0)
    zero = np.zeros([1, df.shape[1]])
    df_delta = np.vstack((zero, df_tmp))
    df_out = np.hstack((df, df_delta))
    return df_out

def key_translate(date):
    index = ''
    if len(date.split('\\')[-1].split('_')) == 4:
        index = ''.join(date.split('\\')[-1].split('_')[0:2])
    elif len(date.split('\\')[-1].split('_')) == 5:
        index = ''.join(date.split('\\')[-1].split('_')[0:2]) + '_' +date.split('\\')[-1].split('_')[2]
    return index
    
def load_audio(feaAudio, AudioPath, index, label):
    for idx,i2 in enumerate(AudioPath):
        if i2.split('\\')[-1][-5] != 'N' and i2.split('\\')[-1][-5] != 'H'and i2.split('\\')[-1][-5] != 'X'and i2.split('\\')[-1][-5] != 'F':
            if  i2.split('\\')[-1][0:-6] == keepname:
                da=sp.io.loadmat(i2)['Audio_data'].tolist()
                da[0].append(label)
                feaAudio[index].append(da[0])
    return feaAudio

def Col_feature_extend(feature, start_time, end_time, start_extend, end_extend):
    new_df = np.array([])
    for key3, value in feature.items():
        new_df_tmp, x = select_index(value, start_time+start_extend, end_time+end_extend)
        
        if len(new_df) == 0:
            new_df = new_df_tmp.as_matrix()
        else:
            new_df = np.vstack((new_df, new_df_tmp))
                            
        if len(new_df) != 0:
            New_feat = getFunctional(new_df)
        else:
            New_feat = getFunctional(df_tmp_array) 
    return New_feat

def Act_feature_extend(feature, start_time, end_time, start_extend, end_extend):
    accum = 0
    for key3, value in fea_Com.items():
        new_df_tmp, x = select_index(value, start_time+start_extend, end_time+end_extend)
        if x == 1:
            new_df_tmp = new_df_tmp.as_matrix()
            accum += 1
            if accum == 1:
                New_feat = getFunctional(new_df_tmp)
            else:
                New_feat += getFunctional(new_df_tmp)
    if accum != 0:
        New_feat = New_feat/accum
    else:
        New_feat = getFunctional(df_tmp_array)
    return New_feat
        
def OutDirectoryBuilder(FEATUREDIR, index):
    if index == 0:
        out_dir =os.path.join(FEATUREDIR + 'Info')+'\\'
    elif index == 1:
        out_dir = os.path.join(FEATUREDIR + 'Traject')+'\\'
    elif index == 2:
        out_dir = os.path.join(FEATUREDIR + 'HOG')+'\\'
    elif index == 3:
        out_dir = os.path.join(FEATUREDIR + 'HOF')+'\\'
    elif index == 4:
        out_dir = os.path.join(FEATUREDIR + 'MBHx')+'\\'
    elif index == 5:
        out_dir = os.path.join(FEATUREDIR + 'MBHy') + '\\'
    return out_dir
#%%
#os.chdir('D:\\Lab\\Dennis\\Gamania\\Script')
os.chdir('G:\\GAMANIA\\Data_smaller\\')
WORKDIR = '.\\DT_FeatureNew\\'
#num = 3
LABELDIR = '.\\VideoLabelNew\\'
FEATUREDIR = '.\\VideoFeature\\'
ROOT = os.getcwd()
LABELTYPE = ['Act']
frame_rate = 30 

info = range(0, 10)
traject = range(10, 30)
info_traject = range(0, 30)
HOG = range(30, 126)
HOF = range(126, 234)
MBHx = range(234, 330)
MBHy = range(330, 426)

#LABELTYPE = ['Act']
#PATH = os.path.join(ROOT, WORKDIR, WORKDIR.split('\\')[2]+'-crop')   
#%%
#for type_name, feature_type in enumerate([range]):
#    out_dir = OutDirectoryBuilder(FEATUREDIR, type_name)
if 1 ==1:
    out_dir = os.path.join(FEATUREDIR + 'Info_traj') + '\\'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    for Label in LABELTYPE:
#        feaAudio = collections.defaultdict(list)
        feaVideo = collections.defaultdict(list) 
#        fesLength2 = collections.defaultdict(list)
        for date in glob.glob(WORKDIR+'\\*DT_feature.pkl'):
    #        if date ==  '..\\Data\\Feature\\06_26_1_feature.pkl':
            if 1 != 0:
                index = key_translate(date)
                Act_label = ib.load(LABELDIR+ index + '_' + Label + '.pkl')
                fea_Com = ib.load(date)
                print("Loading" + date)
                
                #Label Preprocessing
                for key in Act_label:
                    for i in range(len(Act_label[key])):
                        Act_label[key][i] = Act_label[key][i].split(' ')
                    b_set = set(tuple(x) for x in Act_label[key] )
                    Act_label[key] = [ list(x) for x in b_set ]
                
                #Audio path 
                '''
                ifiles  = glob.glob('D:\\Lab\\Dennis\\Gamania\\Jim\\labeled_wav\\feature_tovideo_new\\*.mat')    
                '''
                #Initialize the feature
                df = []
                length = 0
                
                #loop the people
#                label_index = collections.defaultdict(list)
                for key2 in sorted(fea_Com.keys()):
                    print('Now doing people ' + key2 +'\n')
#                    label_tmp = []
#                    add = []
                    detect = []
#                    add.append(length)
#                    length_tmp = 0
                    
                    #loop the label
                    for i1, tmp in enumerate(Act_label[key2]):
                        lab = tmp[0]
                        if lab == '3':
                            lab =  '1'
                        start = tmp[1]
                        end = tmp[2]
                        keepname = tmp[3]
                        answer = 0
                        df_tmp, answer = select_index(fea_Com[key2], float(start), float(end), frame_rate)
                        
                        if answer ==1:
#                            label_tmp.append(1)
                            detect.append(1)
#                        else:
#                            label_tmp.append(0)
                            
                        
                        #load Au feature
                        '''
                        if answer == 1:
                            feaAudio = load_audio(feaAudio, ifiles, index, int(lab))
                        '''    
                        #Video Feature Preprocessing
                        df_tmp_array = df_tmp[:,info_traject]
                        if len(df_tmp_array) != 0:  
                            df_tmp_array = delta_extract(df_tmp_array)
                            if len(np.where(np.isnan(df_tmp_array))[0]) > 0:
                                pdb.set_trace()
                            feaComCut =  getFunctional(df_tmp_array)
                            
                            
                            #Create New Feature
                            '''
                            if Label == 'Col':#fea_Com, start, end, -10, 0, 
                                New_feat = Col_feature_extend(fea_Com, float(start), float(end), -10, 0)
                            
                            if Label == 'Act':
                                New_feat = Act_feature_extend(fea_Com, float(start), float(end), 0, 5)
                            '''
                            
    #                        feaComCut = np.append(feaComCut, New_feat)  #!!!!!!!! 
                            
                            if len(np.where(np.isnan(feaComCut))[0]) > 0:
                                pdb.set_trace()
                            #Append the label to the encoded feature
                            feaComCut = np.append(feaComCut, int(lab))
                            df.append(feaComCut.tolist())
#                            length_tmp += 1
                    
#                    label_index[key2] = label_tmp
                    #Feature Length statistics 
#                    length += length_tmp 
#                    add.append(length)
#                    fesLength2[index + key2] = add
                    print('The Dense Extraction Rate of ' + str(key2) + ' :'+ str(len(detect)/len(Act_label[key2])) )
    
                #Save the feature to dictionary
                feaVideo[index] = df
                
#        ib.dump(feaAudio, out_dir + Label+'_feaAudio.pkl')
        ib.dump(feaVideo, out_dir + Label+'_feaVideo.pkl')
#        ib.dump(fesLength2, out_dir + Label+ '_fesLength2.pkl')
