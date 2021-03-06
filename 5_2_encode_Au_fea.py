# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 17:31:59 2017

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
def select_index(df, start, end, frame_rate):    
    answer = 1
    index_tmp = np.where((df.index.values> start*frame_rate)& (df.index.values <end*frame_rate))
    if len(index_tmp[0]) ==0:
#        print("Sorry. We cannot find feature between " +  str(start)+" and "+ str(end))
        answer = 0
    
    index = df.index.values[index_tmp[0]]
    df_out = df.ix[index]
    df_out.drop('confidence', axis=1, inplace=True)
    return df_out, answer

def delta_extract(df):
    df_delta = df.diff().fillna(0)
    fea = [df, df_delta]
    df_out = pd.concat(fea, axis = 1)
    return df_out

def key_translate(date):
    index = ''
    if len(date.split('\\')[-1].split('_')) == 3:
        index = ''.join(date.split('\\')[-1].split('_')[0:2])
    elif len(date.split('\\')[-1].split('_')) == 4:
        index = ''.join(date.split('\\')[-1].split('_')[0:2]) + '_' +date.split('\\')[-1].split('_')[2]
    return index
    
def load_audio(feaAudio, AudioPath, index, label, keepname):
    for idx,i2 in enumerate(AudioPath):
        if i2.split('\\')[-1][-5] != 'N' and i2.split('\\')[-1][-5] != 'H'and i2.split('\\')[-1][-5] != 'X'and i2.split('\\')[-1][-5] != 'F':
            if  i2.split('\\')[-1][0:-6] == keepname:
                da=sp.io.loadmat(i2)['Audio_data'].tolist()
                da[0].append(label)
                feaAudio[index].append(da[0])
    return feaAudio
'''
def Col_feature_extend(feature, start_time, end_time, start_extend, end_extend, frame_rate):
    new_df = np.array([])
    for key3, value in feature.items():
        new_df_tmp, x = select_index(value, start_time+start_extend, end_time+end_extend, frame_rate)
        
        if len(new_df) == 0:
            new_df = new_df_tmp.as_matrix()
        else:
            new_df = np.vstack((new_df, new_df_tmp))
                            
        if len(new_df) != 0:
            New_feat = getFunctional(new_df)
        else:
            New_feat = getFunctional(df_tmp_array) 
    return New_feat
'''
def Act_feature_extend(feature, start_time, end_time, start_extend, end_extend , frame_rate):
    accum = 0
    for key3, value in feature.items():
        new_df_tmp, x = select_index(value, start_time+start_extend, end_time+end_extend, frame_rate)
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
'''
def whospeakmost(label, moreorless): #moreorless == 1(more), 0(less)
    tmp = 0
    for key, value in label.items():
        if tmp == 0:
            tmp = len(value)
            speakmost = key
        else:
            if moreorless == 1:
                if len(value) > tmp:
                    tmp = len(value)
                    speakmost = key
            else:
                if len(value) < tmp:
                    tmp = len(value)
                    speakmost = key
    return speakmost
'''
def label_preprocess(Act_label):            
    for key in Act_label:
        for i in range(len(Act_label[key])):
            Act_label[key][i] = Act_label[key][i].split(' ')
        b_set = set(tuple(x) for x in Act_label[key] )
        Act_label[key] = [ list(x) for x in b_set ]
    return Act_label        
        
def Act_feature_extend_new(feature, start_time, end_time, start_extend, end_extend, frame_rate, fea_main):
    new_df_tmp, x = select_index(feature, start_time+start_extend, end_time+end_extend, frame_rate)
    if x ==1: 
        new_df_tmp = new_df_tmp.as_matrix()
        New_feat = getFunctional(new_df_tmp)
        fea_out =  fea_main - New_feat
    else:
        fea_out = fea_main
    return fea_out
'''
def Act_feature_extend_new2(feature, start_time, end_time, start_extend, end_extend, frame_rate, fea_main):
    new_df_tmp, x = select_index(feature, start_time+start_extend, end_time+end_extend, frame_rate)
    if x ==1: 
        new_df_tmp = new_df_tmp.as_matrix()
        New_feat = getFunctional(new_df_tmp)
        fea_out = New_feat - fea_main  
    else:
        fea_out = -fea_main
    return fea_out
'''

#%%
os.chdir('D:\\Lab\\Dennis\\Gamania\\Script')
WORKDIR = '..\\Data\\AU_Feature\\'
LABELDIR = '.\\VideoLabel\\VideoLabel\\VideoLabelNewCut\\'
FEATUREDIR = '.\\VideoFeatureNewCut\\ActionUnit\\'
#FEATUREDIR = '.\\VideoFeature_3class\\ActionUnit\\Interact+delta\\'
ROOT = os.getcwd()
LABELTYPE = ['Act']
frame_rate = 30 

#commingtohelp = [ '..\\Data\\AU_Feature\\07_11_1_feature.pkl', '..\\Data\\AU_Feature\\07_11_2_feature.pkl', '..\\Data\\AU_Feature\\07_12_1_feature.pkl', '..\\Data\\AU_Feature\\07_12_2_feature.pkl', '..\\Data\\AU_Feature\\07_12_3_feature.pkl', '..\\Data\\AU_Feature\\07_13_1_feature.pkl', '..\\Data\\AU_Feature\\07_13_2_feature.pkl', '..\\Data\\AU_Feature\\07_14_1_feature.pkl', '..\\Data\\AU_Feature\\07_14_2_feature.pkl', '..\\Data\\AU_Feature\\07_18_feature.pkl', '..\\Data\\AU_Feature\\07_19_1_feature.pkl', '..\\Data\\AU_Feature\\07_19_2_feature.pkl', '..\\Data\\AU_Feature\\07_19_3_feature.pkl']   
#commingtohelp2 = [ '..\\Data\\AU_Feature\\07_20_1_feature.pkl', '..\\Data\\AU_Feature\\07_20_3_feature.pkl', '..\\Data\\AU_Feature\\07_21_1_feature.pkl', '..\\Data\\AU_Feature\\07_21_2_feature.pkl', '..\\Data\\AU_Feature\\07_21_3_feature.pkl', '..\\Data\\AU_Feature\\07_24_1_feature.pkl', '..\\Data\\AU_Feature\\07_24_2_feature.pkl', '..\\Data\\AU_Feature\\07_24_3_feature.pkl', '..\\Data\\AU_Feature\\07_25_1_feature.pkl', '..\\Data\\AU_Feature\\07_25_2_feature.pkl', '..\\Data\\AU_Feature\\07_26_feature.pkl', '..\\Data\\AU_Feature\\07_27_2_feature.pkl', '..\\Data\\AU_Feature\\07_27_3_feature.pkl']   
#commingtohelp3 = [ '..\\Data\\AU_Feature\\06_20_2_feature.pkl', '..\\Data\\AU_Feature\\06_21_feature.pkl', '..\\Data\\AU_Feature\\06_26_1_feature.pkl', '..\\Data\\AU_Feature\\06_26_2_feature.pkl', '..\\Data\\AU_Feature\\06_27_feature.pkl', '..\\Data\\AU_Feature\\06_28_feature.pkl', '..\\Data\\AU_Feature\\06_30_feature.pkl', '..\\Data\\AU_Feature\\07_03_1_feature.pkl', '..\\Data\\AU_Feature\\07_03_2_feature.pkl', '..\\Data\\AU_Feature\\07_05_feature.pkl', '..\\Data\\AU_Feature\\07_06_1_feature.pkl', '..\\Data\\AU_Feature\\07_06_2_feature.pkl', '..\\Data\\AU_Feature\\07_07_feature.pkl']   
commingtohelp4 = ['..\\Data\\AU_Feature\\05_24_feature.pkl', '..\\Data\\AU_Feature\\06_20_1_feature.pkl', '..\\Data\\AU_Feature\\06_20_2_feature.pkl', '..\\Data\\AU_Feature\\06_21_feature.pkl', '..\\Data\\AU_Feature\\06_26_2_feature.pkl', '..\\Data\\AU_Feature\\06_27_feature.pkl', '..\\Data\\AU_Feature\\07_24_3_feature.pkl', '..\\Data\\AU_Feature\\07_25_1_feature.pkl']
commingtohelp5 = ['..\\Data\\AU_Feature\\06_28_feature.pkl']
#%%
for Label in LABELTYPE:
    feaAudio = collections.defaultdict(list)
    feaVideo = collections.defaultdict(list) 
    fesLength2 = collections.defaultdict(list)
    for date in sorted(glob.glob(WORKDIR+'\\*_feature.pkl')):
        if date not in  commingtohelp4:
#        if 1 != 0:
            index = key_translate(date)
            Act_label = ib.load(LABELDIR+ index + '_' + Label + '.pkl')
            fea_Com = ib.load(date)
            print("Loading" + date)
            
            #Label Preprocessing
            Act_label = label_preprocess(Act_label)
#            spk_less = whospeakmost(Act_label, 0)
#            spk_most = whospeakmost(Act_label, 1)
            
            #Audio path 
            '''
            ifiles  = glob.glob('D:\\Lab\\Dennis\\Gamania\\Jim\\labeled_wav\\feature_tovideo_new\\*.mat')    
            '''
            #Initialize the feature
            df = []
            length = 0
            
            #loop the people
            label_index = collections.defaultdict(list)
            for key2 in fea_Com.keys():
                print('Now doing people ' + key2 +'\n')
                label_tmp = []
                add = []
                add.append(length)
                length_tmp = 0
                
                #loop the label
                for i1, tmp in enumerate(Act_label[key2]):
                    if tmp[0]  == '0':
                        lab1 =  '0'
                        lab2 = '0'
                    elif tmp[0] == '1':
                        lab1 = '1'
                        lab2 = '1'
                    elif tmp[0] == '2':
                        lab1 = '1'
                        lab2 = '2'
                    elif tmp[0] == '3':
                        lab1 = '1'
                        lab2 = '2'
                    start = tmp[1]
                    end = tmp[2]
                    keepname = tmp[3]
                    answer = 0
                    fea_withdelta = delta_extract(fea_Com[key2])
                    df_tmp, answer = select_index(fea_Com[key2], float(start), float(end), frame_rate)
#                    df_tmp, answer = select_index(fea_Com[key2], float(start), float(end), frame_rate)
                    if answer ==1:
                        label_tmp.append(1)
                    else:
                        label_tmp.append(0)
                    
                    #load Au feature
                    '''
                    if answer == 1:
                        feaAudio = load_audio(feaAudio, ifiles, index, int(lab), keepname)
                    ''' 
                    #Video Feature Preprocessing
                    df_tmp_array = df_tmp.as_matrix()                    
                    if len(df_tmp_array) != 0:                  
                        if len(np.where(np.isnan(df_tmp_array))[0]) > 0:
                            pdb.set_trace()
                        feaComCut =  getFunctional(df_tmp_array)
                        
                        '''
                        #Create New Feature                        
                        if Label == 'Col':#fea_Com, start, end, -10, 0, 
                            New_feat = Col_feature_extend(fea_Com, float(start), float(end), -10, 0, frame_rate)
                        '''
                        #Create New Feature2
                        
                        if Label == 'Act':#fea_Com, start, end, start_extend, end_extend
                            New_feat = Act_feature_extend(fea_Com, float(start), float(end), 0, 10, frame_rate)
                            '''
                            New_feat = Act_feature_extend_new(fea_Com[spk_less], float(start), float(end), 5, 0, frame_rate, feaComCut)
                            New_feat2 = Act_feature_extend_new2(fea_Com[spk_most], float(start), float(end), 0, 5, frame_rate, feaComCut)
                            '''
                        feaComCut = np.append(feaComCut, New_feat)  #!!!!!!!! 
                        if len(np.where(np.isnan(feaComCut))[0]) > 0:
                            pdb.set_trace()
                        #Append the label to the encoded feature
                        print(feaComCut.shape)
                        feaComCut = np.hstack((feaComCut, np.array([int(lab1), int(lab2)])))
                        df.append(feaComCut.tolist())
                        length_tmp += 1
                
                label_index[key2] = label_tmp
                
                #Feature Length statistics 
                length += length_tmp 
                add.append(length)
                fesLength2[index + key2] = add

            #Save the feature to dictionary
            feaVideo[index] = df
            
#    ib.dump(feaAudio, FEATUREDIR + Label+'_feaAudio.pkl')
#    ib.dump(feaVideo, FEATUREDIR + Label+'_feaVideo.pkl')
#    ib.dump(fesLength2, FEATUREDIR + Label+ '_fesLength2.pkl')
#        
