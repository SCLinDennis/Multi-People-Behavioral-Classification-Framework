# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 16:06:14 2017

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
from scipy import io
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

def select_index_DT(dt, start_time, end_time, frame_rate):    
    answer = 1
    index_tmp = np.where((dt[:, 0:1] > start_time * frame_rate)& (dt[:, 0:1] < end_time*frame_rate))
    if len(index_tmp[0]) ==0:
        #print("Sorry. We cannot find feature between " +  str(start_time)+" and "+ str(end_time))
        answer = 0
    
    dt_out = dt[index_tmp[0]]
    return dt_out, answer

def select_index_po(df, start, end, frame_rate):
    answer = 1    
    start = round(start*frame_rate)
    end = round(end*frame_rate)
    if start == end:
        answer = 0
    df_out = df[start:end+1]
    return df_out, answer    

def delta_extract(df):
    df_delta = df.diff().fillna(0)
    fea = [df, df_delta]
    df_out = pd.concat(fea, axis = 1)
    return df_out

def np_delta_extract(df):
    df_tmp = np.diff(df, axis = 0)
    zero = np.zeros([1, df.shape[1]])
    df_delta = np.vstack((zero, df_tmp))
    df_out = np.hstack((df, df_delta))
    return df_out

def delta_extract_DT(df):
    df_tmp = np.diff(df, axis = 0)
    zero = np.zeros([1, df.shape[1]])
    df_delta = np.vstack((zero, df_tmp))
    df_out = np.hstack((df, df_delta))
    return df_out
        
def key_translate(date):
    if len(date.split('/')[-1].split('_')) == 3:
        index = ''.join(date.split('/')[-1].split('_')[0:2])
    elif len(date.split('/')[-1].split('_')) == 4:
        index = ''.join(date.split('/')[-1].split('_')[0:2]) + '_' +date.split('/')[-1].split('_')[2]
    return index
    
def load_audio(feaAudio, AudioPath, index, label, keepname):
    for idx,i2 in enumerate(AudioPath):
        if i2.split('/')[-1][-5] != 'N' and i2.split('/')[-1][-5] != 'H' and i2.split('/')[-1][-5] != 'X'and i2.split('/')[-1][-5] != 'F':
            if  i2.split('/')[-1][0:-6] == keepname:
                da=sp.io.loadmat(i2)['Audio_data'].tolist()
                da[0].extend(label)
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
def Act_feature_extend(feature, start_time, end_time, start_extend, end_extend, frame_rate):
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

def get_length_angle(feature):
    max_length = np.max(np.array(feature)[:,0], 0)
    even_idx = range(0, 22, 2)
    odd_idx = range(1, 23, 2)
    fea_even = np.array(feature)[:,even_idx] / max_length 
    #print(str(np.shape(fea_even)) + str(np.shape(np.array(feature)[:,odd_idx])))
    fea_normalized = np.hstack((fea_even, np.array(feature)[:,odd_idx]))
    return fea_normalized
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
        Act_label[key] = [list(x) for x in b_set]
    return Act_label
'''        
def Act_feature_extend_new(feature, start_time, end_time, start_extend, end_extend, frame_rate, fea_main):
    new_df_tmp, x = select_index(feature, start_time+start_extend, end_time+end_extend, frame_rate)
    if x ==1: 
        new_df_tmp = new_df_tmp.as_matrix()
        New_feat = getFunctional(new_df_tmp)
        fea_out = fea_main - New_feat
    else:
        fea_out = fea_main
    return fea_out
'''
#%%
os.chdir('/mnt/HGY/Gamania/Script_encoded/')
WORKDIR = '../RawFeatures/Trained/Video/PoseParamGaze/'
WORKDIR_Au = '../RawFeatures/Trained/Video/ActionUnit/'
WORKDIR_DT = '../RawFeatures/Trained/Video/DenseTrajectory/'
WORKDIR_Po = '../RawFeatures/Pose_on_progress/2_pose-crop-final/'
#LABELDIR = './VideoLabel/VideoLabel/VideoLabelNew/'
LABELDIR = '../Label/VideoLabelNewCut/'
WORKDIR_dim_PPG = '../EncodedFeature/Trained/VideoFeatureNewCut/NewFeature2/Delta/'
WORKDIR_dim_AU = '../EncodedFeature/Trained/VideoFeatureNewCut/ActionUnit/Interact+delta/'
WORKDIR_dim_DT = '../EncodedFeature/Trained/VideoFeatureNewCut/Info_traj/'
WORKDIR_dim_Po = '../EncodedFeature/Trained/VideoFeatureNewCut/Pose/Delta/'
WORKDIR_AUDIO = '../RawFeatures/Trained/Audio/labeled_wav/feature_tovideo_newcut/'
FEATUREDIR = '../EncodedFeature/Trained/VideoFeatureNewCut/Combine2(pose)/'
ROOT = os.getcwd()
LABELTYPE = [ 'Act']
frame_rate = 30
frame_rate_po = 10
#commingtohelp = [ '../Data/Feature/07_11_1_feature.pkl', '../Data/Feature/07_11_2_feature.pkl', '../Data/Feature/07_12_1_feature.pkl', '../Data/Feature/07_12_2_feature.pkl', '../Data/Feature/07_12_3_feature.pkl', '../Data/Feature/07_13_1_feature.pkl', '../Data/Feature/07_13_2_feature.pkl', '../Data/Feature/07_14_1_feature.pkl', '../Data/Feature/07_14_2_feature.pkl', '../Data/Feature/07_18_feature.pkl', '../Data/Feature/07_19_1_feature.pkl', '../Data/Feature/07_19_2_feature.pkl', '../Data/Feature/07_19_3_feature.pkl']
#commingtohelp2 = [ '../Data/Feature/07_20_1_feature.pkl', '../Data/Feature/07_20_3_feature.pkl', '../Data/Feature/07_21_1_feature.pkl', '../Data/Feature/07_21_2_feature.pkl', '../Data/Feature/07_21_3_feature.pkl', '../Data/Feature/07_24_1_feature.pkl', '../Data/Feature/07_24_2_feature.pkl', '../Data/Feature/07_24_3_feature.pkl', '../Data/Feature/07_25_1_feature.pkl', '../Data/Feature/07_25_2_feature.pkl', '../Data/Feature/07_26_feature.pkl', '../Data/Feature/07_27_2_feature.pkl', '../Data/Feature/07_27_3_feature.pkl']   
#commingtohelp3 = [ '../Data/Feature/06_20_2_feature.pkl', '../Data/Feature/06_21_feature.pkl', '../Data/Feature/06_26_1_feature.pkl', '../Data/Feature/06_26_2_feature.pkl', '../Data/Feature/06_27_feature.pkl', '../Data/Feature/06_28_feature.pkl', '../Data/Feature/06_30_feature.pkl', '../Data/Feature/07_03_1_feature.pkl', '../Data/Feature/07_03_2_feature.pkl', '../Data/Feature/07_05_feature.pkl', '../Data/Feature/07_06_1_feature.pkl', '../Data/Feature/07_06_2_feature.pkl', '../Data/Feature/07_07_feature.pkl']   
commingtohelp4 = ['05_24_feature.pkl', '06_20_1_feature.pkl', '06_20_2_feature.pkl', '06_21_feature.pkl', '06_26_2_feature.pkl', '06_27_feature.pkl', '07_24_3_feature.pkl', '07_25_1_feature.pkl']

#%%
dim_PPG = ib.load(WORKDIR_dim_PPG + 'dim.pkl')
dim_AU  = ib.load(WORKDIR_dim_AU + 'dim.pkl')
dim_DT = ib.load(WORKDIR_dim_DT + 'dim.pkl') 
dim_Po = ib.load(WORKDIR_dim_Po + 'dim.pkl')
for Label in LABELTYPE:
    feaAudio = collections.defaultdict(list)
    feaVideo = collections.defaultdict(list) 
    fesLength2 = collections.defaultdict(list)
    for date in sorted(glob.glob(WORKDIR+'*_feature.pkl')):
        if date.split('/')[-1] not in commingtohelp4:
#        if 1 != 0:
            index = key_translate(date)
            Act_label = ib.load(LABELDIR+ index + '_' + Label + '.pkl')
            fea_Com = ib.load(date)
            fea_Com_au = ib.load(WORKDIR_Au + date.split('/')[-1])
            fea_Com_DT = ib.load(WORKDIR_DT + date.split('/')[-1][:-11] + 'DT_feature.pkl')
            if len(date.split('/')[-1].split('_')) == 3:
                date2 = '2017-' + '-'.join(date.split('/')[-1].split('_')[0:2])
            else:
                date2 = '2017-' + '-'.join(date.split('/')[-1].split('_')[0:3]) 
            fea_Com_Po = ib.load(WORKDIR_Po + date2 + '-ratio.pickle')
            print("Loading" + date)
            
            #Label Preprocessing
            Act_label = label_preprocess(Act_label)
#            spk_less = whospeakmost(Act_label, 0)
#            spk_most = whospeakmost(Act_label, 1)
            #Audio path 
            
            ifiles  = glob.glob(WORKDIR_AUDIO + '*.mat')    
            
            #Initialize the feature
            df = []
            length = 0
            #loop the people
            label_index = collections.defaultdict(list)
            for key2 in sorted(fea_Com.keys()):
                print('Now doing people ' + key2 +'\n')
                label_tmp = []
                add = []
                add.append(length)
                length_tmp = 0
                
                #loop the label
                for i1, tmp in enumerate(sorted(Act_label[key2])):
                    #lab = tmp[0]
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
                    fea_au_withdelta = delta_extract(fea_Com_au[key2])
                    fea_po_normalized = get_length_angle(fea_Com_Po[key2])
                    fea_po_withdelta = np_delta_extract(fea_po_normalized)

                    df_tmp, answer = select_index(fea_withdelta, float(start), float(end), frame_rate)
                    df_tmp_au, answer2 = select_index(fea_au_withdelta, float(start), float(end), frame_rate)
                    df_tmp_DT, answer3 = select_index_DT(fea_Com_DT[key2], float(start), float(end), frame_rate)
                    df_tmp_po, answer4 = select_index_po(fea_po_withdelta, float(start), float(end), frame_rate_po)
                    
                     
                    #Video Feature Preprocessing
                    df_tmp_array = df_tmp.as_matrix()  
                    df_tmp_array_au = df_tmp_au.as_matrix()
                    df_tmp_array_DT = df_tmp_DT[:, range(0, 30)]
                    if answer == 1 and answer2 == 1 and answer3 == 1 and answer4 == 1:     
                        df_tmp_array_po = np.nan_to_num(df_tmp_po)
                        #load Au feature 
                        feaAudio = load_audio(feaAudio, ifiles, index, [int(lab1), int(lab2)], keepname)            
                        df_tmp_array_DT = delta_extract_DT(df_tmp_array_DT)
                        if len(np.where(np.isnan(df_tmp_array))[0]) > 0:
                            pdb.set_trace()
                        if len(np.where(np.isnan(df_tmp_array_au))[0]) > 0:
                            pdb.set_trace()  
                        if len(np.where(np.isnan(df_tmp_array_po))[0]) > 0:
                            pdb.set_trace()
                        #add delta to matrix
                        feaComCut =  getFunctional(df_tmp_array)
                        feaComCut_au = getFunctional(df_tmp_array_au)
                        feaComCut_DT = getFunctional(df_tmp_array_DT)
                        feaComCut_po = getFunctional(df_tmp_array_po)

                        #Create New Feature2
                        
                        if Label == 'Act':
                            New_feat = Act_feature_extend(fea_Com, float(start), float(end), 0, 10, frame_rate)
                            New_feat2 = Act_feature_extend(fea_Com_au, float(start), float(end), 0, 10, frame_rate)

                                  
                        feaComCut = np.append(feaComCut, New_feat)  #!!!!!!!! 
                        feaComCut_au = np.append(feaComCut_au, New_feat2)
                        #print('dt\'shape = ' + str(feaComCut_DT.shape))
                        #print('fea\'shape = ' + str(feaComCut.shape))
                        if len(np.where(np.isnan(feaComCut))[0]) > 0:
                            pdb.set_trace()
                        
                        #Append the label to the encoded feature                           
                        feaComFinal = np.hstack((feaComCut[dim_PPG],feaComCut_au[dim_AU])) 
                        feaComCut_DT  = np.reshape(feaComCut_DT, [feaComCut_DT.shape[1], ])                        
                        feaComFinal = np.hstack((feaComFinal, feaComCut_DT[dim_DT]))
                        feaComCut_po = np.reshape(feaComCut_po, [feaComCut_po.shape[1], ])
                        feaComFinal = np.hstack((feaComFinal, feaComCut_po[dim_Po]))
                        feaComFinal = np.hstack((feaComFinal, np.array([int(lab1), int(lab2), float(start), float(end)])))
                        if len(np.where(np.isnan(feaComFinal))[0]) > 0:
                            pdb.set_trace()
                        df.append(feaComFinal.tolist())
                        length_tmp += 1
                
                #Feature Length statistics 
                length += length_tmp 
                add.append(length)
                fesLength2[index + key2] = add

            #Save the feature to dictionary
            feaVideo[index] = df
            
    ib.dump(feaAudio, FEATUREDIR + Label+'_feaAudio.pkl')
    ib.dump(feaVideo, FEATUREDIR + Label+'_feaVideo.pkl')
    ib.dump(fesLength2, FEATUREDIR + Label+ '_fesLength2.pkl')
