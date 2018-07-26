# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 12:16:52 2017

@author: dennis60512
"""


import os, glob, sys
import subprocess as sp
import multiprocessing  as mp
import pandas as pd
import collections
import joblib as ib
from collections import defaultdict
import pdb
import numpy as np
#%%
#os.chdir('D:\\Lab\\Dennis\\Gamania\\Script')
os.chdir('C:\\Lab\\Dennis\\Gamania\\Data\\')
WORKDIR2 = '..\\Data\\'
ROOT = os.getcwd()


#%% Load in data and Z-score for each person
def parseCMLFea(feaPath):
    feaFace = pd.read_table(feaPath,delimiter=', ', index_col=0, header=0, engine='python')
    return feaFace

def loadFea_pose(feaList):
    df = []
    for feaPath in sorted(feaList):
        print(feaPath+'\n')
#        fea = pd.read_table(feaPath, delimiter=',')
        fea = parseCMLFea(feaPath)
        df.append(fea)
    df = pd.concat(df, ignore_index=True)  
    successIdx = df['success'] != 0
    df= df.ix[successIdx]
    df.drop('success', axis=1, inplace=True)    
    df_right = normalize(df.ix[:, 1:])     
    df_left = df.ix[:, 0]
    df_out = pd.concat([df_left, df_right], axis = 1)
    return df_out
def loadFea_param(feaList):
    df = []
    for feaPath in sorted(feaList):
        print(feaPath+'\n')
#        fea = pd.read_table(feaPath, delimiter=',')
        fea = parseCMLFea(feaPath)
        df.append(fea)
    df = pd.concat(df, ignore_index=True)  
    successIdx = df['success'] != 0
    df= df.ix[successIdx]
    df.drop('success', axis=1, inplace=True)   
    df = normalize(df)     
    return df
def loadFea_gaze(feaList):
    df = []
    for feaPath in sorted(feaList):
        print(feaPath+'\n')
#        fea = pd.read_table(feaPath, delimiter=',')
        fea = parseCMLFea(feaPath)
        df.append(fea)
    df = pd.concat(df, ignore_index=True)  
    successIdx = df['success'] != 0
    df= df.ix[successIdx]
    df.drop('success', axis=1, inplace=True) 
    df.drop('confidence', axis=1, inplace=True)
    df = normalize(df)     
    return df

def normalize(DataFrame):
    df = (DataFrame - DataFrame.mean(axis = 0))/DataFrame.std(axis = 0)
    return df



#%%
comingtohelp = [
#        '..\\Data\\2017-05-24-stich',
#                 '..\\Data\\2017-06-09-stich',
#                 '..\\Data\\2017-06-13-stich',
#                 '..\\Data\\2017-06-20-1-stich',
                 '..\\Data\\2017-06-20-2-stich', 
                  '..\\Data\\2017-06-21-stich',
                 '..\\Data\\2017-06-26-1-stich',
                 '..\\Data\\2017-06-26-2-stich',
                 '..\\Data\\2017-06-27-stich',
                 '..\\Data\\2017-06-28-stich']
#comingtohelp = [ '..\\Data\\2017-06-20-2-stich', '..\\Data\\2017-07-24-3-stich', '..\\Data\\2017-07-25-1-stich' , '..\\Data\\2017-07-25-2-stich', '..\\Data\\2017-07-26-stich', '..\\Data\\2017-07-27-2-stich', '..\\Data\\2017-07-27-3-stich', '..\\Data\\2017-07-28-1-stich', '..\\Data\\2017-07-28-2-stich', '..\\Data\\2017-07-28-3-stich' ]
#for WORKDIR in glob.glob(WORKDIR2+'\\*-stich'):
for WORKDIR in comingtohelp:
    PATH = os.path.join(ROOT, WORKDIR, WORKDIR.split('\\')[2]+'-crop') 
    cropList = next(os.walk(PATH))[1]
    print('Start ', PATH)

    feaPos = collections.defaultdict(list)
    feaParam = collections.defaultdict(list)
    feaMaster = collections.defaultdict(list)
    for personId in cropList:
        
        if personId != 'H' and personId[-6:] != '-stich' and personId[-6:] != '-ratio':
            print('Processing',personId, 'features...')
            mp4Files = []
            cropDir = os.path.join(PATH, personId)
            
            #-- Get Cropped mp4files
            for mp4 in glob.glob(cropDir+'/*.MP4'):
                mp4Files.append(os.path.abspath(mp4))
            mp4Files = sorted(mp4Files)#??
            
            # Fetch facial-landmark features Action-Unit, Poses, Parameters
            feaDir = os.path.join(cropDir,'facial-landmarks')+'\\'
            feaList = glob.glob(feaDir+'*.txt')
            feaPoseList = [x for x in feaList if (os.path.basename(x).split('.')[0].split('_')[-1])=='pose']
            feaParamList = [x for x in feaList if (os.path.basename(x).split('.')[0].split('_')[-1])=='params']
            feaGazeList = [x for x in feaList if (os.path.basename(x).split('.')[0].split('_')[-1])=='gaze']

            
            # Get features and concatenate
            feaPosTmp = loadFea_pose(feaPoseList)
            feaParamTmp = loadFea_param(feaParamList)

            feaGazeTmp = loadFea_gaze(feaGazeList)
            fea = [feaPosTmp , feaParamTmp, feaGazeTmp]
            feaMasterTmp = pd.concat(fea, axis = 1) 
            feaMasterTmp = feaMasterTmp.dropna(how='any')
            if len(np.where(np.isnan(feaMasterTmp))[0])> 0:
                pdb.set_trace()
            
            
            #save to dict
            feaMaster[personId] = feaMasterTmp
        ib.dump(feaMaster, WORKDIR2+ "For_design\\" +'_'.join(WORKDIR.split('\\')[2][:-6].split('-')[1:])+'_feature.pkl')