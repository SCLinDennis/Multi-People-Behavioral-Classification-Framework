# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 12:10:34 2017

@author: dennis60512
"""

import os, glob, sys
import subprocess as sp
import multiprocessing  as mp
import pandas as pd
import collections
import joblib as ib
from collections import defaultdict

#%%
os.chdir('C:\\Lab\\Dennis\\Gamania\\')
WORKDIR2 = '.\\Data\\'
ROOT = os.getcwd()
OUTDIR = 'F:\\HGY\\'

#%% Load in data and Z-score for each person
def loadFea(feaList):
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
    if 'timestamp' in df.columns:
        df.drop('timestamp', axis=1, inplace=True)
    df = normalize(df)     
    return df

def normalize(DataFrame):
    df = (DataFrame - DataFrame.mean(axis = 0))/DataFrame.std(axis = 0)
    return df

def loadFea_v2(feaList):
    df = []
    for feaPath in sorted(feaList):
        print(feaPath+'\n')
        fea = parseCMLFea(feaPath)
        df.append(fea)
    df = pd.concat(df, ignore_index=True)  
    successIdx = df['success'] != 0
    df= df.ix[successIdx]
    df.drop('success', axis=1, inplace=True) 
    df.drop('timestamp', axis=1, inplace=True)
    df_right = df.ix[:, -35:]
    df_left = df.ix[:, 0]
    df2 = pd.concat([df_left, df_right], axis = 1)
    df2 = normalize(df2)
    return df2 

def parseCMLFea(feaPath):
    feaFace = pd.read_table(feaPath,delimiter=', ', index_col=0, header=0, engine='python')
    return feaFace
#%%
TOTAL = ['.\\Data\\2017-07-14-1-stich']
for WORKDIR in TOTAL:
#for WORKDIR in glob.glob(WORKDIR2+'\\*-stich'):
    if WORKDIR != '.\\Data\\2017-05-24-stich' :
        PATH = os.path.join(ROOT, WORKDIR, WORKDIR.split('\\')[-1]+'-crop') 
        cropList = next(os.walk(PATH))[1]
        print('Start ', PATH)
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
                
                feaAuList = [x for x in feaList if (os.path.basename(x).split('.')[0].split('_')[-1])=='au']
                # Get features and concatenate
                feaAuTmp = loadFea(feaAuList)

                #save to dict
                feaMaster[personId] = feaAuTmp
        ib.dump(feaMaster, OUTDIR+ 'For_HGY\\' +'_'.join(WORKDIR.split('\\')[-1][:-6].split('-')[1:])+'_feature.pkl')
