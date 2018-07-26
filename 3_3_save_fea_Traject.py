# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:00:44 2017

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
import re
import gzip
from numpy import matlib as matlib
#%%
#os.chdir('D:\\Lab\\Dennis\\Gamania\\Script')
os.chdir('G:\\GAMANIA\\Data_smaller')
WORKDIR = '..\\Data_smaller\\'
OUTDIR = 'D:\\Lab\\Dennis\\Gamania\\Data\\DT_feature\\'
ROOT = os.getcwd()


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
    df = normalize(df)     
    return df

def normalize(DataFrame):
    df = (DataFrame - DataFrame.mean(axis = 0))/DataFrame.std(axis = 0)
    return df

def zsco(arg,axis):
    lenn,widd=arg.shape
    uper=(arg-matlib.repmat(np.mean(arg,axis),lenn,1)) 
    lower=matlib.repmat((np.std(arg,axis)),lenn,1)
    return uper/lower,np.mean(arg,axis),np.std(arg,axis)

def loadFeaAu(feaList, reference):
    df = []
    for feaPath in sorted(feaList):
        print(feaPath+'\n')
        fea = pd.read_table(feaPath, delimiter=', ', header = 0, engine = 'python')
        fea['frame'] = fea.index.values + 1
        fea.set_index('frame', inplace = True)
        df.append(fea)
    df = pd.concat(df, ignore_index = True)
    successIdx = df['success'] != 0
    old_fea = df.ix[successIdx]
    replace = old_fea.mean(axis = 0)    
    referenceIdx = reference.index.values    
    df2 = df.ix[referenceIdx]
    df2.loc[df2['success']==0] = [replace]*len(df2.loc[df2['success']==0])
    df2.drop('success', axis=1, inplace=True)
    df2 = normalize(df2)
    return df2 

def parseCMLFea(feaPath):
    feaFace = pd.read_table(feaPath,delimiter=', ', index_col=0, header=0, engine='python')
    return feaFace

def opener(filename):
    f = open(filename,'rb')
    if (f.read(2) == '10'):
        f.seek(0)
        return gzip.GzipFile(fileobj=f)
    else:
        f.seek(0)
        return f
    
def gzipFea(trajectPath):
    f = opener(trajectPath)
    s = f.read().decode()
    line = re.split('\n',s)
    feats = np.zeros([len(line)-1, len(line[0].split('\t'))-1])
    for i in range(len(line)-1):
        current_line = line[i].split('\t')
        for j in range(len(current_line)-1):
            feats[i][j] = float(current_line[j])
    return feats

def loadTraject(TrajectList):
    for idx, feaPath in enumerate(sorted(TrajectList)):
        print(feaPath+'\n')
        fea = gzipFea(feaPath)
        fea[:, 0:1] += 688*30*idx
        if idx == 0:
            DT = fea
        elif idx != 0:
            DT = np.vstack((DT, fea))
    DT_tmp = DT[:, 1:]
    [DT_tmp, mean, std] = zsco(DT_tmp, 0)
    DT_normalized = np.hstack((DT[:, 0:1], DT_tmp))
    return DT_normalized
#%%
comingtohelp = [  '2017-07-11-1-stich', '2017-07-11-2-stich', '2017-07-12-1-stich', '2017-07-12-2-stich', '2017-07-12-3-stich', '2017-07-13-1-stich', '2017-07-13-2-stich', '2017-07-14-2-stich', '2017-07-19-1-stich' , '2017-07-19-2-stich', '2017-07-19-3-stich']
for WORKDIR in glob.glob(WORKDIR+'\\*-stich'):
    if WORKDIR.split('\\')[-1] not in comingtohelp:
        PATH = os.path.join(ROOT, WORKDIR, WORKDIR.split('\\')[2]+'-crop') 
        cropList = next(os.walk(PATH))[1]
        print('Start ', WORKDIR)
    
        feaPos = collections.defaultdict(list)
        feaMaster = collections.defaultdict(list)
        for personId in cropList:           
            if personId[-5:] != 'stich':
                print('Processing',personId, 'features...')
                cropDir = os.path.join(PATH, personId)
                
                # Fetch facial-landmark features Action-Unit, Poses, Parameters
                feaDir = os.path.join(cropDir,'output')+'\\'
                feaList = glob.glob(feaDir+'*.gz')

     
                
                # Get features and concatenate
                feaTrajectTmp = loadTraject(feaList)
                if len(np.where(np.isnan(feaTrajectTmp))[0])> 0:
                    pdb.set_trace()
                
                
                #save to dict
                feaMaster[personId] = feaTrajectTmp
        ib.dump(feaMaster, OUTDIR +'_'.join(WORKDIR.split('\\')[2][:-6].split('-')[1:])+'_DT_feature.pkl')
