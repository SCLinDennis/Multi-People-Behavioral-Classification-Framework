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
from tqdm import tqdm
#%%
os.chdir('D:\\Lab\\Dennis\\Gamania\\Script')
WORKDIR = '..\\Script\\VideoFeature\\NewFeature2\\'
LABELDIR = '.\\VideoLabel\\VideoLabel\\VideoLabelNew\\'
ROOT = os.getcwd()
LABELTYPE = [  'Col', 'Act']
CTYPE = [ 0.001]
#PATH = os.path.join(ROOT, WORKDIR, WORKDIR.split('\\')[2]+'-crop')   
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

my_index = pd.MultiIndex(levels=[[],[]], labels=[[],[]], names=['C','Percemtile'])
Spears = pd.DataFrame(index=my_index, columns=[c for c in LABELTYPE])
#%% load Audio feature
'''
warnings.filterwarnings("ignore")
#nnn=['0410','0411','0418_1','0420','0424']
name=['0327','0330','0328_1','0328_2','0410','0411','0418_1','0420','0424']
#name=['0327','0330','0328_1','0328_2','0410','0411','0418_1','0420','0424','0524','0503_1','0503_2']
#name=['0327','0330','0328_1','0328_2','0411','0424']
feature='egemaps'
#feature='LLD_norm'


data=collections.defaultdict(list)
dataname = collections.defaultdict(list)
dataname=collections.defaultdict(list)
label=collections.defaultdict(list)
T_ts=collections.defaultdict(list)
T_peo=collections.defaultdict(list)
#la='Col'
la='Act'
want=collections.defaultdict(list)
error=[]
keep_name = ib.load(WORKDIR+ str(la)+ '_keep.pkl')
for n in name:
    ifiles  = glob.glob('D:\\Lab\\Dennis\\Gamania\\Jim\\labeled_wav\\'+n+'\\'+feature+'_wav\\*.mat')
    for idx,i in enumerate(ifiles):
        if i.split('\\')[-1][-5] != 'N' and i.split('\\')[-1][-5] != 'H'and i.split('\\')[-1][-5] != 'X'and i.split('\\')[-1][-5] != 'F':
            if  i.split('\\')[-1][0:-6] in keep_name[n][i.split('\\')[-1][-5]]:
                dataname[n].append(i.split('\\')[-1][0:-4])
                da=sp.io.loadmat(i)['Audio_data'].tolist()
                if len(np.where(np.isnan(da))[0])>0:
                    print(i.split('\\')[-1][0:-4])
                    error.extend(np.where(np.isnan(da))[1])
    #                print np.where(np.isnan(da))[1]
                data[n].append(da[0])
                want[n].append(idx)
                dataname[n].append(i.split('\\')[-1][0:-6])
'''
#%%for test
'''
kk_name = collections.defaultdict(list)
for kk in keep_name.keys():
    tttmp = []
    for kkk in keep_name[kk].keys():
        tttmp.extend(keep_name[kk][kkk])
#    kk_name[kk] = list(set(tttmp))
    kk_name[kk] = tttmp
    print("Difference: "+ kk+ str(list(set(kk_name[kk])-(set(data2[kk]))))) # b中有而a中没有的
'''
#%%
for label in LABELTYPE:
    feaAudio = ib.load(WORKDIR + label+ '_feaAudio.pkl')
    data = feaAudio
    feaVideo = ib.load(WORKDIR + label+ '_feaVideo.pkl') 
#    feaVid = collections.defaultdict(list)
#    for path in glob.glob(WORKDIR+'\\*'+str(label)+'_fea.pkl'):
#        feaComCut = ib.load(path)
#        fea = []
#        for key in feaComCut.keys():
#            fea.extend(feaComCut[key])
#        feaVid[path.split('\\')[4].split('.')[0][:-4]] = fea     
    for C_number in CTYPE:           
        for xx in range(20, 40, 10):
#            for yy in range(10, 110, 10):
            for yy in [50,60,70,80,90]:
                tru=[]
                pree=[]
                for n in tqdm(feaAudio.keys()): 
                    tralis=list(set(feaAudio.keys())-{n})
                    train_data_v=[]
                    train_data_a=[]
                    train_label=[]
                    train_data =[]
    #                test_data_a = []
                    for i in tralis:
                        data = np.array(feaAudio[i])
                        data2= np.array(feaVideo[i])
                        train_data_a.extend(data[:,:-1].tolist())
                        train_data_v.extend(data2[:, :-1].tolist())
                        train_label.extend(data2[:, -1].tolist())   
                #    train_data= np.delete(np.array(train_data),error,1)
                    [train_data_v, mean, std] = zsco(np.array(train_data_v), 0)        
                    data = np.array(feaAudio[n])
                    data2= np.array(feaVideo[n])
                    test_data_v = data2[:, :-1].tolist()
                    test_data_v = passzsco(np.array(test_data_v), 0, mean, std)
                    test_label= data2[:, -1].tolist()
                    test_data_a = data[:, :-1].tolist()                    
                    
                    
                    ps_v = SelectPercentile(f_classif, percentile = xx)
                    train_data_v = ps_v.fit_transform(train_data_v, train_label)
                    test_data_v = ps_v.transform(test_data_v)
                    
                    ps_a = SelectPercentile(f_classif, percentile = yy)
                    train_data_a = ps_a.fit_transform(np.array(train_data_a), train_label)
                    test_data_a = ps_a.transform(test_data_a)
                    train_data = np.dot(train_data_v, train_data_v.T)+np.dot(train_data_a, train_data_a.T)
                    test_data = np.dot(test_data_v, train_data_v.T)+np.dot(test_data_a, train_data_a.T)
                    clf=SVC(kernel='precomputed',class_weight='balanced',C=C_number)
                    clf.fit(train_data, train_label)
                    pre=clf.predict(test_data)
                    tru.extend(test_label)
                    pree.extend(pre)
                       
    #            print(cm(tru,pree))
            #        print(UAR(tru,pree))
                accu = np.int(UAR(tru,pree, average='macro')*1000)/10.0
                # Spears.set_value((C_number,xx), label, accu)
                print('Label:',label,'C:',C_number,'per_v:', xx, 'per_a:', yy,'Accu:',accu)
            
            
#%% Save result
# Spears.to_csv('./baseline.csv')
