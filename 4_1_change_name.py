# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 20:13:37 2017

@author: dennis60512
"""
import collections
import joblib as ib

#%%
os.chdir('C:\\Lab\\Dennis\\Gamania\\Data\\')
WORKDIR2 = '.\\For_design\\'
OUTDIR = '.\\For_designNew\\'
ROOT = os.getcwd()
def swap(Changed, tmp, fr, to):
    for i, fro in enumerate(fr):
        tmp[fro] = Changed[to[i]]
    return tmp
#%%
for WORKDIR in glob.glob(WORKDIR2+ '*.pkl'):
#for WORKDIR in [ '.\\Test_Feature\\AU_feature\\08_02_feature.pkl']:
    Changed = collections.defaultdict(list)
    tmp = collections.defaultdict(list)
    Changed = ib.load(WORKDIR)
    check = 1
    if WORKDIR.split('\\')[-1] == '06_09_feature.pkl':
        fro = ['B', 'C', 'A']
        to = ['A', 'C', 'B']
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '06_13_feature.pkl':
        fro = ['B', 'C', 'A','D']
        to = ['D', 'C', 'A', 'B']
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '05_24_feature.pkl':
        fro = ['A', 'B', 'C', 'D','E']
        to = ['C', 'D', 'B', 'A', 'E']
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '06_20_1_feature.pkl':
        fro = [ 'A','B', 'C','D']
        to = ['B', 'C', 'D', 'A']        
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '06_20_2_feature.pkl': 
        fro = ['A', 'B', 'C']
        to = [ 'B', 'C', 'A']
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '06_21_feature.pkl':
        fro = ['A', 'B', 'C','D']
        to = ['C', 'D', 'A', 'B']       
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '06_26_1_feature.pkl':
        fro = ['C', 'B', 'A','D']
        to = ['A', 'D', 'C', 'B']       
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '06_26_2_feature.pkl':
        fro = ['B', 'A']
        to = ['C', 'B']       
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '06_27_feature.pkl':
        fro = ['C', 'A', 'B']
        to = [ 'A', 'B', 'C']          
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '06_28_feature.pkl':
        fro = ['D', 'A', 'C','B']
        to = ['B', 'C', 'A', 'D']          
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '06_29_feature.pkl':
        fro = ['A', 'B', 'D','C']
        to = ['A', 'B', 'D', 'C']          
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '06_30_feature.pkl':
        fro = ['D', 'A', 'C','B']
        to = ['B', 'C', 'A', 'D']          
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '07_03_1_feature.pkl':
        fro = ['C', 'A', 'B']
        to = ['B', 'C', 'A']
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '07_03_2_feature.pkl':
        fro = [ 'A', 'B', 'C', 'D']
        to = [ 'C', 'D', 'A', 'B']
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '07_05_feature.pkl':
        fro = ['A', 'B', 'C', 'D']
        to = ['A', 'B', 'C', 'D']
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '07_06_1_feature.pkl':
        fro = ['A', 'B', 'C']
        to = ['C', 'A', 'B']
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '07_06_2_feature.pkl':
        fro = ['A', 'B', 'C']
        to = [ 'C', 'A', 'B']
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '07_07_feature.pkl':
        fro = ['A', 'B', 'C']
        to = ['B', 'A', 'C']
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '07_14_1_feature.pkl':
        fro = ['A', 'B', 'C', 'D', 'E']
        to = [ 'A', 'B', 'C', 'D', 'E']
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '07_20_1_feature.pkl':
        fro = [ 'A','B', 'C','D']
        to = ['C', 'D', 'A', 'B']        
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '07_20_2_feature.pkl': 
        fro = ['A', 'B', 'C', 'D','E']
        to = ['A', 'B', 'C', 'D','E']
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '07_20_3_feature.pkl':
        fro = ['A', 'B', 'C','D']
        to = ['C', 'D', 'A', 'B']       
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '07_21_1_feature.pkl':
        fro = ['A', 'B', 'C','D']
        to = ['C', 'D', 'A', 'B']      
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '07_21_2_feature.pkl':
        fro = [ 'A', 'B', 'C'] 
        to = [ 'A', 'B', 'C']      
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '07_21_3_feature.pkl':
        fro =  ['A', 'B', 'C','D']
        to = [ 'B', 'A', 'C', 'D']          
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '07_24_1_feature.pkl':
        fro = ['A', 'B', 'C','D']
        to = ['C', 'D', 'A', 'B']           
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '07_24_2_feature.pkl':
        fro = ['A', 'B','C']
        to = ['A', 'B', 'C']          
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '07_24_3_feature.pkl':
        fro = ['A', 'B', 'C','D']
        to = ['C', 'D', 'A', 'B']        
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '07_25_1_feature.pkl':
        fro = ['A', 'B', 'C', 'D', 'E']
        to = [ 'D', 'E', 'A', 'B', 'C']
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '07_25_2_feature.pkl':
        fro = ['A', 'B', 'C', 'D', 'E']
        to = ['A', 'B', 'C', 'D', 'E']
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '07_26_feature.pkl':
        fro = ['A', 'B', 'C', 'D']
        to = [ 'C', 'D', 'A', 'B']
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '07_27_2_feature.pkl':
        fro = ['A', 'B', 'C']
        to = [ 'B','C', 'A']
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '07_27_3_feature.pkl':
        fro = ['A', 'B', 'C', 'D']
        to = [ 'C', 'D', 'A', 'B']
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '07_28_1_feature.pkl':
        fro = ['A', 'B', 'C']
        to = ['B', 'C', 'A']
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '07_28_2_feature.pkl':
        fro = ['A', 'B', 'C', 'D', 'E']
        to = [ 'A', 'B', 'C', 'D', 'E']
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '07_28_3_feature.pkl':
        fro = ['A', 'B', 'C']
        to = [  'C','A', 'B']
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '09_12_feature.pkl':
        fro = ['A', 'B', 'C']
        to = ['B', 'C', 'A']
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '09_21_feature.pkl':
        fro = ['A', 'B', 'C', 'D']
        to = [ 'C', 'D', 'A', 'B']
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '10_03_feature.pkl':
        fro = ['A', 'B', 'C', 'D', 'E']
        to = ['A', 'B', 'C', 'D', 'E']
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '08_22_2_feature.pkl':
        fro = ['A', 'B', 'C', 'D']
        to = ['C', 'D', 'A', 'B']
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '08_22_1_feature.pkl':
        fro = ['A', 'B', 'C', 'D']
        to = ['A', 'B', 'C', 'D']
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '08_03_feature.pkl':
        fro = ['A', 'B', 'C', 'D', 'E']
        to = ['A', 'B', 'C', 'D', 'E']
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    elif WORKDIR.split('\\')[-1] == '08_02_feature.pkl':
        fro = ['A', 'B', 'C', 'D']
        to = ['C', 'D', 'A', 'B']
        print('Finish swaping ' + WORKDIR.split('\\')[-1])
    else: 
        check = 0
        print('There is no need to swap '  + WORKDIR.split('\\')[-1])
    if check == 1:
        tmp = swap(Changed, tmp, fro, to)
        ib.dump(tmp, OUTDIR+ WORKDIR.split('\\')[-1])
    else:
        ib.dump(Changed, OUTDIR+ WORKDIR.split('\\')[-1])
    
