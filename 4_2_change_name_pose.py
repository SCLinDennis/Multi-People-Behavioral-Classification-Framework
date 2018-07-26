# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 20:13:37 2017

@author: dennis60512
"""


import collections
import joblib as ib

os.chdir('D:\\Lab\\Dennis\\Gamania\\Script\\')
WORKDIR2 = '..\\Features\\pose-crop\\'
OUTDIR = '..\\Features\\pose-crop-new\\'
ROOT = os.getcwd()
def swap(Changed, tmp, fr, to):
    for i, fro in enumerate(fr):
        tmp[fro] = Changed[to[i]]
    return tmp

def loadHGYfeature(feature):
    new_feature = collections.defaultdict(list)
    for key, value in feature.items():
        for key2, value2 in value.items(): #value2 = test
            for tes in value2:
                out = []
                for idx, tes2 in enumerate(tes):
                    if idx not in [8, 9, 10, 11, 12, 13]:
                        if len(tes2) != 0:
                            out.extend(list(tes2[0][0:2]))
                        else:
                            out.extend([0, 0]) 
                new_feature[key].append(out)
    return new_feature
def loadHGYfeature2(feature):
    new_feature = collections.defaultdict(list)
    for key, value in feature.items():
        tmp = [0]*22
        for key2, value2 in value.items(): #value2 = test
            for tes in value2:
                out = []
                if len(tes[0]) != 0 and len(tes[1]) != 0:
                    out.extend([tes[1][0][0] - tes[0][0][0], tes[1][0][1] - tes[0][0][1]])
                    tmp[0:2] = [tes[1][0][0] - tes[0][0][0], tes[1][0][1] - tes[0][0][1]]
                else:
                    out.extend(tmp[0:2])
                if len(tes[1]) != 0 and len(tes[2]) != 0:
                    out.extend([tes[2][0][0] - tes[1][0][0], tes[2][0][1] - tes[1][0][1]])
                    tmp[2:4] = [tes[2][0][0] - tes[1][0][0], tes[2][0][1] - tes[1][0][1]]
                else:
                    out.extend(tmp[2:4])                
                if len(tes[2]) != 0 and len(tes[3]) != 0:
                    out.extend([tes[3][0][0] - tes[2][0][0], tes[3][0][1] - tes[2][0][1]])
                    tmp[4:6] = [tes[3][0][0] - tes[2][0][0], tes[3][0][1] - tes[2][0][1]]
                else:
                    out.extend(tmp[4:6])
                if len(tes[3]) != 0 and len(tes[4]) != 0:
                    out.extend([tes[4][0][0] - tes[3][0][0], tes[4][0][1] - tes[3][0][1]])
                    tmp[6:8] = [tes[4][0][0] - tes[3][0][0], tes[4][0][1] - tes[3][0][1]]
                else:
                    out.extend(tmp[6:8])
                if len(tes[0]) != 0 and len(tes[5]) != 0:
                    out.extend([tes[5][0][0] - tes[0][0][0], tes[5][0][1] - tes[0][0][1]])
                    tmp[8:10] = [tes[5][0][0] - tes[0][0][0], tes[5][0][1] - tes[0][0][1]]
                else:
                    out.extend(tmp[8:10])
                if len(tes[5]) != 0 and len(tes[6]) != 0:
                    out.extend([tes[6][0][0] - tes[5][0][0], tes[6][0][1] - tes[5][0][1]])
                    tmp[10:12] = [tes[6][0][0] - tes[5][0][0], tes[6][0][1] - tes[5][0][1]]
                else:
                    out.extend(tmp[10:12])
                if len(tes[6]) != 0 and len(tes[7]) != 0:
                    out.extend([tes[7][0][0] - tes[6][0][0], tes[7][0][1] - tes[6][0][1]])
                    tmp[12:14] = [tes[7][0][0] - tes[6][0][0], tes[7][0][1] - tes[6][0][1]]
                else:
                    out.extend(tmp[12:14])
                if len(tes[0]) != 0 and len(tes[14]) != 0:
                    out.extend([tes[14][0][0] - tes[0][0][0], tes[14][0][1] - tes[0][0][1]])
                    tmp[14:16] = [tes[14][0][0] - tes[0][0][0], tes[14][0][1] - tes[0][0][1]]
                else:
                    out.extend(tmp[14:16])
                if len(tes[0]) != 0 and len(tes[15]) != 0:
                    out.extend([tes[15][0][0] - tes[0][0][0], tes[15][0][1] - tes[0][0][1]])
                    tmp[16:18] = [tes[15][0][0] - tes[0][0][0], tes[15][0][1] - tes[0][0][1]]
                else:
                    out.extend(tmp[16:18])
                if len(tes[14]) != 0 and len(tes[16]) != 0:
                    out.extend([tes[16][0][0] - tes[14][0][0], tes[16][0][1] - tes[14][0][1]])
                    tmp[18:20] = [tes[16][0][0] - tes[14][0][0], tes[16][0][1] - tes[14][0][1]]
                else:
                    out.extend(tmp[18:20])
                if len(tes[15]) != 0 and len(tes[17]) != 0:
                    out.extend([tes[17][0][0] - tes[15][0][0], tes[17][0][1] - tes[15][0][1]])
                    tmp[20:22] = [tes[17][0][0] - tes[15][0][0], tes[17][0][1] - tes[15][0][1]]
                else:
                    out.extend(tmp[20:22])
                new_feature[key].append(out)
    return new_feature
#%%
if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)
helpme = ['2017-07-11-1-ratio.pickle', '2017-07-11-2-ratio.pickle', '2017-07-12-1-ratio.pickle', '2017-07-12-2-ratio.pickle', '2017-07-12-3-ratio.pickle', '2017-07-13-1-ratio.pickle','2017-07-13-2-ratio.pickle', '2017-07-14-1-ratio.pickle', '2017-07-14-2-ratio.pickle', '2017-07-18-ratio.pickle', '2017-07-19-1-ratio.pickle', '2017-07-19-2-ratio.pickle', '2017-07-19-3-ratio.pickle']
for WORKDIR in glob.glob(WORKDIR2+ '*.pickle'):
    if WORKDIR.split('\\')[-1] in helpme:
        Changed = collections.defaultdict(list)
        tmp = collections.defaultdict(list)
        feature = ib.load(WORKDIR)
        Changed = loadHGYfeature2(feature)
        check = 1
        if WORKDIR.split('\\')[-1] == '2017-06-09-ratio.pickle':
            fro = ['B', 'C', 'A']
            to = ['A', 'C', 'B']
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-06-13-ratio.pickle':
            fro = ['B', 'C', 'A','D']
            to = ['D', 'C', 'A', 'B']
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-05-24-ratio.pickle':
            fro = ['A', 'B', 'C', 'D','E']
            to = ['C', 'D', 'B', 'A', 'E']
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-06-20-1-ratio.pickle':
            fro = [ 'A','B', 'C','D']
            to = ['B', 'C', 'D', 'A']        
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-06-20-2-ratio.pickle': 
            fro = ['A', 'B', 'C']
            to = [ 'B', 'C', 'A']
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-06-21-ratio.pickle':
            fro = ['A', 'B', 'C','D']
            to = ['C', 'D', 'A', 'B']       
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-06-26-1-ratio.pickle':
            fro = ['C', 'B', 'A','D']
            to = ['A', 'D', 'C', 'B']       
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-06-26-2-ratio.pickle':
            fro = ['B', 'A']
            to = ['C', 'B']       
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-06-27-ratio.pickle':
            fro = ['C', 'A', 'B']
            to = [ 'A', 'B', 'C']          
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-06-28-ratio.pickle':
            fro = ['D', 'A', 'C','B']
            to = ['B', 'C', 'A', 'D']          
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-06-29-ratio.pickle':
            fro = ['A', 'B', 'D','C']
            to = ['A', 'B', 'D', 'C']          
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-06-30-ratio.pickle':
            fro = ['D', 'A', 'C','B']
            to = ['B', 'C', 'A', 'D']          
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-07-03-1-ratio.pickle':
            fro = ['C', 'A', 'B']
            to = ['B', 'C', 'A']
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-07-03-2-ratio.pickle':
            fro = [ 'A', 'B', 'C', 'D']
            to = [ 'C', 'D', 'A', 'B']
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-07-05-ratio.pickle':
            fro = ['A', 'B', 'C', 'D']
            to = ['A', 'B', 'C', 'D']
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-07-06-1-ratio.pickle':
            fro = ['A', 'B', 'C']
            to = ['C', 'A', 'B']
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-07-06-2-ratio.pickle':
            fro = ['A', 'B', 'C']
            to = [ 'C', 'A', 'B']
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-07-07-ratio.pickle':
            fro = ['A', 'B', 'C']
            to = ['B', 'A', 'C']
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-07-14-1-ratio.pickle':
            fro = ['A', 'B', 'C', 'D', 'E']
            to = [ 'A', 'B', 'C', 'D', 'E']
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-07-20-1-ratio.pickle':
            fro = [ 'A','B', 'C','D']
            to = ['C', 'D', 'A', 'B']        
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-07-20-2-ratio.pickle': 
            fro = ['A', 'B', 'C', 'D','E']
            to = ['A', 'B', 'C', 'D','E']
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-07-20-3-ratio.pickle':
            fro = ['A', 'B', 'C','D']
            to = ['C', 'D', 'A', 'B']       
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-07-21-1-ratio.pickle':
            fro = ['A', 'B', 'C','D']
            to = ['C', 'D', 'A', 'B']      
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-07-21-2-ratio.pickle':
            fro = [ 'A', 'B', 'C'] 
            to = [ 'A', 'B', 'C']      
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-07-21-3-ratio.pickle':
            fro =  ['A', 'B', 'C','D']
            to = [ 'B', 'A', 'C', 'D']          
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-07-24-1-ratio.pickle':
            fro = ['A', 'B', 'C','D']
            to = ['C', 'D', 'A', 'B']           
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-07-24-2-ratio.pickle':
            fro = ['A', 'B','C']
            to = ['A', 'B', 'C']          
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-07-24-3-ratio.pickle':
            fro = ['A', 'B', 'C','D']
            to = ['C', 'D', 'A', 'B']        
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-07-25-1-ratio.pickle':
            fro = ['A', 'B', 'C', 'D', 'E']
            to = [ 'D', 'E', 'A', 'B', 'C']
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-07-25-2-ratio.pickle':
            fro = ['A', 'B', 'C', 'D', 'E']
            to = ['A', 'B', 'C', 'D', 'E']
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-07-26-ratio.pickle':
            fro = ['A', 'B', 'C', 'D']
            to = [ 'C', 'D', 'A', 'B']
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-07-27-2-ratio.pickle':
            fro = ['A', 'B', 'C']
            to = [ 'B','C', 'A']
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-07-27-3-ratio.pickle':
            fro = ['A', 'B', 'C', 'D']
            to = [ 'C', 'D', 'A', 'B']
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-07-28-1-ratio.pickle':
            fro = ['A', 'B', 'C']
            to = ['B', 'C', 'A']
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-07-28-2-ratio.pickle':
            fro = ['A', 'B', 'C', 'D', 'E']
            to = [ 'A', 'B', 'C', 'D', 'E']
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-07-28-3-ratio.pickle':
            fro = ['A', 'B', 'C']
            to = [  'C','A', 'B']
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-09-12-ratio.pickle':
            fro = ['A', 'B', 'C']
            to = ['B', 'C', 'A']
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-09-21-ratio.pickle':
            fro = ['A', 'B', 'C', 'D']
            to = [ 'C', 'D', 'A', 'B']
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-10-03-ratio.pickle':
            fro = ['A', 'B', 'C', 'D', 'E']
            to = ['A', 'B', 'C', 'D', 'E']
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-08-22-2-ratio.pickle':
            fro = ['A', 'B', 'C', 'D']
            to = ['C', 'D', 'A', 'B']
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-08-22-1-ratio.pickle':
            fro = ['A', 'B', 'C', 'D']
            to = ['A', 'B', 'C', 'D']
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-08-03-ratio.pickle':
            fro = ['A', 'B', 'C', 'D', 'E']
            to = ['A', 'B', 'C', 'D', 'E']
            print('Finish swaping ' + WORKDIR.split('\\')[-1])
        elif WORKDIR.split('\\')[-1] == '2017-08-02-ratio.pickle':
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
    
