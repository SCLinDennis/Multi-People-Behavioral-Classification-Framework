#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 21:19:13 2017

@author: chadyang
"""
import os
import glob
import cv2 #conda install -c menpo opencv3
import numpy as np
import imageio #pip install imageio
import joblib as ib
import operator
import collections



def getcolor(idx):
    colordict = {'A':(66, 80, 244), 'B':(65, 208, 244), 'C':(65, 244, 172), 'D':(244, 244, 65), 'E':(244, 65, 133), 'F':(205, 65, 244)}
    return colordict[idx]

def getname(idx):
    ppl_dict = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F'}
    return ppl_dict[idx]

def getminpeople(x_loacte, dic):
    person = 'None'    
    for key, value in dic.items():
        dis = abs(value-x_loacte)
        if person == 'None':
            mini = dis
            person = key
        else:
            if dis < mini:
                person = key
                mini = dis
            else:
                person = person
    return person, mini

def getlocate(peoplenumber, feature):
    tmp = collections.defaultdict(list)
    locate = collections.defaultdict(list)
    for pose in feature: 
        for j_idx, joint in enumerate(pose):
            if j_idx in [0, 1, 2, 3, 5, 6] and len(joint) == peoplenumber:
                joint.sort(key=operator.itemgetter(0))
                for idx, peo in enumerate(joint):
                    tmp[getname(idx)].append(peo[0])
    for idx in range(peoplenumber):
        locate[getname(idx)].append(np.median(tmp[getname(idx)]))
    return locate
#%% Parameters
Path = 'C:\\Lab\\Dennis\\Gamania\\Data\\2017-06-30-stich\\240_480\\'
PATH = os.path.join(Path, Path.split('\\')[-3]+'-pose-240x480') + '\\'
os.chdir(Path)
videoPath = '.\\360_0369_Stitch_XHC.MP4'
SAMPLERATE = 0.1 # Sampling rate for video pose estimation
PEOPLENUM = 5
PLOT = True


#%% Plot video
for videoPath in glob.glob(Path+'*.MP4'):
#    videoPath =  'C:\\Lab\\Dennis\\Gamania\\Data\\2017-06-30-stich\\240_480\\360_0373_Stitch_XHC.MP4'
    print('Processing ' + videoPath.split('\\')[-1])
    fea = ib.load(PATH + videoPath.split('\\')[-1].replace('.MP4', '.pickle'))
    ppl_locate = getlocate(PEOPLENUM, fea)
    vidName = os.path.basename(videoPath)
    vid = imageio.get_reader(videoPath, 'ffmpeg')
    fps = vid.get_meta_data()['fps']
    duration = vid.get_meta_data()['duration']
    timeStamp = .0
    fea_index = 0
    while timeStamp < duration:
        frameIdx = int(np.floor(timeStamp*fps))
        if (frameIdx+1 >= vid.get_meta_data()['nframes']): # break the loop preventing empty frame
            break
        frame = np.asarray(vid.get_data(frameIdx))
        pose = fea[timeStamp]
        
        # Plot video
        if PLOT:
            rearranged = frame[:, :, [2, 1, 0]].copy()
            for j_idx, joint in enumerate(pose):
                if j_idx not in [9, 10, 12, 13]:
                    joint.sort(key=operator.itemgetter(0))
                    bench = collections.defaultdict(list)
                    for ppl in joint:
                        man, distance = getminpeople(ppl[0], ppl_locate)
                        #===================wrong=============
#                        cv2.circle(rearranged, ppl[0:2], int(1), getcolor(man), 2)
#                        cv2.line(rearranged, (int(ppl_locate[man][0]), 0), (int(ppl_locate[man][0]), 240), getcolor(man), thickness=2)
                        #===================wrong=============
                        if man in bench.keys():
                            if list(distance)[0] < bench[man][0][0]:
                                tmp = []
                                tmp.append(list(distance))
                                tmp.append(ppl)
                                bench[man] = tmp
                        else:
                            bench[man].append(list(distance))                        
                            bench[man].append(ppl)
                    for man_correct, info in bench.items():
                        cv2.circle(rearranged, info[1][0:2], int(1), getcolor(man_correct), 2)
                        cv2.line(rearranged, (int(ppl_locate[man_correct][0]), 0), (int(ppl_locate[man_correct][0]), 240), getcolor(man_correct), thickness=2)
            cv2.imshow('Video', rearranged)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        timeStamp += SAMPLERATE
        fea_index += 1
        
cv2.destroyAllWindows()
