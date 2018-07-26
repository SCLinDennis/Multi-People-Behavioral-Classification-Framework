# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 10:46:58 2017

@author: dennis60512
"""
import os
import glob
import json
import cv2 #conda install -c menpo opencv3
import numpy as np
import imageio #pip install imageio
import joblib as ib
import collections
from collections import Counter

#%%
def getcolor(idx):
    colordict = {'A':(66, 80, 244), 'B':(65, 208, 244), 'C':(65, 244, 172), 'D':(244, 244, 65), 'E':(244, 65, 133), 'F':(205, 65, 244)}
    return colordict[idx]

def getname(idx):
    ppl_dict = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F'}
    return ppl_dict[idx]

def getgesture(idx):
    gesture_dict = {'l_hand':0, 'pose':1, 'r_hand':2, 'face':3, 'lp_hand': 4, 'lp_hand & face': 5}
    return gesture_dict[idx]

def getgesturename(idx):
    gesture_dict = {0:'hand_left_keypoints', 1:'pose_keypoints', 2:'hand_right_keypoints', 3:'face_keypoints'}
    return gesture_dict[idx]

def getminpeople(x_loacte, dic):
    person = 'None'    
    for key, value in dic.items():
        dis = abs(value[0]-x_loacte)
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

def getfinger(locate, target):
    if len(target) != 63:
        print('The length of feature must be 63')
    location = []
    for idx, loc in enumerate(target):
        if idx < 15: 
            if idx%3 == 0:
                x = int(loc)
            elif idx%3 == 1:
                y = int(loc)
            elif idx%3 == 2:
                location.append((x, y, 'A'))
        elif idx >= 15 and idx < 27:
            if idx%3 == 0:
                x = int(loc)
            elif idx%3 == 1:
                y = int(loc)
            elif idx%3 == 2:
                location.append((x, y, 'B'))
        elif idx >= 27 and idx < 39:
            if idx%3 == 0:
                x = int(loc)
            elif idx%3 == 1:
                y = int(loc)
            elif idx%3 == 2:
                location.append((x, y, 'C'))
        elif idx >= 39 and idx < 51:
            if idx%3 == 0:
                x = int(loc)
            elif idx%3 == 1:
                y = int(loc)
            elif idx%3 == 2:
                location.append((x, y, 'D'))
        elif idx >= 51: 
            if idx%3 == 0:
                x = int(loc)
            elif idx%3 == 1:
                y = int(loc)
            elif idx%3 == 2:
                location.append((x, y, 'E')) 
    return location

def mean(lst):
    return sum(lst)/float(len(lst))

def Most_Common(lst):
    first_element = Counter([z for (x, y, z, a) in lst])
    average = mean([a for (x, y, z, a) in lst])
    return first_element.most_common(1)[0][0], average

def getfeature(typ, fea_dict):
    fea_type_idx = getgesture(typ)   
    if fea_type_idx < 4: 
        for part in fea_dict:
            if part[0] == getgesturename(fea_type_idx):
                out_lst = part[1]
    elif fea_type_idx == 4:
        for part in fea_dict:
            if part[0] == getgesturename(0):
                out_lst = part[1]     
        for part in fea_dict:
            if part[0] == getgesturename(2):
                out_lst.extend(part[1])
    elif fea_type_idx == 5:
        for part in fea_dict:
            if part[0] == getgesturename(0):
                out_lst = part[1]     
        for part in fea_dict:
            if part[0] == getgesturename(1):
                out_lst.extend(part[1])
        for part in fea_dict:
            if part[0] == getgesturename(2):
                out_lst.extend(part[1])
    return out_lst

def l_hand_position(lis, point_num):
    return lis[3*point_num:3*point_num+2]

def pose_position(lis, point_num):
    return lis[63+3*point_num:63+3*point_num+2]

def r_hand_position(lis, point_num):
    return lis[63+54+3*point_num:63+54+3*point_num+2]


def get_2d_dis(lis, typ1, typ2, point_num1, point_num2):
    if typ1 == 'l_hand':
        lis1 = l_hand_position(lis, point_num1)      
    elif typ1 == 'pose':
        lis1 = pose_position(lis, point_num1)
    elif typ1 == 'r_hand':
        lis1 = r_hand_position(lis, point_num1)
    if typ2 == 'l_hand':
        lis2 = l_hand_position(lis, point_num2)        
    elif typ2 == 'pose':
        lis2 = pose_position(lis, point_num2)
    elif typ2 == 'r_hand':
        lis2 = r_hand_position(lis, point_num2)
#    return lis1, lis2
    dis1 = lis1[0] - lis2[0]
    dis2 = lis1[1] - lis2[1]
    dis_out = (dis1**2 + dis2**2)**0.5
    return dis_out

def get_down_eight(lis, typ1, typ2, typ3, point_num1, point_num2, point_num3):
    if typ1 == 'l_hand':
        lis1 = l_hand_position(lis, point_num1)      
    elif typ1 == 'pose':
        lis1 = pose_position(lis, point_num1)
    elif typ1 == 'r_hand':
        lis1 = r_hand_position(lis, point_num1)
    if typ2 == 'l_hand':
        lis2 = l_hand_position(lis, point_num2)        
    elif typ2 == 'pose':
        lis2 = pose_position(lis, point_num2)
    elif typ2 == 'r_hand':
        lis2 = r_hand_position(lis, point_num2)
    if typ3 == 'l_hand':
        lis3 = l_hand_position(lis, point_num3)
        lis4 = r_hand_position(lis, point_num3)
    dex = (pose_position(lis, 0)[0] + pose_position(lis, 1)[0])/2
    dey = (pose_position(lis, 0)[1] + pose_position(lis, 1)[1])/2
    if ((dex - lis1[0])**2 + (dey - lis1[1])**2)**0.5 < 3 or ((dex - lis2[0])**2 + (dey - lis2[1])**2)**0.5 < 3 or  ((dex - lis3[0])**2 + (dey - lis3[1])**2)**0.5 < 3  or ((dex - lis4[0])**2 + (dey - lis4[1])**2)**0.5 < 3:
        whether = True
    else:
        whether = False
    return whether
#%% Parameters
Path = 'C:\\Lab\\Dennis\\Gamania\\Data\\2017-06-30-stich\\'
PATH = os.path.join(Path,'240_480', Path.split('\\')[-2]+'-pose-240x480') + '\\'
os.chdir(Path)
videoPath = '.\\240_480\\360_0370_Stitch_XHC.MP4'
savePath = '.\\shichen\\' + videoPath.split('\\')[-1][0:8] + '\\' + videoPath.split('\\')[-1].replace('.MP4', '_pose\\')
SAMPLERATE = 1/30 # Sampling rate for video pose estimation
PLOT = True
PEOPLENUM = 5
FEATURETYPE = 'lp_hand & face' #Please specify {'l_hand', 'r_hand', 'face', 'pose', 'lp_hand', 'lp_hand & pose'}
#%%
# Fetch logs
poseLogs = sorted(glob.glob(savePath+'*.json'))
pose = []
for poselog in poseLogs:
    # personlogList = []
    personlogs = json.load(open(poselog))['people']
    personlogList = [list(personlog.items()) for personlog in personlogs]
    pose.append(personlogList)

#poseDict.update({vidName:pose})
tmp = collections.defaultdict(list)
locate = collections.defaultdict(list)
for fea in pose:
    if len(fea) == PEOPLENUM:
        list_tmp = []   
        for ppl_idx, fea_ppl in enumerate(fea):
            for part in fea_ppl:
                if part[0] == getgesturename(1):                    
                    loc = part[1][0]
                    x_locate = int(loc)
                    list_tmp.append(x_locate)
        for idx in range(PEOPLENUM):
            name = getname(idx)
            tmp[name].append(sorted(list_tmp)[idx])
for idx in range(PEOPLENUM):
    name = getname(idx)
    locate[name].append(np.median(tmp[name]))
#%%
vidName = os.path.basename(videoPath)
vid = imageio.get_reader(videoPath, 'ffmpeg')
fps = vid.get_meta_data()['fps']
duration = vid.get_meta_data()['duration']

vidPose = []
prevPeak = []
timeStamp = .0
shoulder = collections.defaultdict(int)
for i in range(PEOPLENUM):
    shoulder[getname(i)] = 0
while timeStamp < duration :
    frameIdx = int(np.floor(timeStamp*fps))
    if (frameIdx+1 >= vid.get_meta_data()['nframes']): # break the loop preventing empty frame
        break
    frame = np.asarray(vid.get_data(frameIdx))
    fea = pose[frameIdx]
    # Plot video
    if PLOT:
        rearranged = frame[:,:,[2,1,0]].copy()
        bench = collections.defaultdict(list)
        for ppl_idx, fea_ppl in enumerate(fea):    
            target = getfeature(FEATURETYPE, fea_ppl)
            location = []            
            for idx, loc in enumerate(target):
                if idx%3 == 0:
                    x = int(loc)
                    ppl, dis = getminpeople(x, locate)
                elif idx%3 == 1:
                    y = int(loc)
                elif idx%3 == 2:
                    location.append((x, y, ppl, dis))
            ppl_, dis_ = Most_Common(location)        
            if ppl_ in bench.keys():
                if dis_ < bench[ppl_][0]:
                    tmp_ = []

#                    if get_2d_dis(target, 'pose', 'l_hand', 3, 12) < 10 and get_2d_dis(target, 'pose', 'r_hand', 6, 12) < 10:
#                        location.append((ppl_))
                    if pose_position(target, 6)[1] > pose_position(target, 7)[1] and pose_position(target, 3)[1] > pose_position(target, 4)[1]:
                        
#                        if get_2d_dis(target, 'pose', 'l_hand', 2, 12) < 10 and get_2d_dis(target, 'pose', 'r_hand', 5, 12) < 10:
#                            location.append((ppl_))
#                        if get_2d_dis(target, 'r_hand', 'l_hand', 12, 12) < 1 :
#                            location.append((ppl_))
                        if get_down_eight(target, 'pose', 'pose', 'l_hand', 4, 7, 9):
#                            location.append((ppl_))
#                       if get_2d_dis(target, 'r_hand', 'l_hand', 10, 12) < 3 and get_2d_dis(target, 'l_hand', 'r_hand', 10, 12) < 3:
                           location.append((ppl_))
                    tmp_.extend([dis_, location])
                    bench[ppl_] = tmp_
            else:    
#                if get_2d_dis(target, 'pose', 'l_hand', 3, 12) < 10 and get_2d_dis(target, 'pose', 'r_hand', 6, 12) < 10:
#                    location.append((ppl_))
                if pose_position(target, 6)[1] > pose_position(target, 7)[1] and pose_position(target, 3)[1] > pose_position(target, 4)[1]:
                    
#                    if get_2d_dis(target, 'pose', 'l_hand', 2, 12) < 10 and get_2d_dis(target, 'pose', 'r_hand', 5, 12) < 10:
#                        location.append((ppl_))
#                    if get_2d_dis(target, 'r_hand', 'l_hand', 12, 12) < 1:
#                        location.append((ppl_))
                    if get_down_eight(target, 'pose', 'pose','l_hand', 4, 7, 9):
#                        location.append((ppl_))
#                    if get_2d_dis(target, 'r_hand', 'l_hand', 10, 12) < 3 and get_2d_dis(target, 'l_hand', 'r_hand', 10, 12) < 3:
                        location.append((ppl_))
                bench[ppl_].extend([dis_, location]) 
        for man_correct, info in bench.items():
            cv2.putText(rearranged, "{}".format(shoulder[man_correct]),(int(locate[man_correct][0]), 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, getcolor(man_correct), 2)
            for point in info[1]:
                if len(point) == 4:
                    distribution = point[0:2]
                    person = man_correct
#                    cv2.circle(rearranged, distribution, int(1), getcolor(person), 2)
#                    cv2.line(rearranged, (int(locate[person][0]), 0), (int(locate[person][0]), 240), getcolor(person), thickness=2)
                elif len(point) == 1:
                    person = point[0]
                    shoulder[person] += 1

            
        cv2.imshow('Video', rearranged)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    timeStamp += SAMPLERATE   
cv2.destroyAllWindows()