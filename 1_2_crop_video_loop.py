# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 20:46:50 2017

@author: dennis60512
"""



import cv2
import os, glob, sys
import subprocess as sp

import dlib
from imutils import face_utils


#os.chdir('D:\\Lab\\Dennis\\Gamania\\Script')

os.chdir('D:\\Lab\\Dennis\\Gamania\\Data')
ROOT = os.getcwd()

#WORKDIR_total = ['.\\2017-04-25-stich\\SmallerSize\\', '.\\2017-04-28-stich\\SmallerSize\\', '.\\2017-05-03-1-stich\\SmallerSize\\', '.\\2017-05-03-2-stich\\SmallerSize\\', '.\\2017-05-04-stich\\SmallerSize\\', '.\\2017-05-24-stich\\SmallerSize\\']
#WORKDIR_total = ['.\\2017-07-21-1-stich\\', '.\\2017-07-21-2-stich\\', '.\\2017-07-20-3-stich\\']
#WORKDIR_total =  [ '.\\2017-07-28-1-stich\\',  '.\\2017-07-28-2-stich\\', '.\\2017-07-28-3-stich\\']
#WORKDIR_total = ['.\\2017-09-12-stich\\',  '.\\2017-09-21-stich\\']
#WORKDIR = '../Data/2017-03-30-stich/'
WORKDIR_total =[ '.\\2017-10-24-stich\\']
MP4IDX = 5 # change??
HEIGHT = 1280
WIDTH = 2560
RESIZE = 0.5
HOP = 5000 # hop 100 ms for face detection
NUM_PEOPLE = 5
RANGE = 150
#%% Find Cropping point by Face Detection
def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

def getColor(idx):
    colorDict = {0:(66,80,244), 1:(65,208,244), 2:(65,244,172), 3:(244,244,65), 4:(244,65,133), 5:(205,65,244)}
    return colorDict[idx]

def facedetect2(img, resize, rng):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    rects = detector(gray, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        color = getColor(i)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        
        # show the face number and cropping line
        cv2.putText(img, "Face #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=1)
        cv2.line(img, (int(x-rng*resize), 0), (int(x-rng*resize), int(1280*resize)), color, thickness=2)
        cv2.line(img, (int(x+w+rng*resize), 0), (int(x+w+rng*resize), int(1280*resize)), color, thickness=2)
        cv2.putText(img, str(int(x/resize-rng)), (int(x-rng*resize), int(100*resize)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2)
        cv2.putText(img, str(int((x+w)/resize+rng)), (int(x+w+rng*resize), int(640*resize)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2)
    return img
#%%
for WORKDIR in WORKDIR_total:
    #%% find mp4 file lists
    mp4Files = []
    PATH = os.path.join(ROOT, WORKDIR) 
    os.chdir(PATH)
    for mp4 in glob.glob("*.MP4"):
        print(mp4)  
        mp4Files.append(os.path.abspath(mp4))
    mp4Files = sorted(mp4Files)
    os.chdir(ROOT)
    
    #%% Save ifg for  Cropping info
#    if WORKDIR == '.\\2017-07-19-1-stich\\':
#        CropList = [(592,980,'C'), (1556,1952, 'A'), (1952,2280, 'B')]
#    elif WORKDIR ==  '.\\2017-07-19-2-stich\\':
#        CropList = [(244,630,'C'),  (582,984, 'D'), (1538,1940, 'A'), (1880, 2266, 'B')]
#    elif WORKDIR =='.\\2017-07-19-3-stich\\':
#        CropList = [(516,902,'C'), (1548,1952, 'A'), (1972,2396, 'B')]
#    if WORKDIR == '.\\2017-07-20-1-stich\\':
#        CropList = [(228,600,'A'), (596,996, 'B'), (1564,1936, 'C'), (1900,2266,'D')]
#    elif WORKDIR ==  '.\\2017-07-20-2-stich\\':
#        CropList = [(120,544,'A'),  (638,1000, 'B'), (868,1240, 'C'), (1276, 1640, 'D'), (1640, 1912, 'E')]
#    if WORKDIR == '.\\2017-07-21-1-stich\\':
#        CropList = [(212,628,'A'), (616,1020, 'B'), (1514,1956, 'C'), (1868,2244,'D')]
#    elif WORKDIR ==  '.\\2017-07-21-2-stich\\':
#        CropList = [(236,650,'A'),  (582,1000, 'B'), (1584,1988, 'C')]
#    elif WORKDIR =='.\\2017-07-20-3-stich\\':
#        CropList = [(304,708,'A'), (584,970, 'B'), (1614,1964, 'C'), (1952,2328, 'D')]
#    if WORKDIR == '.\\2017-07-21-3-stich\\':
#        CropList = [(236,640,'A'), (584,970, 'B'), (1514,1872, 'C'), (1850,2240,'D')]
#    elif WORKDIR ==  '.\\2017-07-24-1-stich\\':
#        CropList = [(200,600,'A'),  (600,1032, 'B'), (1578,1952, 'C'), (1918, 2322, 'D')]
#    if WORKDIR == '.\\2017-07-24-2-stich\\':
#        CropList = [(228,614,'A'), (604,1008, 'B'), (1630,2100, 'C')]
#    elif WORKDIR ==  '.\\2017-07-24-3-stich\\':
#        CropList = [(204,584,'A'),  (490,892, 'B'), (1602,1988, 'C'), (1940, 2314, 'D')]
#    elif WORKDIR =='.\\2017-07-25-1-stich\\':
#        CropList = [(0,288,'A'), (294,650, 'B'), (638,1036, 'C'), (1710,2128, 'D'), (2128,2500, 'E')]
#    elif WORKDIR == '.\\2017-07-25-2-stich\\':
#        CropList =  [(238,568,'A'), (604,1008, 'B'), (926,1280, 'C'), (1388,1798, 'D'), (1754,2160, 'E')]
#    elif WORKDIR == '.\\2017-07-26-stich\\':
#        CropList = [(212,616,'A'), (592,962, 'B'), (1562,1950, 'C'), (1890,2314,'D')] 
#    if WORKDIR ==  '.\\2017-07-27-2-stich\\':
#        CropList = [(266,654,'A'), (1560,1960, 'B'), (1898,2286, 'C')]
#    elif WORKDIR ==   '.\\2017-07-27-3-stich\\':
#        CropList = [(180,552,'A'),  (558,962, 'B'), (1514,1918, 'C'), (1940, 2314, 'D')]
#    if WORKDIR ==  '.\\2017-07-28-1-stich\\':
#        CropList = [(238,624,'A'), (1572,1912, 'B'), (1928,2332, 'C')]
#    elif WORKDIR ==   '.\\2017-07-28-2-stich\\':
#        CropList = [(220,592,'A'),  (592,980, 'B'), (984,1338, 'C'), (1352, 1756, 'D'), (1754, 2140, 'E')]
#    elif WORKDIR =='.\\2017-07-28-3-stich\\':
#        CropList = [(252,624,'A'), (616,1020, 'B'), (1688,2090, 'C')]
#    if WORKDIR ==  '.\\2017-09-12-stich\\':
#        CropList = [(600,1008,'A'), (1466,1872, 'B'), (1820,2194, 'C')]
#    elif WORKDIR ==   '.\\2017-09-21-stich\\':
#        CropList = [(212,628,'A'),  (628,1030, 'B'), (1480,1888, 'C'), (1870, 2256, 'D')]
#    elif WORKDIR =='.\\2017-10-03-stich\\':
#        CropList = [(108,558,'A'), (500,900, 'B'), (834,1250, 'C'), (1360, 1760, 'D'), (1682, 2160, 'E')]
#    if WORKDIR ==  '.\\2017-08-31-stich\\':
#        CropList = [(260,632,'A'), (536,932, 'B'), (1632,2026, 'C'), (2004, 2372, 'D')]
#    if WORKDIR == '.\\2017-10-13-stich\\':
#        CropList = [(300,724,'A'), (1502,1906, 'B'), (1870,2256, 'C')]
#    elif WORKDIR ==   '.\\2017-10-17-stich\\':
#        CropList = [(228,600,'A'),  (632,1018, 'B'), (1496,1852, 'C'), (1856, 2230, 'D')]
#    if WORKDIR == '.\\2017-08-03-stich\\':
#        CropList = [(484,856,'A'), (858,1240, 'B'), (1168,1572, 'C'), (1658, 2044, 'D'),  (1956, 2342, 'E')]
    if WORKDIR == '.\\2017-10-24-stich\\':
        CropList = [(170,558,'A'), (570,974, 'B'), (1558,1982, 'C'),   (1956, 2342, 'E')]
    #%% Cropping videos 
    def subprocess_cmd(command):
        process = sp.Popen(command,stdout=sp.PIPE, shell=True)
        proc_stdout = process.communicate()[0].strip()
        print (proc_stdout)
        
    def cropVideo(personInfo, mp4Files):
        processes = []
#        start = int(personInfo[0]*1000/2560)
#        end = int(personInfo[1]*1000/2560)
        start = personInfo[0]
        end = personInfo[1]
        personId = personInfo[2]
        for mp4File in mp4Files:
            outDir = os.path.join(WORKDIR.split('\\')[1], WORKDIR.split('\\')[1]+'-crop', personId)
            filName = os.path.basename(mp4File).split('.')[0]
            if not os.path.exists(outDir):
                os.makedirs(outDir)
    
            outPath = os.path.join(outDir,filName+'_'+str(personId)+'.MP4')
            args = 'ffmpeg -i '+mp4File+' -filter:v crop=' +str(end-start)+':1280:'+str(start)+':0 '+outPath
            processes.append(args)
        processes = ' & '.join(processes)
        subprocess_cmd(processes)
        print('SLICING '+str(personId)+' DONE !!!')
        
    
    [cropVideo(personInfo, mp4Files) for personInfo in CropList]
    
    print('One Done !!!\n')
print('All Done !!!\n')
