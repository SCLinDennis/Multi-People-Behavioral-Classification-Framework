# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 13:52:04 2017

@author: dennis60512
"""


import os, glob
import subprocess as sp
import cv2


ROOT = os.getcwd()
WORKDIR_pre = 'G:\\Lab\\Gamania\\Train_Data\\'
OUT_WIDTH = 480
OUT_HIGHT = 240
SCALE = 5
WORKDIR_total = ['.\\2017-06-28-stich\\'] 

def subprocess_cmd(command):
    process = sp.Popen(command,stdout=sp.PIPE, shell=True)
    proc_stdout = process.communicate()[0].strip()
    print (proc_stdout)
#%%resize original video

for WORKDIR in WORKDIR_total:
#for WORKDIR in glob.glob(WORKDIR_pre+'*-stich'):
    if WORKDIR.split('\\')[-1] != '2017-06-28-stich':
        print('Start processing ' + WORKDIR.split('\\')[-1])
        PATH = os.path.join(WORKDIR_pre, WORKDIR) 
        os.chdir(PATH)
        mp4Files = []
        for mp4 in glob.glob("*.MP4"):
            print(mp4)  
            mp4Files.append(os.path.abspath(mp4))
        mp4Files = sorted(mp4Files)
        outPath = os.path.join( '240_480')+'\\'
        if not os.path.exists(outPath):
            os.makedirs(outPath)
        processes = []
        for mp4 in mp4Files:
        #    vidcap = cv2.VideoCapture(mp4)
        #    OUT_WIDTH = int(vidcap.get(3)/SCALE)
            
            fileName = os.path.basename(mp4).split('.')[0]
            arg = 'ffmpeg -i '+mp4+' -vf scale='+str(OUT_WIDTH)+':'+str(OUT_HIGHT) +' '+ outPath + mp4.split('\\')[-1][:-4] + '.MP4'
            processes.append(arg)
        processes = '&'.join(processes)
        subprocess_cmd(processes)
        print('covert '+WORKDIR+' DONE !!!')
