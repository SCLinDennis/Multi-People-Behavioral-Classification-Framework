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
#WORKDIR_total = ['.\\2017-07-07-stich\\', '.\\2017-07-11-1-stich\\', '.\\2017-07-11-2-stich\\', '.\\2017-07-12-1-stich\\', '.\\2017-07-12-2-stich\\', '.\\2017-07-12-3-stich\\', '.\\2017-07-13-1-stich\\', '.\\2017-07-13-2-stich\\', '.\\2017-07-14-1-stich\\',  '.\\2017-07-14-2-stich\\', '.\\2017-07-18-stich\\', '.\\2017-07-19-1-stich\\', '.\\2017-07-19-2-stich\\', '.\\2017-07-19-3-stich\\', '.\\2017-07-20-1-stich\\']
#WORKDIR_total = ['.\\2017-07-20-3-stich\\', '.\\2017-07-21-1-stich\\', '.\\2017-07-21-2-stich\\', '.\\2017-07-21-3-stich\\', '.\\2017-07-24-1-stich\\','.\\2017-07-24-2-stich\\', '.\\2017-07-24-3-stich\\',  '.\\2017-07-25-1-stich\\', '.\\2017-07-25-2-stich\\', '.\\2017-07-26-stich\\',  '.\\2017-07-27-2-stich\\', '.\\2017-07-27-3-stich\\', '.\\2017-07-28-1-stich\\', '.\\2017-07-28-2-stich\\', '.\\2017-07-28-3-stich\\']
WORKDIR_total = ['2017-05-24-stich',
                 '2017-06-09-stich',
                 '2017-06-13-stich',
                 '2017-06-20-1-stich',
                 '2017-06-20-2-stich',
                 '2017-06-21-stich',
                 '2017-06-26-1-stich',
                 '2017-06-26-2-stich',
                 '2017-06-27-stich',
                 '2017-06-28-stich',
                 '2017-06-30-stich']
WORKDIR_total = ['2017-07-03-1-stich',
                 '2017-07-03-2-stich',
                 '2017-07-05-stich',
                 '2017-07-06-1-stich',
                 '2017-07-06-2-stich',
                 '2017-07-07-stich',
                 '2017-07-11-1-stich',
                 '2017-07-11-2-stich']

WORKDIR_total = ['2017-07-12-1-stich',
                 '2017-07-12-2-stich',
                 '2017-07-12-3-stich',
                 '2017-07-13-1-stich',
                 '2017-07-13-2-stich',
                 '2017-07-14-1-stich',
                 '2017-07-14-2-stich',
                 '2017-07-18-stich',
                 '2017-07-19-1-stich',
                 '2017-07-19-2-stich',
                 '2017-07-19-3-stich']

WORKDIR_total =  ['2017-07-21-3-stich',
                 '2017-07-24-1-stich',
                 '2017-07-24-2-stich',
                 '2017-07-24-3-stich',
                 '2017-07-25-1-stich',
                 '2017-07-25-2-stich',
                 '2017-07-26-stich',
                 '2017-07-27-2-stich',
                 '2017-07-27-3-stich',
                 '2017-07-28-1-stich',
                 '2017-07-28-2-stich',
                 '2017-07-28-3-stich']

def subprocess_cmd(command):
    process = sp.Popen(command,stdout=sp.PIPE, shell=True)
    proc_stdout = process.communicate()[0].strip()
    print (proc_stdout)
#WORKDIR = '.\\2017-04-25-stich\\'
#%%
'''
for WORKDIR in WORKDIR_total:
    
                                       
    #find mp4 file lists
    mp4Files = []
    PATH = os.path.join(ROOT, WORKDIR_pre, WORKDIR, WORKDIR.split('\\')[1]+'-crop') 
    os.chdir(PATH)
    cropList = next(os.walk(PATH))[1]
#    os.chdir(ROOT)
    if cropList:
        for personId in cropList:
            if personId != 'H' and personId[-5:] != 'stich' and personId[-3:] != 'XXX':
                mp4Files = []
                cropDir = os.path.join(PATH, personId)
                
                #-- Get Cropped mp4files
                for mp4 in glob.glob(cropDir+'/*.MP4'):
                    print(mp4)  
                    mp4Files.append(os.path.abspath(mp4))
                mp4Files = sorted(mp4Files)
    
                
                outPath = os.path.join(PATH ,  WORKDIR.replace('-stich', '-ratio'), personId)+'\\'
                if not os.path.exists(outPath):
                    os.makedirs(outPath)
                processes = []
                for mp4 in mp4Files:
                    vidcap = cv2.VideoCapture(mp4)
                    OUT_WIDTH = int(vidcap.get(3)/SCALE)
                    if OUT_WIDTH %2 ==1:
                        OUT_WIDTH -= 1 
                    print(OUT_WIDTH)
                    fileName = os.path.basename(mp4).split('.')[0]
                    arg = 'ffmpeg -i '+mp4+' -vf scale='+str(OUT_WIDTH)+':'+str(OUT_HIGHT) +' '+ outPath + mp4.split('\\')[-1][:-4] + '.MP4'\
#                    arg = 'ffmpeg -i '+mp4+' -vf scale=' + str(OUT_WIDTH)+':ih*.5 '+ outPath + mp4.split('\\')[-1][:-4] + '.MP4'

                    processes.append(arg)
                processes = '&'.join(processes)
                subprocess_cmd(processes)
                print('covert '+WORKDIR+' DONE !!!')
'''
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
