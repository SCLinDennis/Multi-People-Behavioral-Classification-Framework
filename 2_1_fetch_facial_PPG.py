# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 19:59:37 2017

@author: HGY
"""

import os, glob, sys
import subprocess as sp
import multiprocessing  as mp
import pandas as pd
import collections
import joblib as ib
from collections import defaultdict

#%%
os.chdir('E:\\Gamania\\Data\\')
WORKDIR = '.\\2018-03-16-stich\\'
ROOT = os.getcwd()
FEAEX_PATH = 'E:\\Gamania\\Script\\CML-Face\\FeatureExtraction.exe' # path to feature extraction program

#%% Functions
def subprocess_cmd(cmd):
    p = sp.Popen(cmd, stdout=sp.PIPE)
    # Grab stdout line by line as it becomes available.  This will loop until 
    # p terminates.
    while p.poll() is None:
        print(p.stdout.readline()) # This blocks until it receives a newline.
    # When the subprocess terminates there might be unconsumed output 
    # that still needs to be processed.
    # print p.stdout.read()

def subprocess_cmd2(command):
    process = sp.Popen(command,stdout=sp.PIPE, shell=True)
    proc_stdout = process.communicate()[0].strip()
    print (proc_stdout)
     
    
#%% find mp4 file lists
#------------    -------------
PATH = os.path.join(ROOT, WORKDIR, WORKDIR.split('\\')[1]+'-crop') 
cropList = next(os.walk(PATH))[1]
if cropList:
    for personId in cropList:
        if personId != 'H':
            mp4Files = []
            cropDir = os.path.join(PATH, personId)
            
            #-- Get Cropped mp4files
            for mp4 in glob.glob(cropDir+'/*.MP4'):
                print(mp4)  
                mp4Files.append(os.path.abspath(mp4))
            mp4Files = sorted(mp4Files)
            
 
            #-- Perform Feature Extraction for each person
            # Matlab CML
            inDir = cropDir+'\\'
            outDir = os.path.join(cropDir,'facial-landmarks')+'\\'
            args = []
            for mp4 in mp4Files:
                arg = FEAEX_PATH
                arg = arg + ' -rigid'
                arg = arg + ' -f '+ mp4
                arg = arg + ' -op ' + os.path.join(outDir, os.path.basename(mp4).split('.')[0]+'_pose.txt')
                arg = arg + ' -ogaze ' + os.path.join(outDir, os.path.basename(mp4).split('.')[0]+'_gaze.txt')
                arg = arg + ' -oparams ' + os.path.join(outDir, os.path.basename(mp4).split('.')[0]+'_params.txt')
                
                args.append(arg)
            args = ' & '.join(args)
            subprocess_cmd(args)
     
else:
    sys.exit('Empty Crop List! Something Goes Wrong!! \n')



