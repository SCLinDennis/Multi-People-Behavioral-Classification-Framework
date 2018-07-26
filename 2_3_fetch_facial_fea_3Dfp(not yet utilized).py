# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 19:59:37 2017

@author: HGY
"""

import os, glob, sys
import subprocess as sp
#%%
os.chdir('C:\\Lab\\Dennis\\Gamania\\Data\\')
#TOTAL = [ '.\\2017-07-27-2-stich\\', '.\\2017-07-27-3-stich\\',  '.\\2017-07-28-1-stich\\', '.\\2017-07-28-2-stich\\',  '.\\2017-07-28-3-stich\\', '.\\2017-07-21-1-stich\\', '.\\2017-07-21-2-stich\\',  '.\\2017-07-21-3-stich\\', '.\\2017-07-24-1-stich\\',  '.\\2017-07-24-2-stich\\']
#TOTAL = [ '.\\2017-07-21-1-stich\\', '.\\2017-07-21-2-stich\\',  '.\\2017-07-21-3-stich\\', '.\\2017-07-24-1-stich\\',  '.\\2017-07-24-2-stich\\']

#WORKDIR = '.\\2017-07-11-1-stich\\'
ROOT = os.getcwd()
FEAEX_PATH = 'D:\\Lab\\Dennis\\Gamania\\Script\\OpenFace_0.2.3_win_x64\\OpenFace_0.2.3_win_x64\\FeatureExtraction.exe' # path to feature extraction program

WORKDIR_total1 = ['2017-05-24-stich',
                 '2017-06-26-1-stich',
                 '2017-06-26-2-stich',
                 '2017-06-27-stich',
                 '2017-06-28-stich',
                 '2017-06-30-stich']

WORKDIR_total2 = ['2017-07-03-1-stich',
                 '2017-07-03-2-stich',
                 '2017-07-05-stich',
                 '2017-07-06-1-stich',
                 '2017-07-06-2-stich',
                 '2017-07-07-stich',
                 '2017-07-11-1-stich',
                 '2017-07-11-2-stich']

WORKDIR_total3 = ['2017-07-12-1-stich',
                 '2017-07-12-2-stich',
                 '2017-07-12-3-stich',
                 '2017-07-13-1-stich',
                 '2017-07-13-2-stich',
                 '2017-07-14-1-stich',
                 '2017-07-14-2-stich',
                 '2017-07-18-stich',
                 '2017-07-19-1-stich',
                 '2017-07-19-2-stich',#here
                 '2017-07-19-3-stich']

WORKDIR_total4 =  [
                 '2017-07-25-1-stich',
                 '2017-07-25-2-stich',
                 '2017-07-26-stich',
                 '2017-07-20-1-stich',#here
                 '2017-07-20-3-stich']
                 

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
for WORKDIR in WORKDIR_total4:
    PATH = os.path.join(ROOT, WORKDIR, WORKDIR+'-crop') 
    cropList = next(os.walk(PATH))[1]
    if cropList:
        for personId in cropList:
            if personId != 'H' and personId[-6:] != '-stich' and personId[-6:] != '-ratio':
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
                    arg = arg + ' -f '+ mp4
                    arg = arg + ' -of ' + os.path.join(outDir, os.path.basename(mp4).split('.')[0]+'_3Dfp.txt')
                    arg = arg + ' -no2Dfp '
#                    arg = arg + ' -no3Dfp '
                    arg = arg + ' -noMparams '
                    arg = arg + ' -noPose '
                    arg = arg + ' -noAUs '
                    arg = arg + ' -noGaze '                    
                    args.append(arg)
                args = ' & '.join(args)
                subprocess_cmd(args)
         
    else:
        sys.exit('Empty Crop List! Something Went Wrong!! \n')



