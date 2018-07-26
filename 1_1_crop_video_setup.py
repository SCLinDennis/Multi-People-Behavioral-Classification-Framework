# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 14:15:43 2017

@author: HGY
"""

import cv2
import os, glob, sys
import subprocess as sp

import dlib
from imutils import face_utils

os.chdir('E:\\Gamania\\Data')
#D:\Lab\Dennis\Gamania\Data\2017-09-12-stich
ROOT = os.getcwd()
WORKDIR_total = [ '.\\2018-03-20-stich\\', '.\\2018-03-26-stich\\', '.\\2018-03-27-stich\\']
WORKDIR =  '.\\2018-03-27-stich\\'
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

#%% find mp4 file lists
mp4Files = []
PATH = os.path.join(ROOT, WORKDIR) 
os.chdir(PATH)
for mp4 in glob.glob("*.MP4"):
    print(mp4)  
    mp4Files.append(os.path.abspath(mp4))
mp4Files = sorted(mp4Files)
os.chdir(ROOT)


#%%
video_path = mp4Files[MP4IDX]
vidcap = cv2.VideoCapture(video_path)
for timeStamp in range(200):
    time = timeStamp*HOP
    vidcap.set(cv2.CAP_PROP_POS_MSEC, time)      # just cue to 20 sec. position
    success,frame = vidcap.read()
    if success:
        print('Success!', time)
        imgResize = cv2.resize(frame, (int(WIDTH*RESIZE), int(HEIGHT*RESIZE)))
        cv2.imshow('Face Detection', facedetect2(imgResize, RESIZE, RANGE))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            vidcap.release()     
            cv2.destroyAllWindows() 
            break
        elif cv2.waitKey(1) & 0xFF == ord(' '):
            os.system("pause")
            
    else:
        sys.exit('Failed to load video! Plz check the paths!')


#%% Find Cropping point by Hand
'''
Start = 2300
End = 2550
video_path = mp4Files[0]
vidcap = cv2.VideoCapture(video_path)
vidcap.set(cv2.CAP_PROP_POS_MSEC,20000)      # just cue to 20 sec. position
success,img = vidcap.read()
imgCrop = img
if success:
    cv2.line(img, (Start, 0), (Start, 1280), (0,0,255))
    cv2.line(img, (End, 0), (End, 1280), (0,0,255))
    cv2.putText(img, str(Start), (Start-80,640), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), thickness=3)
    cv2.putText(img, str(End), (End-80,640), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), thickness=3)
    imResize = cv2.resize(img, (1800,900))
    cv2.imshow('img', imResize)
    cv2.waitKey()
    cv2.destroyAllWindows()
'''
#%% Save ifg for  Cropping info
if WORKDIR == '.\\2018-03-20-stich\\':
    CropList = [(600,930,'CH01'), (1000,1250, 'CH02'), (1250,1500, 'CH03'), (1600, 1900, 'CH04'), (2100, 2350, 'CH06')]
elif WORKDIR ==   '.\\2018-03-26-stich\\':
    CropList = [(700,1100,'CH01'),  (1100,1370, 'CH02'), (1370,1590, 'CH03'), (1590, 1810, 'CH04'), (1900, 2150, 'CH05'), (2200, 2450, 'CH06')]
elif WORKDIR =='.\\2018-03-27-stich\\':
    CropList = [(950,1250,'CH01'), (1330,1550, 'CH02'), (1600,1800, 'CH03'), (2050, 2300, 'CH04'), (2300, 2550, 'CH06')]
elif WORKDIR == '.\\2018-02-12-stich\\':
    CropList =  [(350,635,'CH01'), (635,900, 'CH02'), (1000,1300, 'CH03'), (1350,1600, 'CH04'), (1600,1880, 'CH05'), (1880, 2250, 'CH06')]
elif WORKDIR == '.\\2018-02-13-stich\\':
    CropList = [(400,700,'CH01'), (700,1000, 'CH02'), (1050,1300, 'CH03'), (1300,1550,'CH04'), (1650, 1930, 'CH05'), (1930, 2300, 'CH06')] 
elif WORKDIR ==  '.\\2018-02-26-stich\\':
    CropList = [(500,850,'CH01'), (900,1200, 'CH02'), (1250,1550, 'CH03'), (1600,2000, 'CH04'), (2000, 2300, 'CH06')] 
elif WORKDIR ==  '.\\2018-02-27-stich\\':
    CropList = [(400,800,'CH01'), (900,1200, 'CH02'), (1250,1550, 'CH03'), (1730,2150, 'CH04'), (2100, 2300, 'CH06')] 
#'.\\2017-07-14-2-stich\\', '.\\2017-07-17-stich\\', '.\\2017-07-18-stich\\', '.\\2017-07-20-1-stich\\'
imgCrop = frame
for idx, (start,end,person) in enumerate(CropList):
    print(idx,start,end,person)
    start = int(start)
    end = int(end)
    color = getColor(idx)
    cv2.line(imgCrop, (start, 0), (start, 1280), color, thickness=3)
    cv2.line(imgCrop, (end, 0), (end, 1280), color, thickness=3)
    cv2.putText(imgCrop, str(start), (start,100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, thickness=3)
    cv2.putText(imgCrop, str(end), (end,640), cv2.FONT_HERSHEY_SIMPLEX, 2, color, thickness=3)   
#cv2.imshow('Crop', imgCrop)
cv2.imshow('Crop', cv2.resize(imgCrop, (int(WIDTH*RESIZE), int(HEIGHT*RESIZE))))
cv2.waitKey()
cv2.destroyAllWindows()
outDir = os.path.join(WORKDIR, WORKDIR.split('\\')[1]+'-crop')
cv2.imwrite(WORKDIR+'\\Crop.jpg', imgCrop)     # save frame as JPEG file
           
#%% Cropping videos 
def subprocess_cmd(command):
    process = sp.Popen(command,stdout=sp.PIPE, shell=True)
    proc_stdout = process.communicate()[0].strip()
    print (proc_stdout)
    
def cropVideo(personInfo, mp4Files):
    processes = []
    start = int(personInfo[0])
    end = int(personInfo[1])
    personId = personInfo[2]
    for mp4File in mp4Files:
        outDir = os.path.join(WORKDIR, WORKDIR.split('\\')[1]+'-crop', personId)
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

print('All Done !!!\n')

