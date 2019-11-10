# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:41:05 2019

@author: EL KHOU Mohammed

ATELIER 4:
    1) Extraire 2 frames a l'instant t et t+1
        Calculer le flot optique a base de ces 2 frames
        
    2) Calculer le fot optique en temps reel en se basant 
        sur des frames prises au bouts des laps du temps
Source :
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
"""

import cv2
import numpy as np

#  Lecture d’une vidéo capture avec le camera de laptop
#cap = cv2.VideoCapture(0)

# Lecture d’une vidéo enregistrer sur le disque dur
cap = cv2.VideoCapture('highway.mp4')

_,img1 = cap.read()
_,img2 = cap.read()

img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
img1_gray = cv2.GaussianBlur(img1_gray,(5,5),0)
img2_gray = cv2.GaussianBlur(img2_gray,(5,5),0)

# Create some random colors
color = np.random.randint(0,255,(100,3))

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Take first frame and find corners in it
p0 = cv2.goodFeaturesToTrack(img1_gray, mask = None, **feature_params)

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# calculate optical flow
p1, st, err = cv2.calcOpticalFlowPyrLK(img1_gray, img2_gray, p0, None, **lk_params)

# Select good points
good_new = p1[st==1]
good_old = p0[st==1]

# Create a mask image for drawing purposes
mask = np.zeros_like(img1)

# draw the tracks
for i,(new,old) in enumerate(zip(good_new,good_old)):
    a,b = new.ravel()
    c,d = old.ravel()
    mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    img2 = cv2.circle(img2,(a,b),5,color[i].tolist(),-1)
img = cv2.add(img2,mask)
    
cv2.imshow('frame',img)
while(1):
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        cv2.destroyAllWindows()
        cap.release()
        break
