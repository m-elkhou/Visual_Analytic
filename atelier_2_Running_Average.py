# -*- coding: utf-8 -*-
"""
Created on Sat May 25 08:13:10 2019

@author: EL KHOU Mohammed
"""

import cv2
import numpy as np 

#  Lecture d’une vidéo capture avec le camera de laptop
#cap = cv2.VideoCapture(0)

# Lecture d’une vidéo enregistrer sur le disque dur
#cap = cv2.VideoCapture('C:/Users/mhmh2/Downloads/Video/musicDance.mp4')
cap = cv2.VideoCapture('highway.mp4')

ret, current_background = cap.read()
previous_background = current_background

cpp = 0
while cap.isOpened():
    # lire le video image par image
    ret, frame = cap.read()
    if not ret:    
        cap.release()
        cv2.destroyAllWindows()
        break
    
    cpp += 1
    a = float(1.00)/float(cpp)
    if a < 1.00/1000:   # temp d'apprentissage
        a= 1.00/1000
    
    # modify the data type 
    # setting to 32-bit floating point 
    Img = np.float32(frame) 
    p_b = np.float32(previous_background)
    
    # using the cv2.accumulateWeighted() function 
    # that updates the running average 
#    cv2.accumulateWeighted(f, averageValue, 0.02) 
    
    averageValue = (1.00-a) * p_b  + a * Img
    
    # converting the matrix elements to absolute values  
    # and converting the result to 8-bit.  
    current_background = cv2.convertScaleAbs(averageValue) 
    
    frame_diff = cv2.absdiff(frame,current_background)
    
    
    frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)
    cv2.moveWindow('Frame',10,10)
    
    background = cv2.resize(current_background, dsize=None, fx=0.5, fy=0.5)
    cv2.moveWindow('BK Running Average',700,10)
    
    frame_diff = cv2.resize(frame_diff, dsize=None, fx=0.5, fy=0.5)
    cv2.moveWindow('Running Average',10,350)
    
#    affichier l'image dans une frame a chaque instance
    cv2.imshow("Frame", frame)
    cv2.imshow("BK Running Average", background)
    cv2.imshow("Running Average", frame_diff)
    
#   key listener sur le clavier a chaq 20 mili segonde
    key = cv2.waitKey(20) 
    if key == 27: # hadi hiyya Esc
        cap.release()
        cv2.destroyAllWindows()
        break
    
    previous_background = current_background.copy()
    ret, current_background = cap.read()
    
    
    #    img = []
#    for l in range(0,len(frame)):
#        img.append([])
#        img[l]=[]
#        for c in range(0,len(frame[0])):
#            img[l].append([])
#            img[l][c]=[]
#            for cl in range(0,len(frame[0][0])):
#                img[l][c].append(int(abs( (1.00-a)*float(current_background[l][c][cl]) + a * float(frame[l][c][cl]))))
#     current_background =  cv2.convertScaleAbs(img) 
     

#    N=1.0-a
#    pb = [[[i*N for i in x] for x in y] for y in previous_background]
#    fg = [[[i*a for i in x] for x in y] for y in frame]
#     
#    s = abs(pb + fg)
#    p = [[[int(i) for i in x] for x in y] for y in s]
    
#    current_background = abs( float( (1-a) * previous_background ) + float( a * frame_gray ) )