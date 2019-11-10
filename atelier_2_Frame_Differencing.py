# -*- coding: utf-8 -*-
"""
Created on Fri May 24 17:35:36 2019

@author: EL KHOU Mohammed
"""

# Defference d'images _ Frame differencing
import cv2

#  Lecture d’une vidéo capture avec le camera de laptop
cap = cv2.VideoCapture(0)

# Lecture d’une vidéo enregistrer sur le disque dur
#cap = cv2.VideoCapture('highway.mp4')

ret, current_frame = cap.read()

current_frame = cv2.flip(current_frame, 1)
previous_frame = current_frame

while(cap.isOpened()):
    
    ret, current_frame = cap.read()
    if not ret:    
        break

    current_frame = cv2.flip(current_frame, 1)

    # current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    # previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

    # frame_diff = cv2.absdiff(current_frame_gray,previous_frame_gray)
    # frame_diff = abs(current_frame_gray-previous_frame_gray)
    
    frame_diff = cv2.absdiff(current_frame, previous_frame)
#    frame_diff = abs(current_frame - previous_frame)

    frame_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
    frame_diff = cv2.GaussianBlur(frame_diff,(5,5),0)

    _,frame_diff = cv2.threshold(frame_diff,20,255,cv2.THRESH_BINARY) # faire le seuiage globale

    cv2.imshow('frame diff ',frame_diff)    
    
# tab 'ESC'
    key = cv2.waitKey(20) 
    if key == 27: # hadi hiyya Esc
        break
    
    previous_frame = current_frame.copy()
cap.release()
cv2.destroyAllWindows()
