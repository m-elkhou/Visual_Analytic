# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:48:19 2019

@author: Mohammed EL KHOU
"""

import cv2
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('highway.mp4')
#cap = cv2.VideoCapture("D:/Music/Yuri Boyka - Can't Be Touched.mp4")
MOG2 = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=25, detectShadows=True)
#MOG2 = cv2.createBackgroundSubtractorMOG2()

MOG = cv2.bgsegm.createBackgroundSubtractorMOG()

KNN	=	cv2.createBackgroundSubtractorKNN(history=20, dist2Threshold=10, detectShadows=True)

nb_fr = 30   # number of frames used to initialize the background models
de_th = 0.9 # Threshold value, above which it is marked foreground, else background.
GMG = cv2.bgsegm.createBackgroundSubtractorGMG(nb_fr,de_th)

while cap.isOpened():
    # lire le video image par image 
    ret, frame = cap.read()
    if not ret:    # test si le video est terminer
        cap.release()
        cv2.destroyAllWindows()
        break
    
    frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5) 
    
    maskMOG2 = MOG2.apply(frame)
    maskMOG = MOG.apply(frame)
    maskKNN = KNN.apply(frame)
    maskGMG = GMG.apply(frame)
    
    cv2.moveWindow( 'The Original  Frame', 10, 10)
    cv2.moveWindow( 'Mask MOG2', 700, 10)
    cv2.moveWindow( 'Mask MOG', 700, 350)
    cv2.moveWindow( 'Mask GMG', 10, 350)
    cv2.moveWindow( 'Mask KNN', 10, 350)
    
    cv2.imshow("The Original Frame", frame)
    cv2.imshow("Mask MOG2", maskMOG2)
    cv2.imshow("Mask KNN", maskKNN)
    cv2.imshow("Mask GMG", maskGMG)
    cv2.imshow("Mask MOG", maskMOG)
    

    key = cv2.waitKey(20) 
    if key == 27: # hadi hiyya Esc
        cap.release()
        cv2.destroyAllWindows()
        break


# =============================================================================
# if bgsegm.createBackgroundSubtractorMOG() not work
# and u get AttributeError: module 'cv2.cv2' has no attribute 'bgsegm'
# first You need to uninstall opencv before installing opencv-contrib
# Make sure no console is running that has imported cv2 while you execute your installing process
# Run the cmd ( or the Anaconda Prompt ) as Administrateur
# pip uninstall opencv-python
# pip uninstall opencv-contrib-python

# then installing opencv-contrib
# pip install opencv-python
# pip install opencv-contrib-python
# =============================================================================