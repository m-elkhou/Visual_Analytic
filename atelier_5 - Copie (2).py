# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 15:38:00 2019

@author: EL KHOU Mohammed


On souhaite etablir un systeme qui permet de faire le suivi de la main
en utilisant les points d'interets SIFT. 
On suppose que le mouvement majoritaire est effectue par la main

. Soustraire l'arriere-plan complexe a l'aide du MOG2 
    RGB
. Detecter la region de la main
    Utilser les operation morphologiques
. Calculer les points d'interts SIFT
. Apparier les points pour predire le chemin de la main

"""

import cv2, imutils

cap = cv2.VideoCapture(0)

MOG2 = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=25, detectShadows=True)

term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

x, y, width, height = 10, 100, 170, 190#300, 305, 300, 300
# Create SIFT Feature Detector object
sift = cv2.xfeatures2d.SIFT_create()
#sift = cv2.SIFT()

keypoints_old , descriptors_old , frame_old= None, None, None
ap = 0
while cap.isOpened():
    try:  
        ret, frame = cap.read()
        if not ret:
            break
        
        # resize the frame
        frame = imutils.resize(frame, height=350)
        #        frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5) 
        
        frame = cv2.flip(frame, 1)

        
        maskMOG2 = MOG2.apply(frame)
        mask = cv2.threshold(maskMOG2, 5, 255, cv2.THRESH_BINARY)[1]
        
        # find the keypoints and descriptors with SIFT
        keypoints, descriptors = sift.detectAndCompute(mask,None)
#        print("Number of keypoints Detected: ", len(keypoints))
        
        #  Draw rich key points on input image
        SiftFrame = cv2.drawKeypoints(frame, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        '''
        OpenCV fournit également la fonction cv2.drawKeyPoints () qui dessine les petits cercles 
        sur les emplacements des points-clés.
        Si vous passez un drapeau, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, 
        il va dessiner un cercle de la taille du point-clé et même montrer son orientation.
        '''
        
        if ap!=0:
            # BFMatcher with default params
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(descriptors_old,descriptors, k=2)

            # Apply ratio test
            good = []
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good.append([m])
            
            img3 = frame
            # cv2.drawMatchesKnn expects list of lists as matches.
            img3 = cv2.drawMatchesKnn(frame_old, keypoints_old, frame, keypoints,good, img3,flags=2)
            
            cv2.imshow('SIFT 2',img3)
    
        ap=1
        keypoints_old , descriptors_old , frame_old = keypoints.copy() , descriptors.copy() , frame.copy()
        
#        _, track_window = cv2.meanShift(mask, (x, y, width, height), term_criteria)
#        x, y, w, h = track_window
#        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#        cv2.putText(frame,'Main',(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3, cv2.LINE_AA)  
        
        _, track_window = cv2.meanShift(mask, (x, y, width, height), term_criteria)
        x, y, width, height= track_window
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(frame,'Main',(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3, cv2.LINE_AA)     
        
    # segment the hand region :
        
        # get the ROI
        roi = mask[ x:(x + width), y:(y + height)]
        
        # get the contours in the thresholded image
#        (_, contours, _) = cv2.findContours(roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        (_, contours, _) = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#        (_, contours, _) = cv2.findContours(roi.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # return None, if no contours detected
        if len(contours) != 0:
            #find contour of max area(hand)
            cnt = max(contours, key = lambda x: cv2.contourArea(x))
            #make convex hull around hand
            hull = cv2.convexHull(cnt)
            
            areacnt = cv2.contourArea(cnt)
              
            # based on contour area, get the maximum contour which is the hand
            segmented = max(contours, key=cv2.contourArea)
            # draw the segmented region and display the frame
#            cv2.drawContours(frame, [cnt + (x, y)], -1, (0, 0, 255))
            cv2.drawContours(frame, [cnt ], -1, (0, 0, 255))
            cv2.imshow("ROI", roi)

        cv2.moveWindow( 'Mean SIFT', 0, 0)
        cv2.moveWindow( 'Mask MOG2', 470, 0)
        cv2.moveWindow( 'SIFT', 470*2, 0)
        cv2.moveWindow( 'ROI', 470*2, 380)
        cv2.moveWindow( 'SIFT 2', 0, 380)
    
        cv2.imshow("Mean SIFT", frame)
        cv2.imshow("Mask MOG2", maskMOG2)
        cv2.imshow("SIFT", SiftFrame)
        
    except Exception as e:
        print(' Ereur !! ',e)
        pass

    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
        cap.release()
        exit(0)
        break
    
cv2.destroyAllWindows()
cap.release()
exit(0)

'''
SIFT no longer available in opencv > 3.4.2

I had the same problem. I change other opencv-python and opencv-contrib-python version, and solve this problem. Here is the history version about opencv-python.

https://pypi.org/project/opencv-python/#history, and i use the following code :

==> pip install opencv-python==3.4.2.16

==> pip install opencv-contrib-python==3.4.2.16

Edit

For Anaconda User just this instead of pip

==> conda install -c menpo opencv

this will install cv2 3.4.1 and everything you need to run SIFT

\to return to final version:
pip install opencv-python==4.1.0.25
pip install opencv-contrib-python==4.1.0.25

'''