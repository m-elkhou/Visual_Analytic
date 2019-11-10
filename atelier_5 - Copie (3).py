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
import numpy as np

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('highway.mp4')

MOG2 = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=25, detectShadows=True)

term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

x, y, width, height = 10, 100, 170, 190#300, 305, 300, 300
# Create SIFT Feature Detector object
sift = cv2.xfeatures2d.SIFT_create()
#sift = cv2.SIFT()


ap = 0
MIN_MATCH_COUNT= 30

FLANN_INDEX_KDITREE=0
flannParam = dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
flann = cv2.FlannBasedMatcher(flannParam,{})

trainFrame = cv2.imread("hand.png",0)
trainFrame = imutils.resize(trainFrame, height=350)
trainKP,trainDesc=sift.detectAndCompute(trainFrame,None)
#trainKP , trainDesc , trainFrame = None, None, None

while cap.isOpened():
    try:  
        ret, frame = cap.read()
        if not ret:
            break
        
    # resize the frame
        frame = imutils.resize(frame, height=350)
        #        frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5) 
    # inverce l'image
        frame = cv2.flip(frame, 1)

        
        maskMOG2 = MOG2.apply(frame)
    # le seuiage
#        mask = maskMOG2.copy()
        mask = cv2.threshold(maskMOG2, 25, 255, cv2.THRESH_BINARY)[1]
        
        kernel = np.ones((3,3),np.uint8)
    #extrapolate the hand to fill dark spots within
        mask = cv2.dilate(mask,kernel,iterations = 4)
        
    # blur the image to remove the noise for the contorus detection, 
    # otherwise we would many false contours, or not so clean boundaries.
        '''
        pour éliminer le bruit pour la détection des contours, 
        sinon nous aurions beaucoup de faux contours, ou des limites moins nettes.
    
        '''
        mask = cv2.GaussianBlur(mask,(5,5),100) 
#        mask = cv2.GaussianBlur(frame, (5, 5), 0)
        
    # Mean SIFT
#        _, track_window = cv2.meanShift(mask, (x, y, width, height), term_criteria)
#        x, y, w, h = track_window
#        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#        cv2.putText(frame,'Main',(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3, cv2.LINE_AA)  
        
        frameD = frame.copy()
        _, track_window = cv2.meanShift(mask, (x, y, width, height), term_criteria)
        x, y, width, height= track_window
        cv2.rectangle(frameD, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(frameD,'Main',(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3, cv2.LINE_AA)     
        
# segment the hand region :
        
    # get the ROI
        roi = mask[ x:(x + width), y:(y + height)]
        
    # get the contours in the thresholded image
    # finding the approximate contours of all closed objects in image
        (_, contours, _) = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#        (_, contours, _) = cv2.findContours(roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#        (_, contours, _) = cv2.findContours(roi.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
   
    # return None, if no contours detected
        if len(contours) != 0:
    # Find contour of max area(hand) :
    # Finding the contour with maximum size. (hand when kept considerably closer to webcam in comparison to face.
    # based on contour area, get the maximum contour which is the hand
            
#            Max = 0                                                 #
#            ci = 0                                                  #      
#            for i in range(len(contours)):                          #
#                cnt = contours[i]                                   #     
#                area = cv2.contourArea(cnt)                         #
#                if area>Max:                                        # 
#                    Max = area                                      #
#                    ci = i                                          #
#            cnt = contours[ci]                                      #
            
            cnt = max(contours, key=cv2.contourArea) 
#            cnt = max(contours, key = lambda x: cv2.contourArea(x))     
            
        # draw the segmented region and display the frame
        # storing the hull points and contours in "frame" image variable(matrix).
#            cv2.drawContours(frame, [cnt + (x, y)], -1, (0, 0, 255))
            cv2.drawContours(frameD,[cnt],0,(255,0,0),2)            
            
            # Finding the convex hull of largest contour 
            hull = cv2.convexHull(cnt,returnPoints=True)    
                       
            cv2.drawContours(frameD,[hull],0,(0, 0, 255),2) # 5dar :  (0,255,0)
            
        # we approximate the contours to remove the noise.
#            approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
#            cv2.drawContours(frameD, [approx], 0, (0), 3)
           
        '''
        Scale Invariant Feature Transform (SIFT)
        keypoints : sont les points d’intérêt dans une image
        
        OpenCV fournit également la fonction cv2.drawKeyPoints () qui dessine les petits cercles 
        sur les emplacements des points-clés.
        Si vous passez un drapeau, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, 
        il va dessiner un cercle de la taille du point-clé et même montrer son orientation.
        sow how to findContours from keypoints in SIFT in python
        '''
    # find the keypoints and descriptors with SIFT      maskMOG2
        keypoints, descriptors = sift.detectAndCompute(mask,None)
               
    #  Draw rich key points on input image
        frameSift = cv2.drawKeypoints(frame, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
           
        
        if ap!=0:
    # Convert Keypoints to an argument for findhomography()
#            queryFrame = mask.copy()
            queryFrame = maskMOG2.copy()
            queryFrame = frame.copy()
            queryKP, queryDesc = sift.detectAndCompute(queryFrame,None)
            matches=flann.knnMatch(queryDesc,trainDesc,k=2)
        # Apply ratio test
        # store all the good matches as per Lowe's ratio test.
            goodMatch = []
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    goodMatch.append(m)
            
            if(len(goodMatch) > MIN_MATCH_COUNT):
                tp=[]
                qp=[]
                for m in goodMatch:
                    tp.append( trainKP[m.trainIdx].pt)
                    qp.append( queryKP[m.queryIdx].pt)
                tp, qp = np.float32((tp, qp))
                H, status = cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
                h, w = trainFrame.shape
                trainBorder = np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
                queryBorder = cv2.perspectiveTransform(trainBorder,H)
                cv2.polylines(frameSift,[np.int32(queryBorder)],True,(0,255,0),5)
                print("Object found- %d / %d"%(len(goodMatch),MIN_MATCH_COUNT))
            else:
                print ("Not Enough match found- %d / %d" %(len(goodMatch),MIN_MATCH_COUNT))
            
            '''
            Feature Matching   =   Correspondance des fonctionnalités
            Les caractéristiques extraites d’images différentes à l’aide de SIFT ou SURF 
            peuvent être associées pour trouver des objets / motifs similaires présents 
            dans des images différentes.
            '''
        # BFMatcher with default params
            bfMatcher = cv2.BFMatcher()
            matches = bfMatcher.knnMatch(trainDesc,descriptors, k=2)
            
            goodMatch = []
            for m, n in matches:
                if m.distance < 0.75*n.distance:
                    goodMatch.append([m])
                    
            frameM = frame.copy()
        # cv2.drawMatchesKnn expects list of lists as matches.
            frameM = cv2.drawMatchesKnn(trainFrame, trainKP, frame, keypoints, goodMatch, frameM, flags=2)
        else :
            ap=1
        
        trainKP , trainDesc , trainFrame = keypoints.copy() , descriptors.copy() , frame.copy()
        
        cv2.moveWindow( 'Mean SIFT', 0, 0)
        cv2.moveWindow( 'Mask MOG2', 470, 0)
        cv2.moveWindow( 'SIFT', 470*2, 0)
        cv2.moveWindow( 'ROI', 470*2, 380)
        cv2.moveWindow( 'SIFT 2', 0, 380)
    
        cv2.imshow("Mean SIFT", frameD)
        cv2.imshow("Mask MOG2", maskMOG2)
        cv2.imshow("SIFT", frameSift)
        cv2.imshow("ROI", roi)
        cv2.imshow('SIFT 2',frameM)
        
    except Exception as e:
        print(' Ereur !! ',e)
        pass

    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('hand.png',frame)
    prvs = next
    
cv2.destroyAllWindows()
cap.release()
exit(0)

'''
SIFT no longer available in opencv > 3.4.2

I had the same problem. I change other opencv-python and opencv-contrib-python version, and solve this problem. 

Here is the history version about opencv-python : https://pypi.org/project/opencv-python/#history

and i use the following code :

==> pip install opencv-python==3.4.2.16
==> pip install opencv-contrib-python==3.4.2.16

For Anaconda User just this instead of pip

==> conda install -c menpo opencv

this will install cv2 3.4.1 and everything you need to run SIFT

\  To return to final version :
pip install opencv-python==4.1.0.25
pip install opencv-contrib-python==4.1.0.25

'''