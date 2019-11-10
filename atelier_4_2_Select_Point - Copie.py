 # -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 08:59:05 2019

@author: EL KHOU Mohammed

"""

import cv2
import numpy as np
cap = cv2.VideoCapture('highway.mp4')

# Create old frame
_, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Lucas kanade params
lk_params = dict(winSize = (15, 15),
                 maxLevel = 4,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

point_selected = False
points = []

# Mouse function
def select_point(event, x, y, flags, params):
    global point_selected, old_points
    if event == cv2.EVENT_LBUTTONDOWN:
        point_selected = True
        points.append([x, y])

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", select_point)

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
while cap.isOpened():
    ret, new_frame = cap.read()
    if not ret:
        break
    gray_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    
    if point_selected:
        old_points = np.array(points, dtype=np.float32)
        # Calculate optical flow (i.e. track feature points)
        new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
        
        # Select good points
#        good_new = new_points[status==1]
        good_new = new_points
        
        # draw the tracks
        for i,new in enumerate(good_new):
            a,b = new.ravel()
            mask = cv2.circle(mask,(a,b),2,color[i].tolist(),-1)
            
        new_frame = cv2.add(new_frame,mask)
        
        # Now update the previous frame and previous points
        old_gray = gray_frame.copy()
        old_points = new_points
            
    cv2.imshow("Frame", new_frame)
        
    key = cv2.waitKey(27)
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()