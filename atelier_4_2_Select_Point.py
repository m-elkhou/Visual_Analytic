# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 08:59:05 2019

@author: EL KHOU Mohammed

"""

import cv2
import numpy as np
cap = cv2.VideoCapture('Car.mp4')

# Create old frame
_, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Lucas kanade params
lk_params = dict(winSize = (15, 15),
                 maxLevel = 4,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

old_points = None
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

# Mouse function
def select_point(event, x, y, flags, params):
    global old_points, mask, old_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        old_points = np.array([[x, y]], dtype=np.float32)
        mask = np.zeros_like(old_frame)

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", select_point)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_diff = frame.copy()

    if old_points is not None:
        # Calculate optical flow (i.e. track feature points)
        new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_points, None, **lk_params)
        a,b = new_points.ravel()
        print(a,b,'---------------------------')
        mask = cv2.circle(mask, (a, b), 2, (255,0,255), -1)
        
        frame_diff = cv2.add(frame, mask)
        old_gray = frame_gray.copy()
        old_points = new_points   
        
    cv2.imshow("Frame", frame_diff)
        
    key = cv2.waitKey(27)
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()