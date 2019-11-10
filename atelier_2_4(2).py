# -*- coding: utf-8 -*-
"""
Created on Sat May 25 18:54:33 2019

@author: Mohammed EL KHOU

La moyenne glissante (RunningAverage)
"""
import matplotlib
matplotlib.use('TkAgg')

import cv2
import numpy as np
#from matplotlib import pyplot as plt
import matplotlib.pyplot as plt

#  Lecture d’une vidéo capture avec le camera de laptop
#cap = cv2.VideoCapture(0)

# Lecture d’une vidéo enregistrer sur le disque dur
cap = cv2.VideoCapture('highway.mp4')

ret, current_background= cap.read()
previous_background = current_background

fig = plt.figure(1)

cpp = 0
while cap.isOpened():
    # lire le video image par image
    ret, frame = cap.read()
    if not ret:    
        cap.release()
        cv2.destroyAllWindows()
        break
    
    cpp +=1
    a = float(1.00)/float(cpp)
    if a < 1.00/1000:   # temp d'apprentissage
        a= 1.00/1000
    
    # modify the data type 
    # setting to 32-bit floating point 
    f = np.float32(frame) 
    p_b = np.float32(previous_background)
    
    # using the cv2.accumulateWeighted() function 
    # that updates the running average 
#    cv2.accumulateWeighted(f, averageValue, 0.02) 
    
    averageValue = (1.00-a) * p_b  + a * f
    
    # converting the matrix elements to absolute values  
    # and converting the result to 8-bit.  
    current_background = cv2.convertScaleAbs(averageValue) 
    
    frame_diff = cv2.absdiff(frame,current_background)
    
#    affichier l'image dans une frame a chaque instance
    
    plt.subplot(2, 2, 1),  plt.imshow(frame)
    plt.title('Frame')
    plt.xticks([]),plt.yticks([])

    plt.subplot(2, 2, 2), plt.imshow(current_background)
    plt.title('Background')
#    plt.axis("off")
    plt.xticks([]),plt.yticks([])
    
    plt.subplot(2, 2, 3), plt.imshow(frame_diff)
    plt.title('Running Average')
#    plt.axis("off")
    plt.xticks([]),plt.yticks([])
        
    # redraw the canvas
    fig.canvas.draw()
    # convert canvas to image
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,sep='')
    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    # img is rgb, convert to opencv's default bgr
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    
#    img = cv2.resize(img, dsize=None, fx=2, fy=2) 
#     display image with opencv or any operation you like
    cv2.imshow("plot",img)
    
#    fig.show()
#    plt.show()
    
    previous_background = current_background.copy()
    ret, current_background = cap.read()
    
    key = cv2.waitKey(10) 
    if key == 27: # hadi hiyya Esc
        cap.release()
        cv2.destroyAllWindows()
#        fig.
        break
    