# -*- coding: utf-8 -*-
"""
Created on Sat May 25 00:47:54 2019

@author: mhmh2
"""
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QHBoxLayout

def AprocheSimple():
    video = cv2.VideoCapture("1.mp4")
    background = cv2.imread('1.png',0)

    while True:
        _,img = video.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#        mask = abs(img - background)
        mask = cv2.absdiff(img, background)
        cv2.imshow("La différence d’images", mask)
    
        # tab 'ESC'
        key = cv2.waitKey(20) 
        if key == 27: # hadi hiyya Esc
            video.release()
            cv2.destroyAllWindows()
            break

# Defference d'images _ Frame differencing
def FrameDifferencing():
    cap = cv2.VideoCapture(0)

    ret, current_frame = cap.read()
    previous_frame = current_frame

    while(cap.isOpened()):
    
#       current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
#       previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)    
#       frame_diff = cv2.absdiff(current_frame_gray,previous_frame_gray)
#       frame_diff = abs(current_frame_gray-previous_frame_gray)
    
        frame_diff = cv2.absdiff(current_frame,previous_frame)
#       frame_diff = abs(current_frame-previous_frame)
    
        cv2.imshow('frame diff ',frame_diff)    
    
#   tab 'ESC'
        key = cv2.waitKey(20) 
        if key == 27: # hadi hiyya Esc
            cap.release()
            cv2.destroyAllWindows()
            break
    
        previous_frame = current_frame.copy()
        ret, current_frame = cap.read()
        

        

win = QMainWindow()
app = QApplication([])
central_widget = QWidget()

button_1 = QPushButton('Aproche Simple', central_widget)
button_2 = QPushButton('Frame Differencing', central_widget)
button_1.clicked.connect(AprocheSimple)
button_2.clicked.connect(FrameDifferencing)
layout = QVBoxLayout(central_widget)
#layout = QHBoxLayout(central_widget)
layout.addWidget(button_2)
layout.addWidget(button_2)
win.setCentralWidget(central_widget)
win.show()
app.exit(app.exec_())
