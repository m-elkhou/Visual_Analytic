# -*- coding: utf-8 -*-
"""
Created on Fri May 24 16:11:15 2019

@author:  EL KHOU Mohammed

Atelier2:
•En utilisant une vidéo enregistrée sur le disque dur ou capture à partir d’une web Cam. 
    Réaliser une segmentation avant-plan/arrière-plan en utilisant:
•La différence d’images
•La dérivation temporelle
•La moyenne mobile (Movingaverage)
•La moyenne glissante (Running Average)
•Le filtre Médian
"""
import cv2
cv2.__version__

video = cv2.VideoCapture('1.mp4')

for _ in range(6):
    _, background = video.read()
background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
# background = cv2.imread('1.png',0)

while True:
    test, img = video.read()
    if not test:    
        break

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#    mask = abs(img - background)
    mask = cv2.absdiff(img, background)
    
    cv2.imshow("La différence d’images", mask)
    
    # tab 'ESC'
    key = cv2.waitKey(20) 
    if key == 27: # hadi hiyya Esc
        break
        
video.release()
cv2.destroyAllWindows()