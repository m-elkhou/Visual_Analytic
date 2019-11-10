# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:20:23 2019

@author: EL KHOU Mohammed

Atelier 1:
•Lecture d’une vidéo enregistrer sur le disque dur.
"""
import cv2
#  Lecture d’une vidéo capture avec le camera de laptop
#cap = cv2.VideoCapture(0)

# Lecture d’une vidéo enregistrer sur le disque dur
cap = cv2.VideoCapture('Car.mp4')

while True:
    # lire le video image par image
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)
#    frame = cv2.flip(frame, 1)
    cv2.moveWindow('Frame',700,10)
    
#    affichier l'image dans une frame a chaque instance
    cv2.imshow("Frame", frame)

#   key listener sur le clavier a chaq 20 mili segonde
    key = cv2.waitKey(20) 
    if key == 27: # hadi hiyya Esc
        break

cap.release()
cv2.destroyAllWindows()
