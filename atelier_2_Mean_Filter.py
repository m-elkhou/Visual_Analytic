# -*- coding: utf-8 -*-
"""
Created on Sat May 25 08:13:10 2019

@author: Mohammed EL KHOU
"""
import cv2
import numpy as np


#  Lecture d’une vidéo capture avec le camera de laptop
#cap = cv2.VideoCapture(0)

# Lecture d’une vidéo enregistrer sur le disque dur
#cap = cv2.VideoCapture('C:/Users/mhmh2/Downloads/Video/musicDance.mp4')
cap = cv2.VideoCapture('highway.mp4')

n = 3  # nomber d'image pris pour faire la moyenne
Img_List = []

# l'apprentissage initiale
for _ in range(n):
    _,background = cap.read()
    background = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
    background = cv2.GaussianBlur(background,(5,5),0)
    Img_List.append(background)

while cap.isOpened():
    # lire le video image par image 
    ret, img = cap.read()
    if not ret:    # test si le video est terminer
        break
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(5,5),0)
    
    # convirtir le tableau 2 ArrayList
    arrayList = np.array(Img_List)
    
    
    Background = arrayList.mean(0).astype('uint8')

    for i in range(n-1):    # Ecrasi l'img avec l'img suivant
        Img_List[i] = Img_List[i+1]
    Img_List[n-1]=img # Ajouter la novel img a la fin du tableau
    
    frame_diff = cv2.absdiff(img,Background)

    _,frame_diff = cv2.threshold(frame_diff,20,255,cv2.THRESH_BINARY) # faire le seuillage globale
        
    img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
    cv2.moveWindow('Frame',10,10)
    
    Background = cv2.resize(Background, dsize=None, fx=0.5, fy=0.5)
    cv2.moveWindow('Background Mean Filter',700,10)
    
    frame_diff = cv2.resize(frame_diff, dsize=None, fx=0.5, fy=0.5)
    cv2.moveWindow('Mean Filter',10,350)
    
#    affichier l'image dans une frame a chaque instance
    cv2.imshow("Frame", img)
    cv2.imshow("Background Mean Filter", Background)
    cv2.imshow("Mean Filter", frame_diff)
    
#   key listener sur le clavier a chaq 20 mili segonde
    key = cv2.waitKey(20) 
    if key == 27: # hadi hiyya Esc
        break
    
cap.release()
cv2.destroyAllWindows()