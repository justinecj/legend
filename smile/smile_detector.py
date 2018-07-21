#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 16:39:45 2018

@author: justine
"""

import cv2
import numpy as np
import sys

facePath = 'haarcascade_frontalface_default.xml'
smilePath = "haarcascade_smile.xml"
faceCascade = cv2.CascadeClassifier(facePath)
smileCascade = cv2.CascadeClassifier(smilePath)

cap = cv2.VideoCapture(0)

"""filename = "output.mp4"
fps = 5
size = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.cv.FOURCC('8', 'B', 'P', 'S')     #works, large
#fourcc= cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter(filename, fourcc, fps, size, True)"""


cap.set(3,640)
cap.set(4,480)

sF = 1.05

while True:

    ret, frame = cap.read() # Capture frame-by-frame
    img = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor= sF,
        minNeighbors=8,
        minSize=(55, 55),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    # ---- Draw a rectangle around the faces

    for (x, y, w, h) in faces:
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        smile = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor= 1.7,
            minNeighbors=22,
            minSize=(25, 25),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
            )

        # Set region of interest for smiles
        for (x, y, w, h) in smile:
            print "Found", len(smile), "smiles!"
            font= cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText( img, "Good smile", (10,40), font, 1, (0,0,255), 3)
           # cv2.rectangle(roi_color, (x, y), (x+w, y+h), (255, 0, 0), 1)
            #print "!!!!!!!!!!!!!!!!!"

    #cv2.cv.Flip(frame, None, 1)
    #out.write(frame)
    
    cv2.imshow('Smile Detector', img)
    #c = cv2.cv.WaitKey(7) % 0x100
    #if c == 27:
        #break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
#out.release()
cv2.destroyAllWindows()