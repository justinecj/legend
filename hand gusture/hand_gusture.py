#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 11:00:49 2018

@author: justine
"""

import cv2
import numpy as np
import sys

palmPath = "palm.xml"
closed_frontal_palmPath = "closed_frontal_palm.xml"

palmCascade = cv2.CascadeClassifier(palmPath)
closed_frontal_palm_Cascade = cv2.CascadeClassifier(closed_frontal_palmPath)

cap = cv2.VideoCapture(0)
filename = "output.mp4"
fps = 5
size = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.cv.FOURCC('8', 'B', 'P', 'S')     #works, large
#fourcc= cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter(filename, fourcc, fps, size, True)

while True:

    ret, frame = cap.read() # Capture frame-by-frame
    img = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    
    palm = palmCascade.detectMultiScale(gray, 1.3, 5)
    # ---- Draw a rectangle around the faces

    for (x, y, w, h) in palm:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        font= cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText( img, "Hai", (30,60), font, 1, (255,255,255), 3)
    
    
    
    closed_frontal_palm = closed_frontal_palm_Cascade.detectMultiScale(gray, 1.3, 5)
    # ---- Draw a rectangle around the faces

    for (x, y, w, h) in closed_frontal_palm:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        font= cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText( img, "How are you?", (30,60), font, 1, (255,255,255), 3)
    
    
    
    

    

    #cv2.cv.Flip(frame, None, 1)
    out.write(frame)
    
    cv2.imshow('Smile Detector', img)
    #c = cv2.cv.WaitKey(7) % 0x100
    #if c == 27:
        #break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()