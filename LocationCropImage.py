#!/usr/bin/env python

import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
import math

imgBig = cv2.imread('StarMap.png',1) #Big/main image
imgCrop = cv2.imread('Small_area_rotated.png',1) #cropped image

def pointsCrop(img1,img2):
    deg=0
    maxVal=0
    maxLoc=(0,0)
    minLoc=(0,0)
    imgR=img2
    h, w, c = img2.shape
    #The cropped image will be rotated until the best match is found
    for r in range(0, 360, 2):
        imgRot = imutils.rotate_bound(img2, r)         
        method = eval('cv2.TM_CCOEFF_NORMED')  #method is the template matching method      
        res = cv2.matchTemplate(img1,imgRot,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        #maxVal is the best match score value
        if max_val>maxVal:
            maxVal=max_val
            maxLoc=max_loc #maxLoc is the best match location of the big image
            minLoc=min_loc #minLoc is the best match location of the big image
            deg=r #deg is the best match rotation angle
            imgR=imgRot #imgR is the best match rotated image
            h2, w2, c2 = imgRot.shape
            
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = minLoc
    else:
        top_left = maxLoc        
    bottom_right = (top_left[0] + w2, top_left[1] + h2)

    #top_left is the corner of rectangle.
    #But we're looking for rotated rectangular points.
    #So;
    angle = deg * math.pi / 180.0 #angle transformation degrees to radians
    b = math.cos(angle) * 0.5
    a = math.sin(angle) * 0.5
    x0=top_left[0]+h2/2
    y0=top_left[1]+w2/2
    pt0 = (int(x0 - a * h - b * w), int(y0 + b * h - a * w))
    pt1 = (int(x0 + a * h - b * w), int(y0 - b * h - a * w))
    pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
    pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))

    return pt0,pt1,pt2,pt3

points = pointsCrop(imgBig,imgCrop)

rect = cv2.minAreaRect(np.int32(points))
box = cv2.boxPoints(rect)
box = np.int0(box)
image2=cv2.drawContours(imgBig,[box],0,(0,0,255),3)
cv2.imshow('ImgBig', imgBig)
cv2.imshow('imgCrop',imgCrop)
cv2.waitKey(0)
cv2.destroyAllWindows()
