'''
https://stackoverflow.com/questions/65157017/creating-a-mask-image-using-range-sliders-trackbars-with-opencv-and-python
'''
import cv2
import numpy as np
import random
import os

def empty(i):
    pass

def on_trackbar(val):
    global img,hsv,img_name,res
    hue_min = cv2.getTrackbarPos("Hue Min", "TrackedBars")
    hue_max = cv2.getTrackbarPos("Hue Max", "TrackedBars")
    sat_min = cv2.getTrackbarPos("Sat Min", "TrackedBars")
    sat_max = cv2.getTrackbarPos("Sat Max", "TrackedBars")
    val_min = cv2.getTrackbarPos("Val Min", "TrackedBars")
    val_max = cv2.getTrackbarPos("Val Max", "TrackedBars")
    scale   = cv2.getTrackbarPos("Scale", "TrackedBars")

    lower = np.array([hue_min, sat_min, val_min])
    upper = np.array([hue_max, sat_max, val_max])

    mask = cv2.inRange(hsv, lower, upper)

    res = cv2.bitwise_and(img,img,mask=mask)

    h,w,c = res.shape
    h = int(h*scale/100)
    w = int(w*scale/100)
    resized = cv2.resize(res,(0,0),fx=scale/100,fy=scale/100)

    cv2.imshow(img_name,resized)

# random read image
path = "./gg"
label = random.choice(os.listdir(path))
img_name = random.choice(os.listdir(os.path.join(path,label)))
img = cv2.imread("gg/charcoal/charcoal_4.jpg") #os.path.join(path,label,img_name
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#create trackbar
cv2.namedWindow("TrackedBars")
cv2.resizeWindow("TrackedBars", 640, 240)

cv2.createTrackbar("Hue Min", "TrackedBars", 0, 179, on_trackbar)
cv2.createTrackbar("Hue Max", "TrackedBars", 179, 179, on_trackbar)
cv2.createTrackbar("Sat Min", "TrackedBars", 0, 255, on_trackbar)
cv2.createTrackbar("Sat Max", "TrackedBars", 255, 255, on_trackbar)
cv2.createTrackbar("Val Min", "TrackedBars", 0, 255, on_trackbar)
cv2.createTrackbar("Val Max", "TrackedBars", 255, 255, on_trackbar)
cv2.createTrackbar("Scale", "TrackedBars",20, 100, on_trackbar)

# Show some stuff
on_trackbar(0)

# Wait until user press some key
k = cv2.waitKey()
if k == 27:
    cv2.imwrite("hsv.jpg",res)
