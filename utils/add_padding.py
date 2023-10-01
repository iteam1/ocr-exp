'''
python3 utils/add_padding.py
'''
import os
import cv2
import random
import numpy as np

def add_padding(image):
    h,w,c = image.shape
    if w < h:
        dim = (h - w)//2
        new_width = dim * 2 + w
        padding = np.zeros((h,dim,3))
        new_image = np.hstack((padding,image,padding))
        return new_image
    else:
        dim = (w - h)//2
        new_height = dim *2 + h
        padding = np.zeros((dim,w,3))
        new_image = np.vstack((padding,image,padding))
        return new_image

# Initialize
dst = 'dst'
src ='data/characters'
image_files = os.listdir(src)
image_path = os.path.join(src,random.choice(image_files))

# Create destination folder
if not os.path.exists(dst):
    os.mkdir(dst)

# Read image
image = cv2.imread(image_path)
new_image = add_padding(image)

cv2.imwrite(os.path.join(dst,'vis.jpg'),new_image)

print('Done')