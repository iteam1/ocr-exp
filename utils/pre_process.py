'''
python3 utils/pre_process.py
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
src ='data/labels'
labels = os.listdir(src)

# Create destination folder
if not os.path.exists(dst):
    os.mkdir(dst)

for label in labels:
    
    print('processing',label)

    # Create label to destination folder
    label_path = os.path.join(dst,label)
    if not os.path.exists(label_path):
        os.mkdir(label_path)

    images = os.listdir(os.path.join(src,label))

    for image in images:
        image_path = os.path.join(src,label,image)
        img = cv2.imread(image_path)
        padding_img = add_padding(img)
        cv2.imwrite(os.path.join(dst,label,image),padding_img)

print('Done!')