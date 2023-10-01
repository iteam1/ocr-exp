'''
python3 scripts/craft/crop_bbox.py
'''
import sys
sys.path.append('/home/gom/Workspace/ocr-exp/modules')

import os
import cv2
import random
import numpy as np
from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
    export_extra_results,
    empty_cuda_cache
)

#init
src ='data/sn'
dst = 'dst'
img_files = os.listdir(src)
img_path = os.path.join(src,random.choice(img_files))
cuda_opt = False
THRESH = 200
kernel = np.ones((10,7), np.uint8)

if not os.path.exists(dst):
    os.mkdir(dst)

# read image
img = read_image(img_path)

# load models
refine_net = load_refinenet_model(cuda=cuda_opt)
craft_net = load_craftnet_model(cuda=cuda_opt)

# perform prediction
pred = get_prediction(
    image=img,
    craft_net=craft_net,
    refine_net=refine_net,
    text_threshold=0.7,
    link_threshold=0.4,
    low_text=0.4,
    cuda=cuda_opt,
    long_size=1280
)

# get heatmap
txt_score = pred['heatmaps']['text_score_heatmap']
h,w,_ = txt_score.shape

# resize img
img_resize = cv2.resize(img,(w,h),interpolation = cv2.INTER_LINEAR)
img_org = img_resize.copy()

# convert to hsv
img_hsv = cv2.cvtColor(txt_score, cv2.COLOR_BGR2HSV)

# lower mask (0-10)
lower_min = np.array([0,50,50])
lower_max = np.array([10,255,255])
mask0 = cv2.inRange(img_hsv, lower_min, lower_max)

# upper mask (170-180)
upper_min = np.array([170,50,50])
upper_max = np.array([180,255,255])
mask1 = cv2.inRange(img_hsv, upper_min, upper_max)

# join my masks
mask = mask0+mask1

# find contours
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for i,cnt in enumerate(contours):
    # init
    X = []
    Y = []
    blank = np.zeros((h,w),np.uint8)
    cv2.drawContours(blank, [cnt], 0, (255), -1)
    
    # dilate
    blank = cv2.dilate(blank, kernel, iterations=5)
    coords = cv2.findNonZero(blank)
    for coord in coords:
        x,y = coord[0]
        X.append(x)
        Y.append(y)
    x_min = sorted(X)[0]
    x_max = sorted(X)[-1]
    y_min = sorted(Y)[0]
    y_max = sorted(Y)[0-1]

    #print(f'x_min {x_min},x_max {x_max},y_min {y_min},y_max {y_max}')
    character = img_org[y_min:y_max,x_min:x_max]
    cv2.rectangle(img_resize, (x_min,y_min), (x_max,y_max), (0,0,255), 1)
    cv2.imwrite(os.path.join(dst,f'char{i}.jpg'),cv2.cvtColor(character,cv2.COLOR_BGR2RGB))

cv2.imwrite(os.path.join(dst,'vis.jpg'),cv2.cvtColor(img_resize,cv2.COLOR_BGR2RGB))