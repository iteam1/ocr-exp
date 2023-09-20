import cv2
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
img_path = 'assets/txt.png'
cuda_opt = False
THRESH = 200

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
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img_resize, contours, -1, (0,255,0), 3)

cv2.imwrite('res.jpg',res)