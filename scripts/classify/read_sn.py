'''
python3 scripts/classify/read_sn.py
'''
import sys
sys.path.append('/home/gom/Workspace/ocr-exp/modules')

import os
import cv2
import random
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
)

# initialize
DIM = 224
THRESH = 0.8
output_path = 'training'
classes_path = os.path.join(output_path,'classes.txt')
model_path = os.path.join(output_path,'checkpoint.pt')

src ='data/sn'
dst = 'dst'
img_files = os.listdir(src)
img_path = os.path.join(src,random.choice(img_files))
cuda_opt = False
THRESH = 200
kernel = np.ones((10,7), np.uint8)

if not os.path.exists(dst):
    os.mkdir(dst)

# define code block

transform = transforms.Compose([
    transforms.Resize((DIM,DIM)),
    transforms.ToTensor()])

def sort_bounding_boxes_by_area(bounding_boxes):
    """
    Sort a list of bounding boxes by their area in ascending order.

    Args:
        bounding_boxes (list): List of bounding boxes, each represented as a tuple (x_min, y_min, x_max, y_max).

    Returns:
        list: List of bounding boxes sorted by area in ascending order.
    """
    return sorted(bounding_boxes, key=lambda box: (box[2] + box[0])/2)

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

def batch_predict(imgs,model,classes):

    # init
    batch_tensors = []
    hstack_img = np.hstack((cv2.resize(img,(DIM,DIM),interpolation=cv2.INTER_AREA) for img in imgs))
    cv2.imwrite(os.path.join(dst,'res.jpg'),hstack_img) #  debug
    
    for img in imgs:
        # Read image
        im_pil = Image.fromarray(img.astype(np.uint8))
        input = transform(im_pil)
        input = torch.unsqueeze(input, 0)
        input = input.to(device)
        batch_tensors.append(input)
    
    # Stack the preprocessed images into a batch tensor
    batch_tensors = torch.stack(batch_tensors,dim=1)[0]

    # Make predictions for the entire batch
    with torch.no_grad():
        output = model(batch_tensors)

    # Assuming it's a classification model, you can get the predicted classes for each image in the batch
    # This will give you a tensor of predicted class indices for each image in the batch
    predicted_classes = torch.argmax(output, dim=1)

    # Convert the tensor to a Python list
    predicted_classes = predicted_classes.tolist()
    
    pred_of_characters = [classes[pred_class] for pred_class in predicted_classes]

    # Print the list of predicted classes
    print(pred_of_characters)

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Predict on',device)

    # Load classes
    with open(classes_path,'r') as f:
        classes = f.readlines()
    classes = [x.replace('\n','') for x in classes]
    print(classes)

    # read image
    img = read_image(img_path)

    # load models
    refine_net = load_refinenet_model(cuda=cuda_opt)
    craft_net = load_craftnet_model(cuda=cuda_opt)
    # Load model
    model = torch.load(model_path)
    model = model.to(device)
    model.eval()

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
    characters = []
    batch_imgs = []
    bbox_coords = []
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
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

        # find bounding box coordinate
        x_min = sorted(X)[0]
        x_max = sorted(X)[-1]
        y_min = sorted(Y)[0]
        y_max = sorted(Y)[-1]
        #print(f'x_min {x_min},x_max {x_max},y_min {y_min},y_max {y_max}')
        bbox_coords.append((x_min,y_min,x_max,y_max))

        cv2.rectangle(img_resize, (x_min,y_min), (x_max,y_max), (0,0,255), 1)

    # sort coordinate
    sorted_bbox_coords = sort_bounding_boxes_by_area(bbox_coords)

    for i,coord in enumerate(sorted_bbox_coords):
        x_min,y_min,x_max,y_max = coord
        character = img_org[y_min:y_max,x_min:x_max]
        characters.append(character)
        cv2.rectangle(img_resize, (x_min,y_min), (x_max,y_max), (0,0,255), 1)
        cv2.imwrite(os.path.join(dst,f'char{i}.jpg'),cv2.cvtColor(character,cv2.COLOR_BGR2RGB))

    cv2.imwrite(os.path.join(dst,'vis.jpg'),cv2.cvtColor(img_resize,cv2.COLOR_BGR2RGB))

    for character in characters:

        img = cv2.cvtColor(character,cv2.COLOR_BGR2RGB)
        img_padding = add_padding(img)
        batch_imgs.append(img_padding)

    # Predict
    batch_predict(batch_imgs,model,classes)