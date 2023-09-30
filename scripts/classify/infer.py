'''
python3 infer.py <path_to_image>
'''
import os
import sys
import cv2
import time
import torch
import pickle
import numpy as np
from PIL import Image
from torchvision import models, transforms
from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
    export_extra_results,
    empty_cuda_cache
)

# Init
crop_opt = False
cuda_opt = False
DIM = 224
THRESH = 0.5
image_path = sys.argv[1]
dst = 'dst'
training_path ='training'
classes_path = os.path.join(training_path,'classes.txt')
model_path = os.path.join(training_path,'checkpoint.pt')
string = ''
kernel = np.ones((10,5), np.uint8)

transform = transforms.Compose([
    transforms.Resize((DIM,DIM)),
    transforms.ToTensor()])

def predict(img,model,classes):

    # Read image
    im_pil = Image.fromarray(img)
    input = transform(im_pil)
    input = torch.unsqueeze(input, 0)
    input = input.to(device)

    # Predict
    output = model(input)

    # Post Process
    output = output.softmax(1)
    output = output.cpu().detach().numpy() 
    output = output.ravel()
    output_argmax = np.argmax(output)
    output_prob = output[output_argmax]
    output_class = classes[output_argmax]

    return (output_class,output_prob)

def crop_char(img_path,refine_net,craft_net):
    
    # init
    bboxes = []
    
    # load image
    img = read_image(img_path)
    
    # predict
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
        blank = cv2.dilate(blank, kernel, iterations=7)
        coords = cv2.findNonZero(blank)
        for coord in coords:
            x,y = coord[0]
            X.append(x)
            Y.append(y)

        x_min = sorted(X)[0]
        x_max = sorted(X)[-1]
        y_min = sorted(Y)[0]
        y_max = sorted(Y)[-1]
        
        #box = [[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min,y_max]]
        box = [x_min,y_min,x_max,y_max]

        #print(f'x_min {x_min},x_max {x_max},y_min {y_min},y_max {y_max}')
        character = img_org[y_min:y_max,x_min:x_max]
        cv2.rectangle(img_resize, (x_min,y_min), (x_max,y_max), (0,0,255), 1)
        bboxes.append(box)
    
    # sort bboxes horizontaly
    dict_x = {}
    list_x  = []
    order_bboxes = []
    for box in bboxes:
        x_min = box[0]
        y_min = box[1]
        x_max = box[2]
        y_max = box[3]
        x_center = int((x_min+x_max)/2)
        y_center = int((y_min+y_max)/2)
        dict_x[x_center] = box
        list_x.append(x_center)
    
    list_x = sorted(list_x)
    for x in list_x:
        order_bboxes.append(dict_x[x])

    #cv2.imwrite('res.jpg',img_resize)
    
    return img_org, order_bboxes, img_resize

if __name__ == "__main__":
    
    print("CRAFT crop characters")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Predict on',device)

    # create destination folder
    if not os.path.exists(dst):
        os.mkdir(dst)
    
    image_name = image_path.split('/')[-1]
    image_name = image_name.split('.')[0]

    # Load classes
    with open(classes_path,'r') as f:
        classes = f.readlines()
    classes = [x.replace('\n','') for x in classes]
    print(classes)

    # load models
    refine_net = load_refinenet_model(cuda=cuda_opt)
    craft_net = load_craftnet_model(cuda=cuda_opt)
    resnet18 = torch.load(model_path)
    resnet18 = resnet18.to(device)
    resnet18.eval()
    print("Load models done")
    
    start_time = time.time()

    # preprocess
    img_resize, bboxes,_ = crop_char(image_path,refine_net,craft_net)
    img_debug = img_resize.copy()
    
    for j,bbox in enumerate(bboxes):
        x_min,y_min,x_max,y_max = bbox
        x_center = int((x_min+x_max)/2)
        y_center = int((y_min+y_max)/2)
        c = img_resize[y_min:y_max,x_min:x_max]
        if crop_opt:
            cv2.imwrite(f'c_{j}.jpg',c)
        pred, coef = predict(cv2.cvtColor(c,cv2.COLOR_BGR2RGB),resnet18,classes)
        if coef > THRESH:
            if pred == '0':
                pred = '@'
            elif pred == 'O':
                pred = 'O'
        else:
            pred = "?"
        string += str(pred)
        cv2.putText(img_debug,str(pred),(x_center,y_center), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255,0,255), 2, cv2.LINE_AA)
        
        # cv2.imwrite(os.path.join(dst,f'{j}_{pred}.jpg'),c)
        
    print(f"Done in {time.time()-start_time} s")
    
    cv2.imwrite(os.path.join(dst,image_name+'.jpg'),img_debug)