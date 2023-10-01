'''
python3 scripts/classify/classify_data.py
'''
import os
import cv2
import torch
import random
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms

# Init
DIM = 224
THRESH = 0.8
dst = 'dst'
src ='data/characters'
image_files = os.listdir(src)
output_path = 'training'
classes_path = os.path.join(output_path,'classes.txt')
model_path = os.path.join(output_path,'checkpoint.pt')

# Create destination folder
if not os.path.exists(dst):
    os.mkdir(dst)

transform = transforms.Compose([
    transforms.Resize((DIM,DIM)),
    transforms.ToTensor()])

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

def predict(img,model,classes):

    img = img.astype(np.uint8)

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

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Predict on',device)

    # Load classes
    with open(classes_path,'r') as f:
        classes = f.readlines()
    classes = [x.replace('\n','') for x in classes]
    print(classes)

    # Load model
    model = torch.load(model_path)
    model = model.to(device)
    model.eval()

    for image_file in tqdm(image_files):
        image_path = os.path.join(src,image_file)

        # Read image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img_padding = add_padding(img)

        # Predict
        pred_class,pred_coef = predict(img_padding,model,classes)

        # Create destination folder
        class_path = os.path.join(dst,pred_class)
        if not os.path.exists(class_path):
            os.mkdir(class_path)
        
        # Move file to destination folder
        dst_path = os.path.join(dst,pred_class,image_file)
        shutil.copy(image_path,dst_path)

    print('Done!')