'''
python3 scripts/classify/predict.py
'''
import os
import sys
import cv2
import torch
import random
import numpy as np
from PIL import Image
from torchvision import models, transforms

# Init
DIM = 224
THRESH = 0.8
src ='data/characters'
image_files = os.listdir(src)
image_path = os.path.join(src,random.choice(image_files))
output_path = 'training'
classes_path = os.path.join(output_path,'classes.txt')
model_path = os.path.join(output_path,'checkpoint.pt')

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

    # Read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    # Predict
    pred = predict(img,model,classes)
    print(pred)

    # Display
    cv2.imshow('visualize',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
    print('Done!')