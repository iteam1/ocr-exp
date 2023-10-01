'''
python3 scripts/classify/batch_predict.py
'''
import os
import cv2
import torch
import random
import numpy as np
from PIL import Image
from torchvision import models, transforms

# Init
DIM = 224
THRESH = 0.8
NO_IMAGE = [1,2,3,4,5,6]
dst = 'dst'
src ='data/characters'
output_path = 'training'
image_files = os.listdir(src)
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

def batch_predict(imgs,model,classes):

    batch_tensors = []
    hstack_img = np.hstack((cv2.resize(img,(DIM,DIM),interpolation=cv2.INTER_AREA) for img in imgs))
    cv2.imwrite(os.path.join(dst,'vis.jpg'),hstack_img)
    
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
    
    characters = [classes[pred_class] for pred_class in predicted_classes]

    # Print the list of predicted classes
    print(characters)

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

    # Choice random number of image for predict
    no_img = random.choice(NO_IMAGE)
    batch_imgs = []

    for i in range(no_img):

        image_path = os.path.join(src,random.choice(image_files))

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img_padding = add_padding(img)
        batch_imgs.append(img_padding)

    # Predict
    batch_predict(batch_imgs,model,classes)
    
    print('Done!')