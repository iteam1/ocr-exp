import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import imgaug as ia
import imgaug.augmenters as iaa
import pickle
import json
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC

ia.seed(1)

path = "samples"
labels = os.listdir(path)
print(labels)

features = []
for l in labels:
    f = os.listdir(os.path.join(path,l))
    features.append(len(f))

# find unique value
n = 1
for f in list(set(features)):
    n *=f
    
print(features)
print('n= ',n)

ms = []
for f in features:
    ms.append(int(n/f))
print(ms)

seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.LinearContrast((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        rotate=(-10, 10),
        shear=(-4, 4)
    )
], random_order=True) # apply augmenters in random order

DIM=227
N = 100 #150 overlimit

path ="samples"
sub = 'train'

if not os.path.isdir(f'{sub}'):
        os.mkdir(f'{sub}')
        
for i,label in enumerate(labels):
    
    # create train/label
    if not os.path.isdir(f'{sub}/{label}'):
        os.mkdir(f'{sub}/{label}')
    
    label_dir = os.path.join(path,label)
    imgs = os.listdir(label_dir)
    for im in imgs:
        img = cv2.imread(os.path.join(label_dir,im),0)
        # resize image
        img = cv2.resize(img,(DIM,DIM), interpolation = cv2.INTER_AREA)
        # duplicate
        images = np.array([ img for _ in range(N*ms[i])],dtype=np.uint8)
        # augmentation
        images_aug = seq(images=images)
        
        for j in range(N*ms[i]):
            cv2.imwrite(f"{sub}/{label}/{im.strip('.jpg')}_{j}.jpg",images_aug[j])
            
            
N = 20
path ="samples"
sub = 'test'

if not os.path.isdir(f'{sub}'):
        os.mkdir(f'{sub}')
        
for i,label in enumerate(labels):
    
    # create train/label
    if not os.path.isdir(f'{sub}/{label}'):
        os.mkdir(f'{sub}/{label}')
    
    label_dir = os.path.join(path,label)
    imgs = os.listdir(label_dir)
    for im in imgs:
        img = cv2.imread(os.path.join(label_dir,im),0)
        # resize image
        img = cv2.resize(img,(DIM,DIM), interpolation = cv2.INTER_AREA)
        # duplicate
        images = np.array([ img for _ in range(N*ms[i])],dtype=np.uint8)
        # augmentation
        images_aug = seq(images=images)
        
        for j in range(N*ms[i]):
            cv2.imwrite(f"{sub}/{label}/{im.strip('.jpg')}_{j}.jpg",images_aug[j])
            
X = []
y = [] #number

path="train"
IMG_SIZE=100

labels = os.listdir(path)

for l in labels:
    imgs = os.listdir(os.path.join(path,l))
    for im in imgs:
        img = cv2.imread(os.path.join(path,l,im),0)
        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        X.append(img)
        y.append(int(l))
        
# shuffle
X,y = shuffle(X,y)

X = np.array(X).reshape(len(X),-1)
# norm
X = X/255.0
y = np.array(y)

X = np.array(X).reshape(len(X),-1)
# norm
X = X/255.0
y = np.array(y)

# split
X_train, X_val, y_train, y_val = train_test_split(X,y)
print("X_train: ",X_train.shape)
print("X_val: ",X_val.shape)
print("y_train: ",y_train.shape)
print("y_val: ",y_val.shape)

print('traing...')

svc = SVC(kernel='rbf',gamma='auto') #linear
svc.fit(X_train, y_train)

print('testing...')

y2 = svc.predict(X_val)

# calc accuracy
print("Accuracy on validation dataset is",accuracy_score(y_val,y2))

print("Accuracy on validation dataset is")
print(classification_report(y_val,y2))

# save the model to disk
filename = 'ppocr/obj_cla.sav'
pickle.dump(svc, open(filename, 'wb'))
