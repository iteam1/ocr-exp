'''
Generate image by imgaug from floder contain original images
CMD: python3 scripts/classify/augment.py <path/to/dataset> <type> 300
'''
import os
import sys
import cv2
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from tqdm import tqdm

ia.seed(1)

# Initialize
DIM = 224
src = sys.argv[1] # path of original images
dst = sys.argv[2] # path of augmented images
typ = sys.argv[3] # train ,val or test
num = int(sys.argv[4]) # number of images augmented

# Create destination folder
if not os.path.exists(dst):
    os.mkdir(dst)

if not os.path.exists(os.path.join(dst,typ)):
    os.mkdir(os.path.join(dst,typ))

def preprocess(img):
    '''
    process input image
    Args:
        img: input image
    Return:
        out: output image
    '''
    output = cv2.resize(img,(DIM,DIM),interpolation=cv2.INTER_AREA)
    return output

if __name__ == "__main__":
    
    N = num # number of images augmented
    
    # check original images
    if not os.path.exists(src):
        print("Can not found folder: ",src)
        exit()
    
    # get labels of original images
    labels = os.listdir(src)
    print("Total: ",len(labels)," (labels)")
    
    # check the quantity
    for label in labels:
        path = os.path.join(src,label)
        imgs = os.listdir(path)
        print(f'- {label}: {len(imgs)}')
    
    # check the destination folder, if not exist create it
    if not os.path.exists(dst):
        os.mkdir(dst)
        print("Created ",dst)
        
    # check the subset folder, if not exist create it
    path = os.path.join(dst,typ)
    if not os.path.exists(path):
        os.mkdir(path)
        print("Created ",path)
    
    # create label folders, it not exist create it
    for label in labels:
        path = os.path.join(dst,typ,label)
        if not os.path.exists(path):
            os.mkdir(path)
            print("Created ",path)
    
    # instance of image augmentation
    augmentation = iaa.Sequential([
    # iaa.Resize({"height": 0.5, "width": 0.5}),
    iaa.Flipud(0.5),                                                # Horizontal flip 50% of images
    # iaa.Crop(percent=(0, 0.20)),                                  # Crop all images between 0% to 20%
    iaa.GaussianBlur(sigma=(0, 0.5)),                               # Add slight blur to images
    iaa.Multiply((0.95, 1.05), per_channel=0.02),                   # Slightly brighten, darken, or recolor images    
    iaa.AdditiveGaussianNoise(scale=(0, 0.005 * 255)),              # Add Gaussian noise
    iaa.ContrastNormalization((0.95, 1.05)),                        # Change contrast
    iaa.Affine(
        scale={"x": (0.95, 1.05), "y": (0.95,1.05)},                # Resize image
        # translate_percent={"x": (0, 0.1), "y": (-0.1, 0.03)},     # Translate image
        rotate=(-7, 7),                                             # Rotate image
        shear={"x": (-1, 1)}                                        # Skew along the x-axis
        # mode=ia.ALL, cval=(0,255)                                 # Filling in extra pixels
        )
    ])
    
    # generate augmented images
    for label in labels:
        path = os.path.join(src,label)
        # list all images
        imgs = os.listdir(path)
        if len(imgs) > N:
            imgs = imgs[0:N]
        # calculate the image augmentation number
        m = len(imgs)
        k = N//m
        print("Generating ",label,": ",k*m,"(images)")
        for i in tqdm(range(m)):
            img = imgs[i]
            path = os.path.join(src,label,img)
            im = cv2.imread(path)
            im = preprocess(im)
            ims = np.array([im for _ in range(k)],dtype=np.uint8)
            imaugs = augmentation(images=ims)
            for j,aug in enumerate(imaugs):
                path = os.path.join(dst,typ,label,label+str(i)+str(j)+".jpg")
                cv2.imwrite(path,aug)