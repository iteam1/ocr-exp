import os
import cv2
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC

def crop_roi(img):
    '''
    crop img
    Args:
        img: source image
    Return:
        roi: cropped image
    '''
    DIM = 50
    OFFSET = 250
    # crop roi
    h,w,c = img.shape
    center = (int(h/2)+OFFSET,int(w/2)+OFFSET)
    top_left = (center[0]-DIM,center[1]-DIM)
    bottom_right = (center[0]+DIM,center[1]+DIM)
    roi = img[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1]]
    return roi

# load saved model
filename = '/home/greystone/GG_Non_Phone/ML_MODULE/ml_models/classification_models/color_classification_for_gg_nest_audio.sav'
loaded_model = pickle.load(open(filename,'rb'))

path = 'debug/DataFromHailu'

dic = {0: 'sage', 1: 'sand', 2: 'charcoal', 3: 'sky', 4: 'chalk'}

imgs = os.listdir(path)
print(imgs)

if __name__ =="__main__":
    for i in imgs:
        img = cv2.imread(os.path.join(path,i))
        img = img/255.0
        # predict
        roi = crop_roi(img)
        x = roi.reshape(1,-1)
        y= loaded_model.predict(x)
        print("img: " + i + " pred: " + dic[int(y)])