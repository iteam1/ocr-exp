'''
Color classification testing tool
Support:
    - label test
    - random test (multi label)
'''
import os
import cv2
import random
import argparse
import numpy as np
from tqdm import tqdm
from color_classifier import Classifier

# init parser
parser = argparse.ArgumentParser(description='batch test tool')

# add argument to parser
parser.add_argument('-d','--dir',type=str,help='path/to/dataset',required=True)
parser.add_argument('-r','--rand',action='store_true',help='random test option, if not label test mode')

# create arguments
args = parser.parse_args()

# create instance of classifier
classifier = Classifier()

def test_label(dir,classifier,label):
    '''
    Test with one label
    Args:
        - dir:
        - classifier:
        - label
    '''
    # metrics
    thresholds = [] # min,max,mean threshold
    count = 0 # false prediction counting
    content= "false predictions:\n"
    # label = random.choice(os.listdir(dir))
    samples = os.listdir(os.path.join(dir,label)) # get samples
    for sample in tqdm(samples):
        # read the image
        img = cv2.imread(os.path.join(dir,label,sample))
        # predict
        pred,thresh = classifier.predict(img)
        thresholds.append(thresh)
        if pred != label:
            count +=1
            content += os.path.join(args.dir,sample) + " pred: " + pred + " thresh: " + str(round(thresh,2)) +"\n"

    # threshold
    min_thresh = np.min(thresholds)
    max_thresh = np.max(thresholds)
    mean_thresh = np.mean(thresholds)

    # print out report
    report = f"accuracy: {round((len(samples)-count)/len(samples),2)} ({len(samples)-count}/{len(samples)})\nthreshold: min= {min_thresh} max={max_thresh} mean={mean_thresh}"
    print(report)

    # export report
    content = report + "\n" + content
    content = "label: " + label + "\n" +content
    with open("report.txt","w") as f:
        f.write(content)

def test_random(dir,classifier,nop):
    '''
    '''
    # metrics
    thresholds = [] # min,max,mean threshold
    count = 0 # false prediction counting
    content= "false predictions:\n"

    #get all labels
    labels = os.listdir(dir)
    for i in tqdm(range(nop)):
        label = random.choice(labels)
        imgs  = os.listdir(os.path.join(dir,label))
        f = random.choice(imgs)
        img = cv2.imread(os.path.join(dir,label,f))
        # predict
        pred,thresh = classifier.predict(img)
        if pred != label:
            count +=1
            content += os.path.join(args.dir,label,f) + " pred: " + pred + " thresh: " + str(round(thresh,2)) +"\n"

    # print out report
    report = f"accuracy: {round((nop-count)/nop,2)} ({nop-count}/{nop})"
    print(report)

    # export report
    content = report + "\n" + content
    with open("report.txt","w") as f:
        f.write(content)

if __name__ == "__main__":

    if args.rand:
        test_random(args.dir,classifier,nop=100)
    else:
        test_label(args.dir,classifier,'sage')
