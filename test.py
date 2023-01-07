import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from color_classifier import Classifier

# init parser
parser = argparse.ArgumentParser(description='batch test tool')

# add argument to parser
parser.add_argument('-d','--dir',type=str,help='path/to/label',required=True)

# create arguments
args = parser.parse_args()

# create instance of classifier
classifier = Classifier()

# label
label = args.dir.split('/')[-1]
f_preds = []

# min,max,mean threshold
thresholds = []
count = 0
content= "false predictions:\n"

if __name__ == "__main__":
    # get samples
    samples = os.listdir(args.dir)
    for sample in tqdm(samples):
        # read the image
        img = cv2.imread(os.path.join(args.dir,sample))
        # predict
        pred,thresh = classifier.predict(img)
        thresholds.append(thresh)
        if pred != label:
            count +=1
            f_preds.append(os.path.join(args.dir,sample))
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
    with open("report.txt","w") as f:
        f.write(content)
