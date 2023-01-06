import cv2
import numpy as np
import random
import os

# parameter
scale=25
font = cv2.FONT_HERSHEY_SIMPLEX # font
org1 = (10, 50) # org
org2 = (10, 100) # org
fontScale = 0.8 # fontScale
color = (0, 255, 255) # Blue color in BGR
thickness = 2 # Line thickness of 2 px

class Classifier:
    def __init__(self):
        # sage HSV
        self.sage = [(np.array([35,0,167]),np.array([72,255,255]))]
        # sky HSV
        self.sky = [(np.array([79,19,0]),np.array([103,255,255]))]
        # sand HSV
        self.sand = [(np.array([0,0,184]),np.array([16,255,255]))]
        # charcoal HSV
        self.charcoal = [(np.array([0,0,0]),np.array([25,20,255]))]
        # chalk HSV
        self.chalk = [(np.array([63,0,70]),np.array([179,74,194])),
                      (np.array([5,0,219]),np.array([23,20,252]))
                      ]
        # HSV ranges
        self.HSV = [self.sage,self.sky,self.sand,self.charcoal,self.chalk]
        # HSV labels
        self.labels = ["sage","sky","sand","charcoal"]
        
    def predict(self,img):
        '''
        '''
        pred = "unknown"
        thresholds = [] # list of threshold
        
        # convert hsv color space
        for i in range(len(self.labels)):
            h,w,c = img.shape
            threshold = 0
            # convert to hsv
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            for j,m in enumerate(self.HSV[i]):
                # get mask
                mask_j = cv2.inRange(hsv,m[0],m[1]) #0,255
                # and with rgb image
                res = cv2.bitwise_and(img,img,mask=mask_j)
                threshold += round(np.sum(res)/np.sum(img),2)
            # append threshold
            thresholds.append(threshold)
        
        # predict
        idx = np.argmax(thresholds)
        pred = self.labels[idx]
        
        return pred,thresholds[idx]
    
def random_read(path):
    '''
    Random read image from dataset
    Args:
        path: path to your dataset
    Return:
        image_name: name of the image
    '''
    # choice label
    label = random.choice(os.listdir(path))
    # choice image of label
    image_name = random.choice(os.listdir(os.path.join(path,label)))
    return os.path.join(path,label,image_name)

if __name__ == "__main__":
    
    # read the first image
    image_name = random_read(path="gg")
    image_previous = image_name
    
    # create instance of classifier
    classifier = Classifier()
    
    # loop processing
    while True:
        
        # read the image
        img = cv2.imread(image_name)
        resized = cv2.resize(img,(0,0),fx=scale/100,fy=scale/100)
        
        # predict
        pred,thresh = classifier.predict(img)
        
        # put text
        label = image_name.split("/")[1]
        text1 = image_name
        text2 = "pred: " + pred + " threshold: " + str(thresh) + " " + str(pred==label)
        
        cv2.putText(img,text1, org1, font,fontScale, color, thickness, cv2.LINE_AA)
        cv2.putText(resized,text1, org1, font,fontScale, color, thickness, cv2.LINE_AA)
        cv2.putText(img,text2, org2, font,fontScale, color, thickness, cv2.LINE_AA)
        cv2.putText(resized,text2, org2, font,fontScale, color, thickness, cv2.LINE_AA)
        
        # display
        cv2.imshow("color",resized)
        
        k = cv2.waitKey(1) & 0xFF
        # save image
        if k == ord('s'):
            cv2.imwrite("hsv.jpg",resized)
        # quit
        elif k == ord('q'):
            break
        # next image
        elif k == ord('z'):
            image_previous = image_name
            image_name = random_read(path="gg")
        elif k == ord('x'):
        # back to previous image
            image_name = image_previous
        else:
            pass
    
    # destroy all window
    cv2.destroyAllWindows()
    