'''
python3 infer.py sn/image_14092023062751_6.jpg
'''
import os
import sys
import cv2
import time
import pickle
import numpy as np
from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
    export_extra_results,
    empty_cuda_cache
)

#init
dst = "dst"
image_path = sys.argv[1]
model_path = 'model1/models/model_7134462.sav'
cuda_opt = False
string = ''
kernel = np.ones((10,7), np.uint8)

def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, 0, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes

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
    
class Inference:
    
    def __init__(self):
        self.loaded_model = None
        self.dict = {0: "1", 1: "O", 2: "Q", 3: "0", 4: "I"}
        self.dict_invert = {v:k for k,v in self.dict.items()}
    
    def load_model(self, model_path):
        '''
        Load pretrained model (*.sav)
        Args:
            - model_path: path of pretrained model
        Return:
            Load model into class.loaded_model
        '''
        # load the model from disk
        self.loaded_model = pickle.load(open(model_path, 'rb'))
        print("Validator loaded pretrain model!")
        
    def predict(self, img):
        '''
        Predict new income data
        Args:
            - img(str or numpy.array): path or numpy array of image
        Return:
            - pred(str): predicted label or None
        '''
        DIM = 224
        pred = None
        
        # check model loaded
        if not self.load_model:
            print("No load pretrained model!")
            
            return pred
    
        # check type of param img
        typ = str(type(img))
        
        if typ == "<class 'str'>":
            img = cv2.imread(img)
        
        elif typ == "<class 'numpy.ndarray'>":
            pass
        
        else:
            print(f'{type(img)} is not supported!')
            
            return pred
        
        # preprocess income dat
        img = img/255.0
        resized = cv2.resize(img,(DIM,DIM), interpolation=cv2.INTER_AREA)
        resized = resized.reshape(1,-1)
        
        # predict
        y = self.loaded_model.predict(resized)
        probs = self.loaded_model.predict_proba(resized)
        prob_argmax = np.argmax(probs)
        coef = probs[0][int(y)]
        
        return self.dict[int(y)], coef

if __name__ == "__main__":
    
    print("CRAFT crop characters")
    
    # create destination folder
    if not os.path.exists(dst):
        os.mkdir(dst)
        
    image_name = image_path.split('/')[-1]
    image_name = image_name.split('.')[0]
    
    # load models
    refine_net = load_refinenet_model(cuda=cuda_opt)
    craft_net = load_craftnet_model(cuda=cuda_opt)
    predictor = Inference()
    predictor.load_model(model_path)
    
    print("Load models done")
    print("Crop image: ",image_path)
    
    start_time = time.time()
    img_resize, bboxes,_ = crop_char(image_path,refine_net,craft_net)
    img_debug = img_resize.copy()
    
    for j,bbox in enumerate(bboxes):
        x_min,y_min,x_max,y_max = bbox
        x_center = int((x_min+x_max)/2)
        y_center = int((y_min+y_max)/2)
        c = img_resize[y_min:y_max,x_min:x_max]
        pred, coef = predictor.predict(c)
        if coef > 0.6:
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