'''
only copy class, or function in utils.py to preprocess.py
'''
import os
import cv2
import sys
import math
import json
import copy
import time
import random
import pickle
import paddle
import paddleocr
import numpy as np
import collections
from PIL import Image
from tqdm import tqdm
from circle_fit import taubinSVD
from pyzbar.pyzbar import decode
from pre_post_processing import *
from openvino.runtime import Core,Dimension
from sklearn.metrics import accuracy_score
from paddleocr import PaddleOCR

def boost_contrast(img):
    '''
    Boost constact of image
    Args:
        img(numpy array): input image
    Return:
        out: constact boosted image
    '''
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    out = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return out

def image_preprocess(input_image, size):
    '''
    Preprocessing image functions for text detection and recognition
    Args:
        input_image (numpy array): input image
        size: size of image
    Return:
        preprocessed image
    '''
    img = cv2.resize(input_image, (size,size))
    img = np.transpose(img, [2,0,1]) / 255
    img = np.expand_dims(img, 0)
    ##NormalizeImage: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], is_scale: True}
    img_mean = np.array([0.485, 0.456,0.406]).reshape((3,1,1))
    img_std = np.array([0.229, 0.224, 0.225]).reshape((3,1,1))
    img -= img_mean
    img /= img_std
    return img.astype(np.float32)

def resize_norm_img(img, max_wh_ratio):
    '''
    Preprocess for Paddle Recognition
    Args:
        img: preprocessed image
        max_wh_ratio: dimension ratio
    Return:
        resized normalized image
    '''
    no=48
    rec_image_shape = [3,no, 320] #[3, 32, 320]
    imgC, imgH, imgW = rec_image_shape
    assert imgC == img.shape[2]
    character_type = "ch"
    if character_type == "ch":
        imgW = int((no * max_wh_ratio))
    h, w = img.shape[:2]
    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im

def rotateAndScale(img, scaleFactor = 0.5, degreesCCW = 30):
    '''
    rotate image with no cut out vertices
    Args:
        img: original image
        scaleFactor: scale of output image
        degreesCCW: angle (degrees)
    Return:
        rotateImg: rotated image
        M: rotate matrix
    '''
    h,w,c = img.shape #note: numpy uses (y,x) convention but most OpenCV functions use (x,y)
    M = cv2.getRotationMatrix2D(center=(w/2,h/2), angle=degreesCCW, scale=scaleFactor) #rotate about center of image.

    #choose a new image size.
    new_w,new_h = w*scaleFactor,h*scaleFactor
    #include this if you want to prevent corners being cut off
    r = np.deg2rad(degreesCCW)
    new_w,new_h = (abs(np.sin(r)*new_h) + abs(np.cos(r)*new_w),abs(np.sin(r)*new_w) + abs(np.cos(r)*new_h))

    #the warpAffine function call, below, basically works like this:
    # 1. apply the M transformation on each pixel of the original image
    # 2. save everything that falls within the upper-left "dsize" portion of the resulting image.

    #So I will find the translation that moves the result to the center of that region.
    (tx,ty) = ((new_w-w)/2,(new_h-h)/2)
    M[0,2] += tx #third column of matrix holds translation, which takes effect after rotation.
    M[1,2] += ty

    rotatedImg = cv2.warpAffine(img, M, dsize=(int(new_w),int(new_h)))
    return rotatedImg,M

def calc_dist(pt1,pt2):
    '''
    Calculate distance of 2 point
    Args:
        pt1: first point
        pt2: second point
    Return:
        dist: distance
    '''
    dist = cv2.norm(pt1-pt2)
    return dist

def infer_circle(mask):
    '''
    Infer centerpoint, radius of curve
    Args:
        mask: mask of curve image (BGR)
    Return:
        xc: x cord of center point
        yc: y cord of center point
        r: radius
        sigma: sigma
    '''
    # convert image to grayscale
    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    h,w = mask.shape
    # get first point in mask
    curve = []
    for i in range(w):
        v_line = mask[:,i:i+1]
        pos = cv2.findNonZero(v_line)
        if pos is not None:
            curve.append((pos[0][0][1],i))
    # infer circle by taubinSVD
    xc,yc,r,sigma = taubinSVD(curve)
    return xc,yc,r,sigma

class CodeReader:

    def __init__(self):
        pass

    def read_code(self,img):
        '''
        Read barcode or qrcode
        Args:
            img: input image
        Return:
            d_list: list of decoded [(left,top,width,height),...]
        '''
        img_org = img.copy()
        # menthod1
        img = img_org.copy()
        img = cv2.GaussianBlur(img,(5,5),0)
        d_1 = decode(img)
        # menthod2
        img = img_org.copy()
        img = cv2.GaussianBlur(img,(5,5),0)
        img = boost_contrast(img)
        d_2 = decode(img)
        #combine
        d_list = d_1 + d_2
        return d_list
    
    def draw_code(self,img,d_list):
        '''
        Draw barcode or qrcode
        Args:
            img (numpy array): input image
            d_list (list) : list of decoded [(left,top,width,height),...]
        Return:
            img (numpy array): output image 
        '''
        for i,d in enumerate(d_list):
            img = cv2.rectangle(img,(d.rect.left,d.rect.top),(d.rect.left+d.rect.width,d.rect.top+d.rect.height),(225,0,0),2)
            img = cv2.polylines(img,[np.array(d.polygon)],True,(0,255,0),2)
            img = cv2.putText(img,str(i+1)+ "_" + d.data.decode(),(d.rect.left,d.rect.top),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),1,cv2.LINE_AA)
        return img
    
    def infer(self,img,aliases):
        '''
        Infer model and serial number of the image (if exist)
        Args:
            img (numpy array): input image
            aliases (dict): dictionary of device alias
        Return:
            model: model number
            serial: model number
        '''
        d_list = self.read_code(img)
        
        s = []
        a = []

        for de in d_list:
            t = de.type
            data = de.data.decode()
            if t == "QRCODE":
                if data[0]=="1":
                    data = data.split('$')
                else:
                    data = data.split(',')
                for d in data:
                    if d[:2]=="S:":
                        s.append(d[2:])
                    elif d[:2]=="I:":
                        if d[2:] in aliases.keys():
                            a.append(aliases[d[2:]])
                        else:
                            a.append(d[2:])
                    elif d[:2]=="C:":
                        if d[2:] in aliases.keys():
                            a.append(aliases[d[2:]])
                        else:
                            a.append(d[2:])
            elif t == "CODE128":
                if (len(data)==14 or len(data)==16) and all(c.isalnum() for c in data):
                    s.append(data)
                elif data[:3]=="SN:":
                    s.append(data[3:])
            else:
                continue
            
        # find serial by the most frequent
        serial = None
        if len(s) != 0:
            count = 0
            serial = s[0]
            for i in s:
                f = s.count(i)
                if f > count:
                    count = f
                    serial = i
                    
        # find model by the most frequent
        model = None
        if len(a) != 0:
            count = 0
            model = a[0]
            for i in a:
                f = a.count(i)
                if f > count:
                    count = f
                    model = i
                    
        return model,serial

class ObjectClassifier:

    def __init__(self):
        self.loaded_model = None
        self.IMG_SIZE = 100 #image input dimension

    def load_model(self,model_path):
        '''
        Load pretrained model (*.sav)
        Args:
            - model_path: path of pretrained model
        Return:
            Load model into class.loaded_model
        '''
        # load the model from disk
        self.loaded_model = pickle.load(open(model_path, 'rb'))
        print("Classifier loaded pretrain model!")
        
    def predict(self,img,dicts):
        '''
        Predict new income data
        Args:
            - img(str or numpy.array): path or numpy array of image
            - dicts (dict): dictionary of label-model
        Return:
            - pred(str): predicted label or None
        '''
        pred = None

        # check model loaded
        if not self.load_model:
            print("No load pretrained model!")
            return pred

        # check type of param img
        typ = str(type(img))
        if typ == "<class 'str'>":
            img = cv2.imread(img,0)
        elif typ == "<class 'numpy.ndarray'>":
            pass
        else:
            print(f'{type(img)} is not supported!')
            return pred
        
        # preprocess income data
        img = cv2.resize(img,(self.IMG_SIZE,self.IMG_SIZE))
        img = img/255.0
        img = img.reshape(1,-1)
        # predict
        pred = self.loaded_model.predict(img)
        pred = int(pred)
        # get model of prediction
        model = dicts[str(pred)]
        
        return model['model']
        
    def test(self,path):
        '''
        Test images,floder structure:
            ./path
                |_1
                |   |_img1.jpg
                |   |_img2.png
                |       ...
                |_2
                |   |_img1.jpg
                |   |_img2.png
                        ...
                    ...
        Args:
            - path: path to testing image floder
        Return:
            Print out metrics
            Export report.txt file
        '''
        content = "TEST OBJECT CLASSIFIER REPORT\n-------\n"
        preds = []
        ground_truth = []
        labels = os.listdir(path)
        for l in range(labels):
            imgs = os.listdir(os.path.join(path,label))
            print("testing label: ",label)
            for i in tqdm(imgs):
                i = os.path.join(path,label,i)
                img = cv2.imread(i,0)
                img = cv2.resize(img,(self.IMG_SIZE,self.IMG_SIZE))
                img = img/255.0
                img = img.reshape(1,-1)
                pred = int(self.loaded_model.predict(img))
                preds.append(pred) # collect pred
                ground_truth.append(int(label)) # collect ground truth
                # note wrong prediction
                if pred != label:
                    content+= "img: " + i + " label: " + label + " pred: " + pred + "\n"
        
        # export report
        with open('report.txt','w') as f:
            f.write(content)
                    
        # calc accuracy
        print("Accuracy valid on ",accuracy_score(ground_truth,preds))

class ColorClassifier:

    def __init__(self):
    	# aka ColorClassifierGGNestAudio
        self.loaded_model = None
        self.IMG_SIZE = 100 #image input dimension
        self.dict = {0:'sky',1:'charcoal',2:'sage',3:'chalk',4:'sand'} # dictionary mapping predict value to label

    def load_model(self,model_path):
        '''
        Load pretrained model (*.sav)
        Args:
            - model_path: path of pretrained model
        Return:
            Load model into class.loaded_model
        '''
        # load the model from disk
        self.loaded_model = pickle.load(open(model_path, 'rb'))
        print("Classifier loaded pretrain model!")

    def predict(self,img):
        '''
        Predict new income data
        Args:
            - img(str or numpy.array): path or numpy array of image
        Return:
            - pred(str): predicted label or None
        '''
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
        # preprocess income data
        img = cv2.resize(img,(self.IMG_SIZE,self.IMG_SIZE)) # resize
        img = img/255.0 # normalize
        img = img.reshape(1,-1) # reshape(N,D)
        # predict
        y = self.loaded_model.predict(img)
        return self.dict[int(y)]

    def test(self,path):
        '''
        Test images,floder structure:
            ./path
                |_label1
                |   |_img1.jpg
                |   |_img2.png
                |       ...
                |_label2
                |   |_img1.jpg
                |   |_img2.png
                        ...
                    ...
        Args:
            - path: path to testing image floder
        Return:
            Print out metrics
            Export report.txt file
        '''
        content = "TEST COLOR REPORT\n-------\n"
        
        # check model loaded
        if not self.load_model:
            print("No load pretrained model!")
            return

        preds = []
        ground_truth = []
        labels = os.listdir(path)
        for label in labels:
            imgs = os.listdir(os.path.join(path,label))
            print("testing label: ",label)
            for i in tqdm(imgs):
                i = os.path.join(path,label,i)
                pred = self.predict(i)
                preds.append(pred) # collect pred
                ground_truth.append(label) # collect ground truth
                if pred != label:
                    content+= "img: " + i + " label: " + label + " pred: " + pred + "\n" 

        # export report
        with open('report.txt','w') as f:
            f.write(content)
            
        # calc accuracy
        print("Accuracy valid on "+ path+ ": ",accuracy_score(ground_truth,preds))

class GGHomeMini:
    def __init__(self):
        # for recognition
        self.ppocr = PaddleOCR(det_model_dir='ppocr/ch_PP-OCRv3_det_infer',
                rec_model_dir='ppocr/ch_PP-OCRv3_rec_infer',
                cls_model_dir='ppocr/ch_ppocr_mobile_v2.0_cls_infer', 
                use_angle_cls=True)
        
        # for detection
        self.ocr = Extractor()
        self.ocr.load_model()
        
        self.K = 41 # dilate kernel size
        self.model = 'H2C' # specific model number of GGHomeMini
        self.serial_no = 14 # specific serial number of GGHomeMini
        
    def extract(self,img):
        '''
        Extract ocr from specific device Google Home Mini
        Args:
            img (numpy array): input image
        Return:
            rec_res: recognized result
        '''
        # check type of param img
        typ = str(type(img))
        if typ == "<class 'str'>":
            img = cv2.imread(img)
        elif typ == "<class 'numpy.ndarray'>":
            pass
        
        mask = self.ocr.get_mask(img)
        #dilate
        kernel = np.ones((self.K,self.K),np.uint8)
        dilate = cv2.dilate(mask.astype('uint8')*255,kernel,iterations=1)
        img = cv2.resize(img,(mask.shape[1],mask.shape[0]))
        
        # find best curve
        w,h = dilate.shape
        h_lines = []
        c = []
        for i in range(h):
            h_line = dilate[i:i+1,:]
            pos = cv2.findNonZero(h_line) #x,y
            if pos is not None:
                c.append((pos[0][0][0],i))
            h_lines.append(h_line)           
            
        C= []
        c_sub = []
        p0 = c[0]
        DMAX = 100
        for i in range(len(c)):
            p = c[i]
            c_sub.append(p)
            d = cv2.norm(np.array(p),np.array(p0))
            if d > DMAX:
                C.append(c_sub)
                c_sub = []
                # print(p0)
            p0 = p
            
        C.append(c_sub)
        
        L = [len(i) for i in C]
        
        c_sub = C[np.argmax(L)] # the best curve
        
        # fit in circle
        xc, yc, r, sigma = taubinSVD(c_sub)
        r = int(r) +50
        
        flags = cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LINEAR
        polar_img = cv2.warpPolar(img,(0,0),(int(xc),int(yc)),r,flags)
        polar_img = polar_img.transpose(1,0,2)[::-1]
        
        rec_res = self.ppocr.ocr(polar_img,det=True,rec=True,cls=True)

        return rec_res
    
    def infer(self,res):
        '''
        Infer serial & model number in ocr's result
        Args:
            res: result of ocr
        Return:
            model: model number
            serial: serial number
        '''
        model = None
        serial = None
        
        for r in res[0]:
            t = r[1][0].lower()
            #print(t)
            # infer model
            count = t.count(self.model.lower())
            if count > 2:
                model = self.model
            # infer serial 14 digit
            if len(t)>self.serial_no:
                for i in range(len(t)):
                    s = t[i:i+self.serial_no]
                    if len(s) == self.serial_no\
                        and all([c.isalnum() for c in s])\
                        and all([c.isnumeric() for c in s[:4]]):
                        #print('serial: ',s)
                        serial = s
                        
        if model:
            model = model.upper()
        if serial:
            serial = serial.upper()
            
        return model,serial
    
    def run(self,img):
        '''
        Run extract and infer
        Args:
            img: input image
        Return:
            model: model number
            serial: serial number
        '''
        res = self.extract(img)
        model,serial = self.infer(res)
        return model,serial
    
class Extractor2:

    def __init__(self):
        self.res=1280 # the resolution range 1280,2560,5120,... TB warnning
        self.det_compiled_model = None
        self.rec_compiled_model = None
        self.det_input_layer = None # input and nodes for text detection
        self.det_output_layer = None # output nodes for text detection
        self.rec_input_layer = None # input and nodes for text recognition
        self.rec_output_layer = None # output nodes for text recognition
        self.PADDING = 60 # roi padding
        self.BOX_NUM = 40 # maximum box number
        self.OFFSET = 30
        self.FLAGS = cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LINEAR
        self.R_MAX = 3000
        self.MIN_POINTS = 300
        self.R_OFFSET=250
        self.Y_OFFSET=20
        self.SHRINK = 50

    def load_model(self,det_model_dir="ppocr/ch_PP-OCRv3_det_infer",rec_model_dir="ppocr/ch_PP-OCRv3_rec_infer"):
        '''
        Load detection and recognition model
        Args:
            det_model_dir: detection inference model (pretrained)
            rec_model_dir: recognition inference model (pretrained)
        Return:
        Load model into self.det_compiled_model,self.rec_compiled_model
        '''
        #Load Network text detection
        det_model_file_path = det_model_dir + "/inference.pdmodel"
        det_params_file_path = det_model_dir + "/inference.pdiparams"

        # initialize inference engine for text detection
        det_ie = Core()
        det_net = det_ie.read_model(model=det_model_file_path, weights=det_params_file_path)
        self.det_compiled_model = det_ie.compile_model(model=det_net, device_name="CPU")

        self.det_input_layer = next(iter(self.det_compiled_model.inputs)) ## get input and output nodes for text detection
        self.det_output_layer = next(iter(self.det_compiled_model.outputs))

        #Load the Network for Text Recognition
        rec_model_file_path = rec_model_dir + "/inference.pdmodel"
        rec_params_file_path = rec_model_dir + "/inference.pdiparams"

        # Initialize the Paddle recognition inference on CPU
        rec_ie = Core()
        # read the model and corresponding weights from file
        rec_net = rec_ie.read_model(model=rec_model_file_path, weights=rec_params_file_path)

        # assign dynamic shapes to every input layer on the last dimension
        for input_layer in rec_net.inputs:
            input_shape = input_layer.partial_shape
            input_shape[3] = Dimension(-1)
            rec_net.reshape({input_layer: input_shape})
        self.rec_compiled_model = rec_ie.compile_model(model=rec_net, device_name="CPU")

        # get input and output nodes
        self.rec_input_layer = next(iter(self.rec_compiled_model.inputs))
        self.rec_output_layer = next(iter(self.rec_compiled_model.outputs))

    def read_image(self,img_path):
        '''
        Read the image
        Args:
            img_path: image path
        Return:
            image_file: scaled image
            test_image: preprocessed array of image
        '''
        # read the image
        image = cv2.imread(img_path)
        res= self.res

        # if frame larger than full HD, reduce size to improve the performance
        scale = res/max(image.shape)

        if scale < 1:
            # resize image
            image = cv2.resize(src=image, dsize=None, fx=scale, fy=scale,interpolation=cv2.INTER_AREA)

        image_file = image
        test_image = image_preprocess(image_file,self.res)
        return image_file,test_image

    def detect(self,image_file,test_image):
        '''
        Extract roi (text regions)
        Args:
            image_file:
            test_image:
        Return:
            dt_boxes: list of detected boxes,example: [(4,2),(4,2),...]
            mask: binary mask of image
        '''
        # Create dectect infer request
        det_request = self.det_compiled_model.create_infer_request()

        #perform the inference step
        det_request.infer(inputs = {self.det_input_layer.any_name: test_image})
        det_results = det_request.get_tensor(self.det_output_layer).data

        # Postprocessing for Paddle Detection
        ori_im = image_file.copy()
        data = {'image': image_file}
        data_resize = DetResizeForTest(data)
        data_norm = NormalizeImage(data_resize)
        data_list = []
        keep_keys =  ['image', 'shape']

        for key in keep_keys:
            data_list.append(data_resize[key]) #
        img, shape_list = data_list
        shape_list = np.expand_dims(shape_list, axis=0)
        pred = det_results[0]

        if isinstance(pred, paddle.Tensor):
            pred = pred.numpy()
        segmentation = pred > 0.3
        boxes_batch = []

        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            mask = segmentation[batch_index]
            boxes, scores = boxes_from_bitmap(pred[batch_index], mask,src_w, src_h) # boxes (n, 4, 2), scores list of coff
            boxes_batch.append({'points': boxes})
        post_result = boxes_batch # list of {'points':[[]]}
        dt_boxes = post_result[0]['points'] # (n,4,2)
        dt_boxes = filter_tag_det_res(dt_boxes, ori_im.shape)

        #Preprocess detection results for recognition
        dt_boxes = sorted_boxes(dt_boxes)
        return dt_boxes,mask

    def recognize(self,img_crop_list):
        '''
        Recognize text in list of cropped image
        Args:
            img_crop_list: list of cropped roi
        Return:
            rec_res: list of text recognized and conffident score
        '''
        #Recognition starts from here
        img_num = len(img_crop_list)

        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_crop_list:
            width_list.append(img.shape[1] / float(img.shape[0]))

        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))
        rec_res = [['', 0.0]] * img_num
        batch_num = 2
        rec_processing_times = 0

        #For each detected text box, run inference for text recognition
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)

            norm_img_batch = []
            max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                h, w = img_crop_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = resize_norm_img(img_crop_list[indices[ino]],max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)

            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()

            #Run inference for text recognition
            #rec_results = self.rec_compiled_model([norm_img_batch])[self.rec_compiled_model.output(0)]
            rec_request = self.rec_compiled_model.create_infer_request()
            rec_request.infer(inputs={self.rec_input_layer.any_name: norm_img_batch})
            rec_results = rec_request.get_tensor(self.rec_output_layer).data
            preds = rec_results #(6, 47, 6625)

            #Postprocessing recognition results
            postprocess_op = build_post_process(postprocess_params)
            rec_result = postprocess_op(preds)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]
        return rec_res

    def extract(self,img):
        '''
        Inference text detection and tex recognition
        Args:
            img_path: image or image path
        Return:
        '''
        # check type of param img
        typ = str(type(img))
        if typ == "<class 'str'>":
            img = cv2.imread(img)
        elif typ == "<class 'numpy.ndarray'>":
            pass
        
        # Read the image
        image_file,test_image=self.read_image(img)
        ori_im = image_file.copy()

        # text detection
        dt_boxes,_ = self.detect(image_file,test_image)

        img_crop_list = []
        rec_res=None
        draw_img=None
        if dt_boxes != []:
            for bno in range(len(dt_boxes)):
                tmp_box = copy.deepcopy(dt_boxes[bno])
                img_crop = get_rotate_crop_image(ori_im, tmp_box)
                img_crop_list.append(img_crop)

            #Recognition starts from here
            img_num = len(img_crop_list)

            # Calculate the aspect ratio of all text bars
            width_list = []
            for img in img_crop_list:
                width_list.append(img.shape[1] / float(img.shape[0]))

            # Sorting can speed up the recognition process
            indices = np.argsort(np.array(width_list))
            rec_res = [['', 0.0]] * img_num
            batch_num = 6

            #For each detected text box, run inference for text recognition
            for beg_img_no in range(0, img_num, batch_num):
                end_img_no = min(img_num, beg_img_no + batch_num)

                norm_img_batch = []
                max_wh_ratio = 0
                for ino in range(beg_img_no, end_img_no):
                    h, w = img_crop_list[indices[ino]].shape[0:2]
                    wh_ratio = w * 1.0 / h
                    max_wh_ratio = max(max_wh_ratio, wh_ratio)
                for ino in range(beg_img_no, end_img_no):
                    norm_img = resize_norm_img(img_crop_list[indices[ino]],max_wh_ratio)
                    norm_img = norm_img[np.newaxis, :]
                    norm_img_batch.append(norm_img)

                norm_img_batch = np.concatenate(norm_img_batch)
                norm_img_batch = norm_img_batch.copy()

                #Run inference for text recognition
                rec_request = self.rec_compiled_model.create_infer_request()
                rec_request.infer(inputs={self.rec_input_layer.any_name: norm_img_batch})
                rec_results = rec_request.get_tensor(self.rec_output_layer).data
                preds = rec_results

                #Postprocessing recognition results
                postprocess_op = build_post_process(postprocess_params)
                rec_result = postprocess_op(preds)
                for rno in range(len(rec_result)):
                    rec_res[indices[beg_img_no + rno]] = rec_result[rno]

                #Text recognition results, rec_res, include two parts:
                #txts are the recognized text results, scores are the recognition confidence level
                if rec_res != []:
                    img = Image.fromarray(cv2.cvtColor(image_file, cv2.COLOR_BGR2RGB))
                    boxes = dt_boxes
                    txts = [rec_res[i][0] for i in range(len(rec_res))]
                    scores = [rec_res[i][1] for i in range(len(rec_res))]

                    #draw text recognition results beside the image
                    # remove pred samller than drop_score
                    draw_img = draw_ocr_box_txt(img,boxes,txts,scores,drop_score=0.2)

        return rec_res,draw_img
    
    def infer(self,rec_res):
        '''
        Infer serial & model number in ocr's result
        Args:
            res: result of ocr
        Return:
            serial: serial number
            model: model number
        '''
        pass
    
    def run(self,img):
        '''
        Run extract and infer
        Args:
            img: input image
        Return:
            model: model number
            serial: serial number
        '''
        # check type of param img
        typ = str(type(img))
        if typ == "<class 'str'>":
            img = cv2.imread(img)
        elif typ == "<class 'numpy.ndarray'>":
            pass
        pass

#
def check_empty_img(image):
    # Checking if the image is empty or not
    if image is None:
        # result = "Image is empty!!"
        result = True
    else:
        # result = "Image is not empty!!"
        result = False
    return result

def image_preprocess(input_image, size):
    """
    Preprocess input image for text detection

    Parameters:
        input_image: input image 
        size: value for the image to be resized for text detection model
    Return:
        img: image preprocessed
    """
    img = cv2.resize(input_image, (size, 3104))
    img = np.transpose(img, [2, 0, 1]) / 255
    img = np.expand_dims(img, 0)
    # NormalizeImage: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], is_scale: True}
    img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std
    return img.astype(np.float32)

def resize_norm_img(img, max_wh_ratio):
    """
    Resize input image for text recognition

    Parameters:
        img: bounding box image from text detection 
        max_wh_ratio: value for the resizing for text recognition model
    """
    rec_image_shape = [3, 48, 320]
    imgC, imgH, imgW = rec_image_shape
    assert imgC == img.shape[2]
    character_type = "en"
    if character_type == "en":
        imgW = int((32 * max_wh_ratio))
    h, w = img.shape[:2]
    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im

def prep_for_rec(dt_boxes, frame):
    """
    Preprocessing of the detected bounding boxes for text recognition

    Parameters:
        dt_boxes: detected bounding boxes from text detection 
        frame: original input frame 
    """
    ori_im = frame.copy()
    img_crop_list = [] 
    for bno in range(len(dt_boxes)):
        tmp_box = copy.deepcopy(dt_boxes[bno])
        img_crop = get_rotate_crop_image(ori_im, tmp_box)
        img_crop_list.append(img_crop)
        
    img_num = len(img_crop_list)
    # Calculate the aspect ratio of all text bars.
    width_list = []
    for img in img_crop_list:
        width_list.append(img.shape[1] / float(img.shape[0]))
    
    # Sorting can speed up the recognition process.
    indices = np.argsort(np.array(width_list))
    return img_crop_list, img_num, indices

def batch_text_box(img_crop_list, img_num, indices, beg_img_no, batch_num):
    """
    Batch for text recognition

    Parameters:
        img_crop_list: processed detected bounding box images 
        img_num: number of bounding boxes from text detection
        indices: sorting for bounding boxes to speed up text recognition
        beg_img_no: the beginning number of bounding boxes for each batch of text recognition inference
        batch_num: number of images for each batch
    """
    norm_img_batch = []
    max_wh_ratio = 0
    end_img_no = min(img_num, beg_img_no + batch_num)
    for ino in range(beg_img_no, end_img_no):
        h, w = img_crop_list[indices[ino]].shape[0:2]
        wh_ratio = w * 1.0 / h
        max_wh_ratio = max(max_wh_ratio, wh_ratio)
    for ino in range(beg_img_no, end_img_no):
        norm_img = resize_norm_img(img_crop_list[indices[ino]], max_wh_ratio)
        norm_img = norm_img[np.newaxis, :]
        norm_img_batch.append(norm_img)

    norm_img_batch = np.concatenate(norm_img_batch)
    norm_img_batch = norm_img_batch.copy()
    return norm_img_batch

def post_processing_detection(frame, det_results):
    """
    Postprocess the results from text detection into bounding boxes

    Parameters:
        frame: input image 
        det_results: inference results from text detection model
    Return:
        dt_boxes: the results from text detection into bounding boxes
        mask: mask of current image
    """
    mask_list = []
    ori_im = frame.copy()
    data = {'image': frame}
    data_resize = DetResizeForTest(data)
    data_list = []
    keep_keys = ['image', 'shape']
    for key in keep_keys:
        data_list.append(data_resize[key])
    img, shape_list = data_list

    shape_list = np.expand_dims(shape_list, axis=0) 
    pred = det_results[0]    
    if isinstance(pred, paddle.Tensor):
        pred = pred.numpy()
    segmentation = pred > 0.5

    boxes_batch = []
    for batch_index in range(pred.shape[0]):
        src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
        mask = segmentation[batch_index]
        boxes, scores = boxes_from_bitmap(pred[batch_index], mask, src_w, src_h)
        boxes_batch.append({'points': boxes})
    post_result = boxes_batch
    dt_boxes = post_result[0]['points']
    dt_boxes = filter_tag_det_res(dt_boxes, ori_im.shape)    
    return dt_boxes,mask

def write_result_to_json_file(json_path, boxes, txts, scores, image_path,result_list):
    json_result = {}
    # name_image = os.path.basename(image_path)
    name_image = image_path
    #print("name image:", name_image)
    json_result['name'] = name_image #json_result[name_image] = []
    json_result['results'] = []
    assert len(boxes) == len(txts) and len(txts) == len(scores)
    for i in range(len(txts)):
        temp = {}
        temp["text"] = txts[i]
        # fix NaN return
        if float(scores[i]) is None:
            temp["score"] = float(scores[i])
        else:
            temp["score"] = 0.0
        box = boxes[i]
        rRect = []
        for point in box:
            if(len(point) == 2): rRect.append([int(point[0]), int(point[1])])
        temp["localization"] = rRect
        json_result['results'].append(temp)
        
    # append current json to my_json
    result_list.append(json_result)
    
    # write down current json
    # with open(json_path, 'w') as outfile:
    #     json.dump(json_result, outfile)
    #     print(("Masks json path saved to: ", json_path))
        
    return result_list

class Extractor:

    def __init__(self):
        self.input = "models/detected.txt" # path to input image
        self.output = "models/output.json" # path to output image
        self.det_model_file_path = "models/ch_PP-OCRv3_det_infer/inference.pdmodel" # path to det_model_file_path
        self.rec_model_file_path = "models/ch_PP-OCRv3_rec_infer/inference.pdmodel" # path to rec_model_file_path
        self.det_compiled_model = None # det model loaded
        self.rec_compiled_model = None # rec model loaded
        
    def load_model(self):
        '''
        Return Openvino optimized detect_model and recognize_model
        '''
        # Initialize OpenVINO Runtime for text detection.
        core = Core()
        det_model = core.read_model(model=self.det_model_file_path)
        self.det_compiled_model = core.compile_model(model=det_model, device_name="CPU")

        # Get input and output nodes for text detection.
        det_input_layer = self.det_compiled_model.input(0)
        det_output_layer = self.det_compiled_model.output(0)

        # Read the model and corresponding weights from a file.
        rec_model = core.read_model(model=self.rec_model_file_path)

        # Assign dynamic shapes to every input layer on the last dimension.
        for input_layer in rec_model.inputs:
            input_shape = input_layer.partial_shape
            input_shape[3] = -1
            rec_model.reshape({input_layer: input_shape})

        self.rec_compiled_model = core.compile_model(model=rec_model, device_name="CPU")

        # Get input and output nodes.
        rec_input_layer = self.rec_compiled_model.input(0)
        rec_output_layer = self.rec_compiled_model.output(0)
        
    def get_mask(self,image):
        '''
        Get binary mask output from ocr paddle of image
        Args:
            img: input image (BGR)
        Return:
            mask: binary mask of image
        '''
        test_image = image_preprocess(image, 4192)
        det_results = self.det_compiled_model([test_image])[self.det_compiled_model.output(0)]  # (1, 1, 3104, 4192)
        dt_boxes,mask = post_processing_detection(image, det_results)
        mask = mask.astype('int')*255
        
        return mask
        
    def extract(self,rois):
        '''
        Run extract
        Args:
            rois: the list of detected roi from detector
        Return:
            result_list: list of the ocr result extracted
        '''
        print("Extractor executing..")

        result_list = []
        
        processing_times = collections.deque()
       
        count = 0
        for k in range(len(rois)):
            
            start_time = time.time()
            
            image = rois[k] # read the input image

            if check_empty_img(image) == True:
                print("The image is empty")
                return

            test_image = image_preprocess(image, 4192)
            preprocess_time = time.time()
            det_results = self.det_compiled_model([test_image])[self.det_compiled_model.output(0)]  # (1, 1, 3104, 4192)
            dt_boxes,mask = post_processing_detection(image, det_results)
            mask = mask.astype('int')*255            
            dt_boxes = sorted_boxes(dt_boxes)
            batch_num = 6
            img_crop_list, img_num, indices = prep_for_rec(dt_boxes, image)
            
            # For storing recognition results, include two parts:
            # txts are the recognized text results, scores are the recognition confidence level. 
            rec_res = [['', 0.0]] * img_num
            txts = [] 
            scores = []
            
            for beg_img_no in range(0, img_num, batch_num):
                # Recognition starts from here.
                norm_img_batch = batch_text_box(img_crop_list, img_num, indices, beg_img_no, batch_num)
                # Run inference for text recognition. 
                rec_results = self.rec_compiled_model([norm_img_batch])[self.rec_compiled_model.output(0)]
                # Postprocessing recognition results.
                postprocess_op = build_post_process(postprocess_params)
                rec_result = postprocess_op(rec_results)
                for rno in range(len(rec_result)):
                    rec_res[indices[beg_img_no + rno]] = rec_result[rno]   
                if rec_res:
                    txts = [rec_res[i][0] for i in range(len(rec_res))] 
                    scores = [rec_res[i][1] for i in range(len(rec_res))]
            stop_time = time.time()
            
            processing_times.append(stop_time - start_time)
            # Use processing times from last 200 frames.
            if len(processing_times) > 200:
                processing_times.popleft()
            processing_time_det = np.mean(processing_times) * 1000

            image_result = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            boxes = dt_boxes
            draw_img = draw_ocr_box_txt(image_result,boxes,txts,scores,drop_score=0.5)
            write_path = self.output
            result_list = write_result_to_json_file(write_path, boxes, txts, scores,"img",result_list)
            
            f_height, f_width = draw_img.shape[:2]
            fps = 1000 / processing_time_det
            text = "Inference time: {:.1f}ms ({:.1f} FPS)".format(processing_time_det, fps)
            cv2.putText(img=draw_img, text=text, org=(120, 120),fontFace=cv2.FONT_HERSHEY_COMPLEX, 
                        fontScale=f_width / 1500, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
            write_path = self.output.replace(".json", "_{}.jpg".format(count))
            count += 1
            
        return result_list
    
    def infer(self,result_list):
        '''
        Infer serial & model number in ocr's result
        Args:
            res: result of ocr
        Return:
            serial: serial number
            model: model number
        '''
        pass
    
    def run(self,img):
        '''
        Run extract and infer
        Args:
            img: input image
        Return:
            model: model number
            serial: serial number
        '''
        # check type of param img
        typ = str(type(img))
        if typ == "<class 'str'>":
            img = cv2.imread(img)
        elif typ == "<class 'numpy.ndarray'>":
            pass
        pass
