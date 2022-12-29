'''
Usage: content preprocessing function and classes
Used by:
    - Itself (class detector, class reporter)
    - main.py
    - detector.py
    - reporter.py
'''
# DETECTOR and REPORTER packages
import os
import cv2
import json
import numpy as np
from pyzbar.pyzbar import decode
# PADDLE-OCR packages
import paddle
import math
import time
import collections
from PIL import Image
from openvino.runtime import Core
import copy
import pre.pre_post_processing as processing

# DETECTOR and REPORTER utils

def scale_img(img,factor:float):
    '''
    scale the image
    Args:
        img: the image
        factor: scale value (0-1),type=float
    Return:
        img_scaled: scaled_img
    '''
    # check the value of factor
    img_scaled = cv2.resize(img,None,fx=factor,fy=factor)
    return img_scaled

def center_point(img,length,draw=False):
    '''
    Find the center point of image
    Args:
        img: source image
        draw: draw center option
        length: edge length of 4 corner cube
    Return:
        img: drawed (optional)
        center_pt: center point(x,y)
        corners: tuple of image four corner (topleft,topright,bottomleft,bottomright)
    '''
    h,w,c = img.shape
    center_x = int(w/2)
    center_y = int(h/2)
    topleft = img[0:length,0:0+length] #(y,x)
    topright = img[0:0+length,w-length:w] #(y,x)
    bottomleft = img[h-length:h,0:length] # (y,x)
    bottomright = img[h-length:h,w-length:w] # (y,x)
    if draw:
        # draw center point
        img = cv2.circle(img, (center_x,center_y), radius=30, color=(255, 0, 0), thickness=-1)
        # draw top left rectangle (img, start_point, end_point(x,y), color, thickness)
        img = cv2.rectangle(img,(0,0), (length,length), (255,0,0), 10) # (y,x)
        # top right rectangle
        img = cv2.rectangle(img,(w-length,0), (w,length), (255,0,0), 10) # (x,y)
        # bottom left rectangle
        img = cv2.rectangle(img,(0,h-length), (length,h), (255,0,0), 10) # (x,y) 
        # bottom right rectangle
        img = cv2.rectangle(img,(w-length,h-length), (w,h), (255,0,0), 10) # (x,y)
        # cv2.imshow("corner",bottomright)
    return img,(center_x,center_y),(topleft,topright,bottomleft,bottomright)

def back_project(img,roi):
    '''
    the object histogram should be normalized before passing on to the backproject function.
    It returns the probability image. Then we convolve the image with a disc kernel and apply threshold
    Args:
        img: source image
        roi: roi image
    Return:
        res: image remain the region have color similar to your roi
        thresh: binary threshold mask (0,1) of res
    '''
    # covert colorspace
    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hsv_roi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    #calculating object histogram
    roihist = cv2.calcHist([hsv_roi],[0,1],None,[180,256],[0,180,0,256])
    #normalize histogram and apply backprojection
    cv2.normalize(roihist,0,255,cv2.NORM_MINMAX)
    dst = cv2.calcBackProject([hsv_img],[0,1],roihist,[0,180,0,256],1)
    # now convolute with circular disc
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    cv2.filter2D(dst,-1,disc,dst)
    # threshold and binary AND
    ret,thresh = cv2.threshold(dst,240,255,0)
    # thresh =cv2.bitwise_not(thresh)
    thresh_color = cv2.merge((thresh,thresh,thresh))
    res = cv2.bitwise_and(img,thresh_color)
    return res,thresh

def detect_contour(img,thresh,s_thresh,padding):
    '''
    This function use for detect contour in image
    Args:
        img: source image
        thresh: binary theshold mask (output of back_project function)
        s_thresh: value threshold of the square area for counting as a R.O.I (region of interesting)
        padding: padding value of image (pixel)
    Return:
        img: drawed contour image
    Notes:
        cv2.RETR_TREE: It simply retrieves all the contours, but doesn't create any parent-child relationship
        cv2.RETR_EXTERNAL: It returns only extreme outer flags. All child contours are left behind.
        cv2.RETR_CCOMP: This flag retrieves all the contours and arranges them to a 2-level hierarchy
    '''
    contours,hierachy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print(f"Total contour detected: {len(contours)}") speed up by not running
    S = thresh.shape[0] * thresh.shape[1]
    # define the biggest ROI
    s_max = x_max = y_max = h_max = w_max = 0
    for cnt in contours:
        M = cv2.moments(cnt) # calculate the moment of contour
        if M['m00'] != 0:
            # finding centerpoint
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            perimeter = cv2.arcLength(cnt,True) # calcuate the perimeter of the contour
            # straight bounding rectangle
            x,y,w,h = cv2.boundingRect(cnt) # cnt
            # calculate area
            s = w*h 
            if s > s_thresh * S:
                # update the biggest roi
                if s > s_max:
                    s_max = s
                    x_max = x
                    y_max = y
                    h_max = h
                    w_max = w
                # cv2.rectangle(img,(x-padding,y-padding),(x+w+padding,y+h+padding),(0,0,255),6)
        else:
            pass
    return img,(x_max,y_max,w_max,h_max)

def check_limit(img,x,y,w,h,padding):
    '''
    Check the roi add padding exceed image size or not, if is this, reset it to limited value
    Args:
        img: source image
        x: x position of roi
        y: y position of roi
        w: width of roi
        h: height of roi
    Return:
        (topleft,bottomright): topleft bottomright point relimited
    '''
    x_topleft = y_topleft = x_bottomright = y_bottomright = 0
    if (x-padding) < 0:
        x_topleft = 0
    else: x_topleft = x - padding
    if (y-padding) < 0:
        y_topleft = 0
    else: y_topleft = y - padding
    if (x+w+padding) > img.shape[1]:
        x_bottomright = img.shape[1]
    else:
        x_bottomright = x+w+padding
    if (y+h+padding) > img.shape[0]:
        y_bottomright = img.shape[0]
    else:
        y_bottomright = y+h+padding
    return (x_topleft,y_topleft),(x_bottomright,y_bottomright)
    
def blur_img(img,method="gaussian"):
    '''
    Image processing apply blur method
    Args:
        img: source image
        method: type of blur method (default=gaussian)
            - gaussian blur
            - average blur 1/(k*k)
    '''
    if method == 'gaussian':
        img = cv2.GaussianBlur(img,(7,7),0)
    elif method == 'avarage':
        img = cv2.blur(img,(5,5))
    elif method == 'median':
        img = cv2.medianBlur(img,5)
    else:
        pass
    return img

def parse_text(ocr):
    '''
    Find serial and model
    Args:
        ocr: list of ocr sesult in current image
    Return:
        serial_no: (string) serial_no
        serial_pts:(array) point array
        serial_loc: location of serial number
        model_no: (string) model_no
        model_pts: (array) point array
        model_loc: location of model number
    '''
    serial_no = "None"
    model_no = "None"
    serial_pts = None # points of serial number
    model_pts = None # points of model number
    serial_loc = None
    model_loc = None
    
    # create text list
    text = []
    text_org = []
    score_m = []
    score_s = []
    pts_list = []
    loc_list = []
    
    for i,t in enumerate(ocr['results']):
        
        # get polygon point
        pts = np.array(t['localization']) # polygon points array
        pts = pts.reshape((-1, 1, 2)) # topleft point position
        pts_list.append(pts)
        # get location
        loc = t['localization'][0]
        loc_list.append(loc)
        # append current text
        txt_org = t['text'].strip()
        text_org.append(txt_org)
        txt = txt_org.lower() # convert to lower case
        text.append(txt)
        
    # loop to each condtion
    L1 = [mcont_1,mcont_2,mcont_3,mcont_4,mcont_5,mcont_6] # model condition
    L2 = [scont_1,scont_2,scont_3,scont_4,scont_5,scont_6,scont_7,scont_8,scont_9,scont_10,scont_11] # serial condition

    for t in text:
        s = 0.0 # starting score of each text
        for cont in L1:
            s += cont(t)
        score_m.append(s)
        
    for t in text:
        s = 0.0 # starting score of each text
        for cont in L2:
            s += cont(t)
        score_s.append(s)

    n_m = np.array(score_m)            
    n_s = np.array(score_s)

    max_m = np.argmax(n_m)
    max_s = np.argmax(n_s)
    
    serial_no = text_org[max_s]
    serial_pts = pts_list[max_s]
    serial_loc = loc_list[max_s]
    model_no = text_org[max_m]
    model_pts = pts_list[max_m]
    model_loc = loc_list[max_m]
            
    return serial_no,serial_pts,serial_loc,model_no,model_pts,model_loc

def decode_img(img):
    '''
    Detect barcode and qrcode and return info
    Args:
        img: the original image
    Return:
        img_decoded: the image draw barcode
        d_list: list of decoded result by pyzbar.pyzbar.decode
    '''
    d_list = decode(img)
    # print(d_list)
    for i,d in enumerate(d_list):
        img = cv2.rectangle(img,(d.rect.left,d.rect.top),(d.rect.left+d.rect.width,d.rect.top+d.rect.height),(225,0,0),2)
        img = cv2.polylines(img,[np.array(d.polygon)],True,(0,255,0),2)
        img = cv2.putText(img,str(i+1)+ "_" + d.data.decode(),(d.rect.left,d.rect.top),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),1,cv2.LINE_AA)
    return img,d_list

def parse_code(d_list):
    '''
    Extract serial and model in qrcode,barcode
    Args:
        d_list: output of pyzbar decode_img function
    Return:
        serial: serial number
        model: model number
    '''
    serial = None
    model = None
    model_loc = None
    serial_loc = None
    l = None
    if len(d_list) > 0:
        # get list of splited element
        for d in d_list:
            #print(i) #rect=Rect(left=1879, top=908, width=433, height=430)
            content = d.data.decode()
            if "$" in content:
                l = content.split("$")
            elif "," in content:
                l = content.split(",")
            else:
                pass
        # if list not None
        if l:
            for i in l:
                # looking for serial
                if "S:" in i or (len(i)==14 and ":" not in i) or (len(i)==16 and ":" not in i):
                    serial=i.replace("S:","") # for 3 cases above
                    serial_loc = (d.rect.left,d.rect.top)
                    print("Serial loc:",serial_loc)
                    return model,serial,model_loc,serial_loc
                
    return model,serial,model_loc,serial_loc

def compare_result(model_code,model_ocr,serial_code,serial_ocr,model_ocr_loc,serial_ocr_loc,model_code_loc,serial_code_loc):
    '''
    Compare serial and model between ocr and code, refer code result
    Args:
        model_code: model from code
        model_ocr: model from ocr
        serial_code: serial from code
        serial_ocr: serial from ocr
        model_ocr_loc: location of model detected by ocr
        serial_ocr_loc: location of serial detected by ocr
        model_code_loc: location of model detected by code
        serial_code_loc: location of model detected by code
    Return:
        model: model number
        serial: serial number
        model_loc: location of model
        serial_loc: location of serial
    '''
    model = None
    serial = None
    model_loc = None
    serial_loc = None
    
    model_loc = model_ocr_loc
    
    # model logic
    if model_code:
        model = model_code
    else:
        if model_ocr:
            model = model_ocr
            
    # model serial
    if serial_code:
        serial = serial_code
        serial_loc = serial_code_loc
    else:
        if serial_ocr:
            serial = serial_ocr
            serial_loc =serial_ocr_loc
            return model,serial,model_loc,serial_loc
            
    return model,serial,model_loc,serial_loc

def draw(img,d_list,ocr,serial_no,serial_pts,serial_loc,model_no,model_pts,model_loc,test):
    '''
    Draw Code and OCR
    Args:
        d_list: list of decoded (output from pyzbar.decode_img) 
        ocr: json for specific image
        serial_no: (string) serial_no
        serial_pts:(array) point array
        serial_loc: location of serial number
        model_no: (string) model_no
        model_pts: (array) point array
        model_loc: location of model number
        test: test mode option
    Return:
        img: drawed image
        content: output.txt content
    '''
    count = 0
    content = ""
    # font params
    font_size = 2
    font_weight = 8
    text_color = (0,0,255) #(222,222,0)
    # code, serial, model vars
    code = "None"
    code_pts = None # points of code
    
    # barcode and qrcode processing
    for i,d in enumerate(d_list):
        # if test mode display
        if test:
            img = cv2.rectangle(img,(d.rect.left,d.rect.top),(d.rect.left+d.rect.width,d.rect.top+d.rect.height),(225,0,0),font_weight)
            img = cv2.polylines(img,[np.array(d.polygon)],True,(0,255,0),6)
            img = cv2.putText(img,f"[{count}]",(d.rect.left,d.rect.top),cv2.FONT_HERSHEY_SIMPLEX,font_size,text_color,10,cv2.LINE_AA) #d.data.decode()
            text = f"[{count}]: {d.data.decode()}"
            content = content + text +"\n"
        else:
            text = f"{d.data.decode()}"  
            content = content + "CODE:" + text +"\n"
             
        img = cv2.polylines(img,[np.array(d.polygon)],True,(0,255,0),6) # draw barcode or qrcode
        code = d.data.decode()
        pts_code = np.array(d.polygon)
        count = count + 1
        
    # ocr processing    
    for i,t in enumerate(ocr['results']):
        # get polygon point
        pts = np.array(t['localization'])
        pts = pts.reshape((-1, 1, 2))
        if test:
            img = cv2.polylines(img,[pts],True,(255,0,0),6)
            img = cv2.putText(img,f"[{count}]",t['localization'][0],cv2.FONT_HERSHEY_SIMPLEX,font_size,text_color,font_weight,cv2.LINE_AA) #t['text']
        text = f"[{count}]: {t['text']}"
        content = content + text + "\n"
        count = count + 1
        
    # if print out on test mode    
    if test:
        print(content)
    else:
        # draw serial number if serial location and polygon points is not null
        img = cv2.polylines(img,[serial_pts],True,(255,0,0),6)
        img = cv2.putText(img,"Sn: "+serial_no,serial_loc,cv2.FONT_HERSHEY_SIMPLEX,font_size,text_color,font_weight,cv2.LINE_AA)
        # draw model number if model location and polygon points is not null
        img = cv2.polylines(img,[model_pts],True,(255,0,0),6)
        img = cv2.putText(img,"Model: "+model_no,model_loc,cv2.FONT_HERSHEY_SIMPLEX,font_size,text_color,font_weight,cv2.LINE_AA)
        content = "SERIAL_NO: " + serial_no + "\n" + "MODEL_NO: " + model_no
        #print(content)
        
    return img,content

# UPDATED FROM utils/parse_text.py (MUST EQUAL the functions from utils/parse_text.py)

def mcont_1(text):
    score = 0.
    if "google" in text:
        return score
    else:
        return 0.0
    
def mcont_2(text):
    score = 1.0
    if "model" in text or "model number:" in text:
        return score
    else:
        return 0.0
    
def mcont_3(text):
    score = 0.6
    if len(text) == 6:
        return score
    else:
        return 0.0
    
def mcont_4(text):
    score = 0.5
    if len(text) == 3:
        return score
    else:
        return 0.0
    
def mcont_5(text):
    score = 0.1
    if "listed" not in text:
        return score
    else:
        return 0.0
    
def mcont_6(text):
    score = 0.8
    if "connect a0078" in text or "a0077" in text or "gvnz4" in text or "g3aal9" in text :
        return score
    else:
        return 0.0
    
def scont_1(text):
    score = 1.0
    if "serial" in text or "serialno." in text or "serial:" in text or "sn:" in text:
        return score
    else:
        return 0.0
    
def scont_2(text):
    score = 1.0
    if len(text) == 16:
        return score
    else:
        return 0.0
    
def scont_3(text):
    score = 1.0
    if len(text) == 14:
        return score
    else:
        return 0.0
    
def scont_4(text):
    score = 0.2
    if "made" in text:
        return score
    else:
        return 0.0
    
def scont_5(text):
    score = 0.3
    if any(char.isdigit() for char in text):
        return score
    else:
        return 0.0
    
def scont_6(text):
    score = 0.1
    if " "  not in text:
        return score
    else:
        return 0.0

def scont_7(text):
    score = 0.3
    if "sku"  not in text:
        return score
    else:
        return 0.0
    
def scont_8(text):
    score = 0.3
    if "f2ic"  not in text:
        return score
    else:
        return 0.0
    
def scont_9(text):
    score = -0.7
    if "amphitheatre" in text:
        return score
    else:
        return 0.0
    
def scont_10(text):
    score = -0.9
    if "thefcc"  not in text:
        return score
    else:
        return 0.0
    
def scont_10(text):
    score = .2
    if "serial#"  not in text:
        return score
    else:
        return 0.0
    
def scont_11(text):
    score = .5
    if "designed by google" in text:
        return score
    else:
        return 0.0
    
def transcript(pot_model,pot_serial):
    '''
    transcript the potential model text and potential serial text
    Args:
        pot_model: original text of optential model text
        pot_serial: original text of optential serial text
    Return:
        model: model number
        serial: serial number
    '''
    model = pot_model.lower()
    serial = pot_serial.lower()
    
    # C1 check lenght
    if len(model)<4:
        model = None
        
    if len(serial)<14:
        serial = None
        
    # C3 check any digit
    if serial:
        if not any(c.isdigit() for c in serial):
            serial = None

    # C4 check equal regular bad words
    m_l = ["listed","listeo","us usted"]
    if model:
        for i in m_l:
            if i == model:
                model = None
                break
            
    # C5 remove substring
    m_l = ["10395a","google","model","number","nest","connect","designed","the","built","china",
           "made","thailand","operatg","operator","operating","control","us","ic","in"] # less characters word in the end
    if model:
        for i in m_l:
            model = model.replace(i,"")
        model = model.strip()
        
    m_s = ["google","model","number","nest","connect","designed","the","built","china",
           "made","thailand","operatg","operator","control","in","us","ic","serial","sn","serialno"]  
    if serial:
        for i in m_s:
            serial = serial.replace(i,"")
        serial = serial.strip()
        
    # C6 space strip
    l_s = ["  ","   ","    ","     "]
    if model:
        for i in l_s:
            model = model.replace(i,"")
        model = model.strip()
        
    if serial:
        for i in l_s:
            serial = serial.replace(i,"")
        serial = serial.strip()
        
    # C7 replace delimiter
    l_d = [",",".","/","!",":","|","-"]
    if model:
        for i in l_d:
            model = model.replace(i,"$")
        model = model.strip()
        
    # C8  extrem space strip
    l_s = [" ","  ","   ","    ","     "]
    if model:
        for i in l_s:
            model = model.replace(i,"")
        model = model.strip()
        
    if serial:
        for i in l_d:
            serial = serial.replace(i,"$")
        serial = serial.strip()
        
    # C9 replace united delimiter
    if model:
        #print(model.split('$'))
        l = model.split('$')
        #print(l)
        for i in l:
            # if i != "" or i != " ":
            if any(c.isdigit() for c in i):
                model = i
        
    if serial:
        #print(serial.split('$'))
        l = serial.split('$')
        for i in l:
            #if i != "" or i != " ":
            if any(c.isdigit() for c in i):
                serial = i
                
    # print("Model: ",model)
    
    # C10 replace delimiter $
    if model:
        model = model.replace('$',"")
    if serial:
        serial = serial.replace('$',"")
            
    # C convert to uppercase
    if model:
        model = model.upper()
    if serial:
        serial = serial.upper()
        
    return model,serial

# PADDLE-OCR utils (from parse_ocr_using_paddle.py)

def check_empty_img(image):
    # Checking if the image is empty or not
    if image is None:
        # result = "Image is empty!!"
        result = True
    else:
        # result = "Image is not empty!!"
        result = False
    return result

# Preprocess for text detection.
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

# Preprocess for text recognition.
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
        img_crop = processing.get_rotate_crop_image(ori_im, tmp_box)
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
    data_resize = processing.DetResizeForTest(data)
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
        boxes, scores = processing.boxes_from_bitmap(pred[batch_index], mask, src_w, src_h)
        boxes_batch.append({'points': boxes})
    post_result = boxes_batch
    dt_boxes = post_result[0]['points']
    dt_boxes = processing.filter_tag_det_res(dt_boxes, ori_im.shape)    
    return dt_boxes,mask

def checkFileIsInFolder(folder):
    for file_image in os.listdir(folder):
        # print("folder: {} -- > {}".format(folder, file_image))
        path_img = "{}/{}".format(folder, file_image)
        if(os.path.isfile(path_img) and (file_image.endswith(".png") or file_image.endswith(".PNG") or file_image.endswith(".jpg"))):
            current_size = os.path.getsize(path_img)
            time.sleep(0.02)
            file_size = os.path.getsize(path_img)
            if(current_size == file_size):
                return True, path_img

    return False, None

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

# DETECTOR class

class Detector:
    '''
    Class form of detector.py
    '''
    def __init__(self):
        self.indir = "samples"
        self.outdir = "list_image.txt"
        self.outto = "paddle_ocr"
        self.length = 100 # edge lenght of 4 corner cube
        self.square = 0.03 # square factor for accepting the region as a roi
        self.padding = 100 # ROI padding
        
    def export_list(self):
        '''
        Export list of images of input folder
        Args:
        Return:
            images: list of image from input image folder
        '''
        content = ""
        images = os.listdir(self.indir)
        #print(images)
        return images
        
    def run(self,images):
        '''
        Run detector
        Args:
            images: list of image from input image folder
        Return:
            img_list: list of image detected for extractor
            rois: the list of detected roi for extractor and reporter
            links: list of dictionary link between iteration and image name
        '''
        print("Detector executing..")
        
        img_list = [] # list of image detected for extractor
        content = ""
        links = [] # list of dictionary of of input image name and output image name
        
        # trying to read the images
        print(f"Reading {self.outdir}")
            
        imgs = [] # the list of input images
        rois = [] # the list of detected roi for extractor
        for i,line in enumerate(images):
            subdict = {"iteration":0,
                       "name":"",
                       "serial":"sss",
                       "model":"mmm"} # sub dictionary of image in links
            try:
                img = cv2.imread(os.path.join(self.indir,line.strip()))
                if img is not None:
                    subdict["iteration"] = i
                    subdict["name"] = line.strip()
                    imgs.append(img) # is not , !=
            except:
                subdict["iteration"] = i
                subdict["name"] = "invalid"
                print(f"CAN NOT READ {line}")
            links.append(subdict)
                
        print(f'Read {len(imgs)}/{len(images)} images')
        
        # check empty list
        if len(imgs) == 0:
            print("NO images valid, STOP RUN!")
            pass
        else:
            for i,img in enumerate(imgs):
                # make a copy of orignal image
                img_org = img.copy()
                
                #get centerpoint,(topleft,topright,bottomleft,bottomright)
                img,(center_x,center_y),_ = center_point(img,self.length,draw=False)
                
                # extract roi
                obj_color = img_org[center_y-self.length:center_y+self.length,center_x-self.length:center_x+self.length] #(y,x)
                
                # back projection
                _,thresh = back_project(img,obj_color)
                
                # find biggest roi
                _,(x_max,y_max,w_max,h_max) = detect_contour(img_org,thresh,self.square,self.padding)
                
                # draw the biggest roi
                (x_topleft,y_topleft),(x_bottomright,y_bottomright) = check_limit(img_org,x_max,y_max,w_max,h_max,self.padding)
                
                # extract roi from the original image
                roi = img_org[y_topleft:y_bottomright,x_topleft:x_bottomright]
                rois.append(roi) # collect roi for extractor
                
                content = content + f"{self.outto}/input{i}.jpg" + "\n"
                #cv2.imwrite(self.outto+f"/input{i}.jpg",roi) # export output (roi for cropped,img fullsize but no cropping wrong)
                
                img_list.append(f"{self.outto}/input{i}.jpg")
                
        # print(links)
                
        # with open(self.outto+'/detected.txt', 'w') as f: # write the list_image for ocr-docker progress
        #         f.write(content)
                
        return img_list,rois,links
        
# PADDLE-OCR class

class Extractor:
    '''
    Class form of parse_ocr_using_paddle.py
    '''
    def __init__(self):
        self.input = "paddle_ocr/detected.txt" # path to input image
        self.output = "paddle_ocr/output.json" # path to output image
        self.det_model_file_path = "paddle_ocr/ch_PP-OCRv3_det_infer/inference.pdmodel" # path to det_model_file_path
        self.rec_model_file_path = "paddle_ocr/ch_PP-OCRv3_rec_infer/inference.pdmodel" # path to rec_model_file_path
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
        
    def run(self,img_list,rois):
        '''
        Run extract
        Args:
            img_list: list of image output from detector
            rois: the list of detected roi from detector
        Return:
            result_list: list of the ocr result extracted
        '''
        print("Extractor executing..")
        
        # locchuong
        result_list = []
        
        # self.det_compiled_model, self.rec_compiled_model= load_model(self.det_model_file_path,self.rec_model_file_path)
        processing_times = collections.deque()

        LIST_IMAGES_PATH = img_list
        
        # with open(self.input, 'r') as file:
        #     LIST_IMAGES_PATH = [x.strip() for x in file.readlines()]
        
        count = 0
        for k,image_path in enumerate(LIST_IMAGES_PATH):
            
            print("Processing image: {}".format(image_path))
            start_time = time.time()
            
            image = rois[k] # HERE cv2.imread(image_path) 
            #print(f"Image {k}: {image.shape}")
            #print(f"ROI {k}: {rois[k].shape}")

            if check_empty_img(image) == True:
                print("The image is empty")
                return

            test_image = image_preprocess(image, 4192)
            preprocess_time = time.time()
            #print("Preprocess time: {}".format(preprocess_time-start_time))
            det_results = self.det_compiled_model([test_image])[self.det_compiled_model.output(0)]  # (1, 1, 3104, 4192)
            dt_boxes,mask = post_processing_detection(image, det_results)
            mask = mask.astype('int')*255            
            #cv2.imwrite(f"snippets/mask_{k}.jpg",mask) # for debug only
            dt_boxes = processing.sorted_boxes(dt_boxes)
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
                postprocess_op = processing.build_post_process(processing.postprocess_params)
                rec_result = postprocess_op(rec_results)
                for rno in range(len(rec_result)):
                    rec_res[indices[beg_img_no + rno]] = rec_result[rno]   
                if rec_res:
                    txts = [rec_res[i][0] for i in range(len(rec_res))] 
                    scores = [rec_res[i][1] for i in range(len(rec_res))]
            stop_time = time.time()
            #print("Inference time: {}".format(stop_time-preprocess_time))
            
            processing_times.append(stop_time - start_time)
            # Use processing times from last 200 frames.
            if len(processing_times) > 200:
                processing_times.popleft()
            processing_time_det = np.mean(processing_times) * 1000

            image_result = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            boxes = dt_boxes
            draw_img = processing.draw_ocr_box_txt(image_result,boxes,txts,scores,drop_score=0.5)
            write_path = self.output
            result_list = write_result_to_json_file(write_path, boxes, txts, scores, image_path,result_list)
            
            f_height, f_width = draw_img.shape[:2]
            fps = 1000 / processing_time_det
            text = "Inference time: {:.1f}ms ({:.1f} FPS)".format(processing_time_det, fps)
            cv2.putText(img=draw_img, text=text, org=(120, 120),fontFace=cv2.FONT_HERSHEY_COMPLEX, 
                        fontScale=f_width / 1500, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
            write_path = self.output.replace(".json", "_{}.jpg".format(count))
            #cv2.imwrite(write_path,draw_img) #write out image drawl
            count += 1
            # print("Completed!")
        
        # with open(self.output,"w") as f:
        #     json.dump(result_list,f)
            
        return result_list

# REPORTER class

class Reporter:
    '''
    Class form of reporter.py
    '''
    def __init__(self):
        self.workdir = "paddle_ocr" #directory of working floder (contain input.jpg and output.json)
        self.outto = "results" #directory of output file output.txt
    
    def run(self,result_list,rois,links):
        '''
        Class form of reporter.py
        Args:
            result_list: list of the ocr result extracted
            rois: the list of detected roi from detector
            links: list of dictionary link between iteration and image name
        Return:
            Export the images and json file
            links: list of result image
        '''      
        print("Reporter executing..")
        
        # check outo dir exist, if not create it
        if not os.path.isdir(self.outto):
            os.mkdir(self.outto)

        imgs = [] # numpy array form of image list
        img_orgs = [] # original image
        
        # trying to read the image, if fail skip the next image
        for i,ocr in enumerate(result_list):
            subdict = links[i]
            img_name = self.workdir + "/" + ocr['name'].split('/')[-1]
            output_img = self.outto + f"/output_{i}.jpg"
            output_text = self.outto + f'/output_{i}.txt'
            
            # reading the current image
            print("\nReading image: ",img_name)
            img = rois[i]#cv2.imread(img_name)   
            img_org = img.copy() # make a copy of orignal image
            
            # detect qrcode,barcode
            img = blur_img(img) # blur the image for barcode,qrcode detection
            _,d_list = decode_img(img)
            
            # parse code
            model_code,serial_code,model_code_loc,serial_code_loc = parse_code(d_list) 
            
            # parse text
            serial_ocr,serial_ocr_pts,serial_ocr_loc,model_ocr,model_ocr_pts,model_ocr_loc = parse_text(ocr)
            
            # stranscript
            model_ocr,serial_ocr = transcript(model_ocr,serial_ocr)
            
            # compare model,serial in both ocr and code
            model,serial,model_loc,serial_loc = compare_result(model_code,model_ocr,serial_code,serial_ocr,
                                                                model_ocr_loc,serial_ocr_loc,model_code_loc,serial_code_loc)
                
            # draw image and save it
            if not model:
                model = "None"
            if not serial:
                serial = "None"
            img,content = draw(img_org,d_list,ocr,serial,serial_ocr_pts,serial_loc,model,model_ocr_pts,model_loc,False)
            
            # update model and serial
            subdict["model"] = model
            subdict["serial"] = serial
            
            # store the output image (if we want  fix bug write out the image)
            #cv2.imwrite(output_img,img)
                
            # write down json results
            # with open(output_text, 'w') as f:
            #     f.write(content)
            with open(os.path.join(self.outto,"results.json"),"w") as f:
                json.dump(links,f)
        
        return links
                
    def find_one(self,links):
        '''
        Select one serial number, and model number from many of them
        Args:
            links: list of dictionary link between iteration and image name
        Return:
            serial: serial number
            model: model number
        '''
        # get unique serial number and model number
        serials = []
        models = []
        serial_counts = []
        model_counts = []
        links = links
        for i,img in enumerate(links):
            if img['serial'] not in serials: serials.append(img['serial'])
            if img['model'] not in models: models.append(img['model'])
        
        # count each serial number
        for s in serials:
            count = 0
            for img in links:
                if img['serial'] == s:
                    count +=1
            serial_counts.append(count)
            
        # count each model number
        for m in models:
            count = 0
            for img in links:
                if img['model'] == m:
                    count +=1
            model_counts.append(count)
            
        #print(serial_counts)
        #print(model_counts)
        
        # we care about model and take the coresponding serial
        model_max = np.argmax(model_counts)
        model = models[model_max]
        serial = serials[model_max]
        
        # export json
        output = {
            "serial":serial,
            "model":model
            }
        
        # export json output
        with open(os.path.join(self.outto,"output.json"),"w") as f:
                json.dump(output,f)
        
        return serial,model
    