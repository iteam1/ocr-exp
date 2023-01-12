import os
import cv2
import copy
import json
import math
import time
import paddle
import collections
import numpy as np
from PIL import Image
import  pre_post_processing as processing
from openvino.runtime import Core

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

class OCR:

    def __init__(self):
        self.input = "ppocr/detected.txt" # path to input image
        self.output = "ppocr/output.json" # path to output image
        self.det_model_file_path = "ppocr/ch_PP-OCRv3_det_infer/inference.pdmodel" # path to det_model_file_path
        self.rec_model_file_path = "ppocr/ch_PP-OCRv3_rec_infer/inference.pdmodel" # path to rec_model_file_path
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
        
    def run(self,rois):
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
            
            processing_times.append(stop_time - start_time)
            # Use processing times from last 200 frames.
            if len(processing_times) > 200:
                processing_times.popleft()
            processing_time_det = np.mean(processing_times) * 1000

            image_result = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            boxes = dt_boxes
            draw_img = processing.draw_ocr_box_txt(image_result,boxes,txts,scores,drop_score=0.5)
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