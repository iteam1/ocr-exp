import cv2
import time
import numpy as np
import tools.infer.utility as utility
from openvino.runtime import Core
from ppocr.data import create_operators,transform
from ppocr.postprocess import build_post_process

class TextE2E(object):
    def __init__(self,pgnet_path):
        '''
        Init E2E model
        Args:
            pgnet_path (string): directory to E2E PGNet pretrained model
                        example: '../e2e_server_pgnetA_infer/inference.pdmodel' 
        '''
        core = Core()
        self.predictor = core.compile_model(model=pgnet_path, device_name="CPU")
        
        self.e2e_algorithm ='PGNet' # only support pgnet
        self.use_onnx = False
        self.use_openvino = True
        
        pre_process_list = [{'E2EResizeForTest':{'max_side_len':768,
                                                 'valid_set':'totaltext'}},
                            {'NormalizeImage': {'std': [0.229, 0.224, 0.225],
                                                'mean': [0.485, 0.456, 0.406],
                                                'scale': '1./255.',
                                                'order': 'hwc'}},
                            {'ToCHWImage': None},
                            {'KeepKeys':{'keep_keys': ['image', 'shape']}}]
        
        postprocess_params = {}
        postprocess_params['name']='PGPostProcess'
        postprocess_params['score_thresh']=0.5
        postprocess_params['character_dict_path']='../models/ic15_dict.txt'
        postprocess_params['valid_set']='totaltext'
        postprocess_params['mode']='fast'
        
        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = build_post_process(postprocess_params)
        
    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes
    
    def __call__(self,img):
        '''
        infer image by PGNET
        Args:
            img (numpy array): input image
        Return:
            dt_boxes: coordniate of text region, None if error happen
            strs: list of ocr string, None if error happen
            elapse: time elapse, None if error happen
        '''
        ori_img = img.copy() # original image
        # format input data
        data = {'image':img}
        data = transform(data,self.preprocess_op)
        img,shape_list = data
        if img is None:
            return None,None,None
        img = np.expand_dims(img,axis=0)
        shape_list = np.expand_dims(shape_list,axis=0)
        img =img.copy()
        starttime = time.time()
       
        # OpenVINO Support here (Only support OpenVINO)
        if self.use_openvino:
            outputs = self.predictor([img])
            out_layers = self.predictor.output
            preds = {}
            preds['f_border'] = outputs[out_layers(0)]
            preds['f_char'] = outputs[out_layers(1)]
            preds['f_direction'] = outputs[out_layers(2)]
            preds['f_score'] = outputs[out_layers(3)]
            # postprocess result
            post_result = self.postprocess_op(preds,shape_list)
            points = post_result['points']
            strs = post_result['texts']
            dt_boxes = self.filter_tag_det_res_only_clip(points,ori_img.shape)
            elapse = time.time() - starttime
            return dt_boxes,strs,elapse
        else:
           return None,None,None

if __name__ == "__main__":
    pgnet_path = "../models/e2e_server_pgnetA_infer/inference.pdmodel"
    img_path = "../debug/DataFromHailu/84BCDAC7-8E70-4046-AF7B-FA73FF2EA309_1_201_a copy.jpg"
    img = cv2.imread(img_path)
    text_detector = TextE2E(pgnet_path)
    dt_boxes,strs,elapse = text_detector(img)
    output = utility.draw_e2e_res(dt_boxes, strs,img_path)
    cv2.imwrite('ouput.jpg',output)
    