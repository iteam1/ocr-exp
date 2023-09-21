'''
python3 predict.py model1/models/model_9866697.sav\
        data/unseen/1.jpg
'''
import sys
import cv2
import pickle
import numpy as np

model_path = sys.argv[1]
image_path = sys.argv[2]

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
    
    predictor = Inference()
    
    predictor.load_model(model_path)
    
    img = cv2.imread(image_path)
    
    pred, coef = predictor.predict(img)
    
    print(image_path,pred, coef)