'''
python3 scripts/craft/quickstart.py
'''
import sys
sys.path.append('/home/gom/Workspace/ocr-exp/modules')

import os
import random
# import Craft class
from craft_text_detector import Craft

# set image path and export folder directory
src ='data/sn'
image_files = os.listdir(src)
image_path = os.path.join(src,random.choice(image_files))
image = image_path # can be filepath, PIL image or numpy array
output_dir = 'outputs/'

# create a craft instance
craft = Craft(output_dir=output_dir, crop_type="poly", cuda=False)

# apply craft text detection and export detected regions to output directory
prediction_result = craft.detect_text(image)

# unload models from ram/gpu
craft.unload_craftnet_model()
craft.unload_refinenet_model()