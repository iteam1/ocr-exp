'''
python3 scripts/craft/text_detector.py
'''
import os
import random
import sys
sys.path.append('/home/gom/Workspace/ocr-exp/modules')
# import craft functions
from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
    export_extra_results,
    empty_cuda_cache
)

# set image path and export folder directory
src ='data/sn'
image_files = os.listdir(src)
image = os.path.join(src,random.choice(image_files)) # can be filepath, PIL image or numpy array
output_dir = 'outputs/'
cuda_opt = False

# read image
image = read_image(image)

# load models
refine_net = load_refinenet_model(cuda=cuda_opt)
craft_net = load_craftnet_model(cuda=cuda_opt)

# perform prediction
prediction_result = get_prediction(
    image=image,
    craft_net=craft_net,
    refine_net=refine_net,
    text_threshold=0.7,
    link_threshold=0.4,
    low_text=0.4,
    cuda=cuda_opt,
    long_size=1280
)

# export detected text regions
exported_file_paths = export_detected_regions(
    image=image,
    regions=prediction_result["boxes"],
    output_dir=output_dir,
    rectify=True
)

# export heatmap, detection points, box visualization
export_extra_results(
    image=image,
    regions=prediction_result["boxes"],
    heatmaps=prediction_result["heatmaps"],
    output_dir=output_dir
)

# unload models from gpu
if cuda_opt:
    empty_cuda_cache()