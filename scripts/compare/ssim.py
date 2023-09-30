'''
python3 scripts/compare/ssim.py
'''
import os
import cv2
import random
from skimage.metrics import structural_similarity as compare_ssim

# Initialize
DIM = 128
THRESH = 0.3 # You can adjust this threshold based on your requirements
src = 'data/dataset'

# Choice file randomly
labels = os.listdir(src)
label1 = random.choice(labels)
label2 = random.choice(labels)
file1 = random.choice(os.listdir(os.path.join(src,label1)))
file2 = random.choice(os.listdir(os.path.join(src,label2)))
image1_path = os.path.join(src,label1,file1)
image2_path = os.path.join(src,label2,file2)
print('Image1:',image1_path)
print('Image2:',image2_path)

# Load the two images you want to compare
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# Resize
image1 = cv2.resize(image1,(DIM,DIM),interpolation=cv2.INTER_AREA)
image2 = cv2.resize(image2,(DIM,DIM),interpolation=cv2.INTER_AREA)

# Convert the images to grayscale (optional)
image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Calculate SSIM between the two images
ssim_score = compare_ssim(image1_gray, image2_gray)

# Print the SSIM score (1.0 means the images are identical)
print(f"SSIM Score: {ssim_score}")

# You can set a threshold to determine whether the images are similar or not
if ssim_score > THRESH:
    print("Images are similar")
else:
    print("Images are not similar")
