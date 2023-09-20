import cv2

img_path = "assets/txt.jpg"
heatmap_path = "assets/txt_text_score_heatmap.png"

img = cv2.imread(img_path)
heatmap = cv2.imread(heatmap_path)

# resize
h,w,_ = img.shape
heatmap_resize = cv2.resize(heatmap,(w,h),interpolation = cv2.INTER_LINEAR)

res = cv2.addWeighted(img, 0.5, heatmap_resize, 0.5, 0.0)

cv2.imwrite('res.jpg',res)