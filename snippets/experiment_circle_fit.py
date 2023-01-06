img_mask = cv2.imread("/mnt/Data/debug/samples/mask_3.jpg", 0)
_, img_thres = cv2.threshold(img_mask, 150, 255, cv2.THRESH_BINARY)
img_name = "img_thres.jpg"
path_img = os.path.join(path_debug, img_name)
cv2.imwrite(path_img, img_thres)

contours, hierarchy = cv2.findContours(img_thres, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
height, width = img_thres.shape
# blank_image = np.zeros((height, width, 3), np.uint8)
img_filter = np.zeros_like(img_thres)
for cnt in contours:
rrect = cv2.minAreaRect(cnt)
width = rrect[1][0]
height = rrect[1][1]
rect = cv2.boundingRect(cnt)
if width < 200 and height < 200:
    continue
height_bounding = rect[3]
if height_bounding < 200:
    continue
# print(rect)
cv2.drawContours(img_filter, [cnt], 0, (255), -1, cv2.LINE_8,
                 hierarchy, 0)
img_name = "img_filter.jpg"
path_img = os.path.join(path_debug, img_name)
cv2.imwrite(path_img, img_filter)
# print(img_filter.shape)
height, width = img_filter.shape
vertical_px = np.sum(img_filter, axis=0)
# Normalize
normalize = vertical_px / 255
# create a black image with zeros
img_vert_prj = np.zeros_like(img_filter)
# Make the vertical projection histogram
for idx, value in enumerate(normalize):
cv2.line(img_vert_prj, (idx, 0), (idx, height - int(value)),
                       (255, 255, 255), 1)

img_vert_prj = cv2.bitwise_not(img_vert_prj)
offset_border = 2
cv2.rectangle(img_vert_prj, (0, height - offset_border),
                        (width, height), (0), -1)
img_name = "vertical_projection.jpg"
path_img = os.path.join(path_debug, img_name)
cv2.imwrite(path_img, img_vert_prj)

contours, hierarchy = cv2.findContours(img_vert_prj, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
img_filter_vert = np.zeros_like(img_filter)
for cnt in contours:
# print(type(cnt))
# print(cnt)
rect = cv2.boundingRect(cnt)
width_bounding = rect[2]
if width_bounding < 400:
    continue
rect_filter = (rect[0], 0, rect[2], height)
x = rect_filter[0]
y = rect_filter[1]
w = rect_filter[2]
h = rect_filter[3]
img_filter_vert[y:y+h, x:x+w] = img_filter[y:y+h, x:x+w]
img_name = "img_filter_vert.jpg"
path_img = os.path.join(path_debug, img_name)
cv2.imwrite(path_img, img_filter_vert)

img_rm_left = img_filter_vert.copy()
cv2.rectangle(img_rm_left, (0, 0), (int(width/2), height), (0), -1)
# img_rm_right = img_filter_vert.copy()
# cv2.rectangle(img_rm_right, (width/2, 0), (width, height), (0), -1)
list_point = []
for y in range(height - 1):
img_line = img_rm_left[y:y+1, 0:width]
pos = cv2.findNonZero(img_line)
if pos is not None:
    # print(type(pos[-1]))
    list_point.append([pos[-1][0][0], y])
# list_point = np.array(list_point, np.int32)

# point_coordinates = [[1, 0], [-1, 0], [0, 1], [0, -1]]
xc, yc, r, sigma = taubinSVD(list_point)
img_debug = np.zeros_like(img_filter)
for point in list_point:
cv2.circle(img_debug, (point[0], point[1]), 3, (255), -1)

img_name = "img_debug.jpg"
path_img = os.path.join(path_debug, img_name)
cv2.imwrite(path_img, img_debug)

cv2.circle(img_debug, (int(xc), int(yc)), 5, (255), -1)
cv2.circle(img_debug, (int(xc), int(yc)), int(r), (255), 3)
img_name = "img_debug1.jpg"
path_img = os.path.join(path_debug, img_name)
cv2.imwrite(path_img, img_debug)

