import cv2
import numpy as np
# import matplotlib.pyplot as plt

def extract_roi(image, in_circle_radius, out_circle_radius, start_angle, end_angle):
    # create a mask image of the same shape as input image, filled with 0s (black color)
    mask = np.zeros_like(image)
    rows, cols,_ = mask.shape
    if in_circle_radius == "none":
        # create a white filled ellipse
        mask=cv2.ellipse(mask, center=(rows/2, cols/2), axes=(out_circle_radius,out_circle_radius), angle=0, startAngle=start_angle, endAngle=end_angle, color=(255,255,255), thickness=-1)
    else:
        # create a white filled out-ellipse
        mask=cv2.ellipse(mask, center=(rows/2, cols/2), axes=(out_circle_radius,out_circle_radius), angle=0, startAngle=start_angle, endAngle=end_angle, color=(255,255,255), thickness=-1)
        # create a black filled in-ellipse
        mask=cv2.ellipse(mask, center=(rows/2, cols/2), axes=(in_circle_radius,in_circle_radius), angle=0, startAngle=start_angle-1, endAngle=end_angle+1, color=(0,0,0), thickness=-1)
    # Bitwise AND operation to black out regions outside the mask
    result = np.bitwise_and(image,mask)
    return result

# load source image
path_to_image = './Stress.jpg'
image = cv2.imread(path_to_image)
rows, cols,_ = image.shape
radius = cols/2

# region 1
cv2.imwrite("roi_1_1.jpg", extract_roi(image, "none", radius/4, 0, 360))

# region 2
cv2.imwrite("roi_2_1.jpg", extract_roi(image, radius/4, 2*radius/4, -135, -45))
cv2.imwrite("roi_2_2.jpg", extract_roi(image, radius/4, 2*radius/4, -45, 45))
cv2.imwrite("roi_2_3.jpg", extract_roi(image, radius/4, 2*radius/4, 45, 135))
cv2.imwrite("roi_2_4.jpg", extract_roi(image, radius/4, 2*radius/4, 135, 225))

# region 3
cv2.imwrite("roi_3_1.jpg", extract_roi(image, 2*radius/4, 3*radius/4, 0, 60))
cv2.imwrite("roi_3_2.jpg", extract_roi(image, 2*radius/4, 3*radius/4, 60, 120))
cv2.imwrite("roi_3_3.jpg", extract_roi(image, 2*radius/4, 3*radius/4, 120, 180))
cv2.imwrite("roi_3_4.jpg", extract_roi(image, 2*radius/4, 3*radius/4, 180, 240))
cv2.imwrite("roi_3_5.jpg", extract_roi(image, 2*radius/4, 3*radius/4, 240, 300))
cv2.imwrite("roi_3_6.jpg", extract_roi(image, 2*radius/4, 3*radius/4, 300, 360))

# region 4
cv2.imwrite("roi_4_1.jpg", extract_roi(image, 3*radius/4, 4*radius/4, 0, 60))
cv2.imwrite("roi_4_2.jpg", extract_roi(image, 3*radius/4, 4*radius/4, 60, 120))
cv2.imwrite("roi_4_3.jpg", extract_roi(image, 3*radius/4, 4*radius/4, 120, 180))
cv2.imwrite("roi_4_4.jpg", extract_roi(image, 3*radius/4, 4*radius/4, 180, 240))
cv2.imwrite("roi_4_5.jpg", extract_roi(image, 3*radius/4, 4*radius/4, 240, 300))
cv2.imwrite("roi_4_6.jpg", extract_roi(image, 3*radius/4, 4*radius/4, 300, 360))

