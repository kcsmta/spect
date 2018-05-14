# matplotlib inline
import cv2
import os
from matplotlib import pyplot as plt

# path_to_image='./src.jpg'
path_to_image='./0_1.jpg'
img = cv2.imread(path_to_image)
cv2.imshow("Source_image", img)

#Stress data
x_stress=572
y_stress=98
h_stress=314
w_stress=314
stress_crop_img = img[y_stress:y_stress+h_stress, x_stress:x_stress+w_stress]
cv2.imshow("Stress_image", stress_crop_img)
cv2.imwrite("Stress.jpg", stress_crop_img)

#Rest data
x_rest=572
y_rest=432
h_rest=314
w_rest=314
rest_crop_img = img[y_rest:y_rest+h_rest, x_rest:x_rest+w_rest]
cv2.imshow("Rest_image", rest_crop_img)
cv2.imwrite("Rest.jpg", rest_crop_img)

cv2.waitKey(0)
# #Stress data
# x_stress=1040
# y_stress=155
# h_stress=160
# w_stress=190
# crop_img = img[y_stress:y_stress+h_stress, x_stress:x_stress+w_stress]
# cv2.imshow("Stress2", crop_img)

# #Rest data
# x_rest=1040
# y_rest=490
# h_rest=160
# w_rest=190
# crop_img = img[y_rest:y_rest+h_rest, x_rest:x_rest+w_rest]
# cv2.imshow("Rest2", crop_img)



