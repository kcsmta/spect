import os, cv2
from os.path import basename

folder_name = '/home/kcsmta/Desktop/WORKS/SPECT/code/spect_data'
folder_dst = '/home/kcsmta/Desktop/WORKS/SPECT/code/stress'

patients_folder = [folder_name+'/'+i for i in os.listdir(folder_name)]
# print len(patients_folder)
for patient in patients_folder:
    # print patient
    files = [patient+'/'+i for i in os.listdir(patient)]
    for f in files:
        if '_1.jpg' in f:
            path_to_image=f
            img = cv2.imread(path_to_image)
            # cv2.imshow("Source_image", img)

            #Stress data
            x_stress=572
            y_stress=98
            h_stress=314
            w_stress=314
            stress_crop_img = img[y_stress:y_stress+h_stress, x_stress:x_stress+w_stress]
            # cv2.imshow("Stress_image", stress_crop_img)
            cv2.imwrite(folder_dst+"/Stress_"+basename(f), stress_crop_img)

            # #Rest data
            # x_rest=572
            # y_rest=432
            # h_rest=314
            # w_rest=314
            # rest_crop_img = img[y_rest:y_rest+h_rest, x_rest:x_rest+w_rest]
            # cv2.imshow("Rest_image", rest_crop_img)
            # cv2.imwrite("Rest.jpg", rest_crop_img)
