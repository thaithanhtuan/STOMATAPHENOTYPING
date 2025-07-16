import glob
import cv2
import numpy as np
import json
import math
import csv
import os
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

predict = True
isTrain = True
if(predict == True):
    if (isTrain == True):
        seg_dir = 'D:/Jeju/Thai/Research/Stomata/ultralytics-main/runs\segment/predict_Stomata_trainset/labels/'
        image_dir = 'D:/Jeju/Thai/Research/Stomata/Dataset/labelme-main/examples/instance_segmentation/YOLO/train/images/'
        out_data_file = 'Data_train_Predict.csv'
        data_folder = "images"
        replace_folder_viz = "Viz"
    else:
        seg_dir = 'D:/Jeju/Thai/Research/Stomata/ultralytics-main/runs\segment/predict_Stomata_valset/labels/'
        image_dir = 'D:/Jeju/Thai/Research/Stomata/Dataset/labelme-main/examples/instance_segmentation/YOLO/val/images/'
        out_data_file = 'Data_val_Predict.csv'
        data_folder = "images"
        replace_folder_viz = "Viz"
else:
    if (isTrain == True):
        seg_dir = 'D:/Jeju/Thai/Research/Stomata/Dataset/labelme-main/examples/instance_segmentation/YOLO/train/labels/'
        image_dir = 'D:/Jeju/Thai/Research/Stomata/Dataset/labelme-main/examples/instance_segmentation/YOLO/train/images/'
        out_data_file = 'Data_train_GT.csv'
        data_folder = "images"
        replace_folder_viz = "Viz"
    else: #val
        seg_dir = 'D:/Jeju/Thai/Research/Stomata/Dataset/labelme-main/examples/instance_segmentation/YOLO/val/labels/'
        image_dir = 'D:/Jeju/Thai/Research/Stomata/Dataset/labelme-main/examples/instance_segmentation/YOLO/val/images/'
        out_data_file = 'Data_val_GT.csv'
        data_folder = "images"
        replace_folder_viz = "Viz"

if not os.path.isdir(seg_dir.replace(data_folder,replace_folder_viz)):
    os.mkdir(seg_dir.replace(data_folder,replace_folder_viz))
    os.mkdir(seg_dir.replace(data_folder,replace_folder_viz)+"/images")

pore_points_list = []
pore_ellipse_list = []
pore_area_list = []

stomata_points_list = []
stomata_ellipse_list = []
stomata_area_list = []

object_points_list = []
object_ellipse_list = []
object_area_list = []

points_list = []
image_list = []
THRESHOLD_DISTANCE_STOMATA_PORE = 50
PIXEL2MICROMET = 0.17

# open the file in the write mode
with open(out_data_file, 'w', newline='') as csv_f:
    # create the csv writer
    writer = csv.writer(csv_f)
    # image path, image height, image width
    # central width, central length, central angle, central area, Central ellipse area
    # segment width, segment length, segment angle, segment area, segment ellipse area
    fields = ['No.', 'Image Path', 'Image Width', 'Image Height', 'Label', 'Area', 'Minor Elipse', 'Mojor Elipse', 'Center Elipse X', 'Center Elipse Y', 'Angle Elipse']
    writer.writerow(fields)

    image_width_list = []
    image_height_list = []
    label_list = []

    for file_path in glob.glob(seg_dir + '*.txt'):
        print(file_path)
        line_data = ""
        img_path = file_path.replace(".txt", ".jpg")
        if(predict == True):
            img_path = image_dir + os.path.basename(img_path) 
        else:            
            img_path = img_path.replace("labels","images")
        image = cv2.imread(img_path)

        """# Định nghĩa các tham số vật liệu
        shininess = 100
        reflectivity = 0.5
        light_position = (0, 0, 1)

        # Tính toán Specular Lighting
        specular_lighting = cv2.illumination(image, light_position, shininess, reflectivity)"""

        h, w = image.shape[:2]
        

        # print (h , w)
        # cv2.imshow("Hello", image)
        # cv2.waitKey(0)

        
        points_list = []

        file1 = open(file_path, 'r')
        Lines = file1.readlines()

        count = 0
        # Strips the newline character
        pore_area = 0
        pore_ellipse = None
        stomata_area = 0
        stomata_ellipse= None

        for line in Lines:

            count += 1
            print("Line{}: {}".format(count, line.strip()))
            line = line.strip()
            li = list(map(np.float32, line.split()))
            label = li[0]

            if(predict == True):
                points = np.array(li[1:-1])
            else:
                points = np.array(li[1:])
            points = points.reshape(-1, 2)
            points = points * [w,h]
            # points = list(dict.fromkeys(points))

            # if (len(points) == 4):
            #     points = np.append(points, [[points[0][0] + 1, points[0][1] + 1]]).reshape(-1, 2)
            points = np.array(points, dtype=np.float32)
            if (len(points) <= 4):
                continue                
            
            # image_list.append(os.path.split(img_path)[1])
            
            ellipse = cv2.fitEllipse(points)
            # Retrieve ellipse parameters
            (center, axes, angle) = ellipse
            x = np.array(points[:, 0])
            y = np.array(points[:, 1])
            i = np.arange(len(x))
            # Area=np.sum(x[i-1]*y[i]-x[i]*y[i-1])*0.5 # signed area, positive if the vertex sequence is counterclockwise
            Area = np.abs(np.sum(x[i - 1] * y[i] - x[i] * y[i - 1]) * 0.5)  # one line of code for the shoelace formula
            # ellipse_area = math.pi * axes[0] * axes[1]
            # names: ['stomata', 'pore']
            #Check if Stomata is near the boundary, add all the data to excel file, but later, remove it from the excel file.

            image_list.append(os.path.split(img_path)[1])
            image_width_list.append(w)
            image_height_list.append(h)
            
            object_ellipse_list.append(ellipse)
            object_area_list.append(Area)
            object_points_list.append(points)
            if (label == 1):
                label_list.append("Stomata")
            else:
                label_list.append("Pore")

            if(label == 1):
                stomata_area = Area
                stomata_ellipse = ellipse
                stomata_points = points

                stomata_ellipse_list.append(stomata_ellipse)
                stomata_area_list.append(stomata_area)
                stomata_points_list.append(stomata_points)
            else:
                pore_area = Area
                pore_ellipse = ellipse
                pore_points = points

                pore_ellipse_list.append(pore_ellipse)
                pore_area_list.append(pore_area)
                pore_points_list.append(pore_points)
    fields = ['No.', 'Image Path', 'Image Width', 'Image Height', 'Label', 'Area', 'Minor Elipse', 'Mojor Elipse',
              'Center Elipse X', 'Center Elipse Y', 'Angle Elipse']
    for i in range(len(image_list)):
        line = [i+1, image_list[i], image_width_list[i], image_height_list[i], label_list[i], object_area_list[i], object_ellipse_list[i][1][0], object_ellipse_list[i][1][1], object_ellipse_list[i][0][0], object_ellipse_list[i][0][1], object_ellipse_list[i][2]]
        writer.writerow(line)


    #Find Matching between stomata and pore
