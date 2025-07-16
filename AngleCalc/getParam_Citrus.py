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
if(predict == True):
    seg_dir = 'D:/Jeju/Thai/Research/Citrus Fruit Segmentation/Dataset/YOLO_Seg_Citrus_ALL_Predict/'
    out_data_file = 'Data_Predict.csv'
    data_folder = "YOLO_Seg_Citrus_ALL_Predict"
    replace_folder_clean = "CalParam/YOLO_Seg_Citrus_ALL_Predict_Clean"
    replace_folder_peak = "CalParam/YOLO_Seg_Citrus_ALL_Predict_Peak"

else:
    seg_dir = 'D:/Jeju/Thai/Research/Citrus Fruit Segmentation/Dataset/YOLO_Seg_Citrus_ALL/'
    out_data_file = 'Data_GT.csv'
    data_folder = "YOLO_Seg_Citrus_ALL"
    replace_folder_clean = "CalParam/YOLO_Seg_Citrus_ALL_Clean"
    replace_folder_peak = "CalParam/YOLO_Seg_Citrus_ALL_Peak"


if not os.path.isdir(seg_dir.replace(data_folder,replace_folder_clean)):
    os.mkdir(seg_dir.replace(data_folder,replace_folder_clean))
    os.mkdir(seg_dir.replace(data_folder,replace_folder_clean)+"/images")
if not os.path.isdir(seg_dir.replace(data_folder,replace_folder_peak)):
    os.mkdir(seg_dir.replace(data_folder,replace_folder_peak))
    os.mkdir(seg_dir.replace(data_folder,replace_folder_peak)+"/images")

central_list = []
central_ellipse_list = []

segment_list = []
segment_ellipse_list = []
segment_area_list = []

points_list = []
image_list = []
THRESHOLD_DISTANCE_CENTRAL_SEGMENT = 1000
PIXEL2MICROMET = 0.17

# open the file in the write mode
with open(out_data_file, 'w', newline='') as csv_f:
    # create the csv writer
    writer = csv.writer(csv_f)
    # image path, image height, image width
    # central width, central length, central angle, central area, Central ellipse area
    # segment width, segment length, segment angle, segment area, segment ellipse area
    fields = ['No.', 'Image Path', 'Image Width', 'Image Height', 'Central Area', 'Segment Area', 'Fruit Area', 'Segment Num', 'Fruit Width', 'Fruit Length']
    writer.writerow(fields)

    central_list = []
    central_ellipse_list = []
    central_area_list = []

    segment_list = []
    segment_ellipse_list = []
    segment_area_list = []


    segment_Num_list = []

    fruit_area_list = []
    fruit_width_list = []
    fruit_length_list = []

    image_width_list = []
    image_height_list = []

    for file_path in glob.glob(seg_dir + 'labels/*.txt'):
        print(file_path)
        line_data = ""
        img_path = file_path.replace(".txt",".jpg")
        img_path = img_path.replace("labels","images")
        image = cv2.imread(img_path)

        """# Định nghĩa các tham số vật liệu
        shininess = 100
        reflectivity = 0.5
        light_position = (0, 0, 1)

        # Tính toán Specular Lighting
        specular_lighting = cv2.illumination(image, light_position, shininess, reflectivity)"""

        h, w = image.shape[:2]
        image_width_list.append(w)
        image_height_list.append(h)

        # print (h , w)
        # cv2.imshow("Hello", image)
        # cv2.waitKey(0)

        image_list.append(os.path.split(img_path)[1])
        points_list = []

        file1 = open(file_path, 'r')
        Lines = file1.readlines()

        count = 0
        # Strips the newline character
        central_area = 0
        central_ellipse = None
        segment_area = 0
        segment_ellipse= None

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
            points = points

            ellipse = cv2.fitEllipse(points)
            # Retrieve ellipse parameters
            (center, axes, angle) = ellipse
            x = np.array(points[:, 0]) * w
            y = np.array(points[:, 1]) * h
            i = np.arange(len(x))
            # Area=np.sum(x[i-1]*y[i]-x[i]*y[i-1])*0.5 # signed area, positive if the vertex sequence is counterclockwise
            Area = np.abs(np.sum(x[i - 1] * y[i] - x[i] * y[i - 1]) * 0.5)  # one line of code for the shoelace formula
            # ellipse_area = math.pi * axes[0] * axes[1]
            if(label == 1):
                if(segment_area < Area):
                    segment_area = Area
                    segment_ellipse = ellipse
                    segment_points = points
            else:
                if(central_area < Area):
                    central_area = Area
                    central_ellipse = ellipse
                    central_points = points

        segment_ellipse_list.append(segment_ellipse)
        segment_area_list.append(segment_area)
        segment_list.append(segment_points)

        central_ellipse_list.append(central_ellipse)
        central_area_list.append(central_area)
        central_list.append(central_points)

        #get fruit area
        mask_fruit = np.zeros(image.shape, dtype=np.uint8)
        (center, axes, angle) = segment_ellipse
        color = [1, 1, 1]
        mask_fruit = cv2.ellipse(mask_fruit,(int(center[0] * w), int(center[1] * h)), (int(axes[0] * w / 2 + 150), int(axes[1] * h / 2 + 150)), angle, 0, 360, color, -1)
        image_fruit = image * mask_fruit

        img_gray = cv2.cvtColor(image_fruit, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

        ret, bw_img = cv2.threshold(image_fruit, 50, 255, cv2.THRESH_BINARY)
        (thresh, im_bw) = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        mask_fruit = (bw_img > 20) * 255

        thresh, ret = cv2.threshold(img_gray, 50, 255, 0)

        se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        mask = cv2.morphologyEx(ret, cv2.MORPH_CLOSE, se1)
        mask = cv2.morphologyEx(ret, cv2.MORPH_CLOSE, se1)
        mask = cv2.morphologyEx(ret, cv2.MORPH_CLOSE, se1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)

        mask3 = np.dstack([mask, mask, mask]) / 255

        clean = image * mask3
        clean = clean.astype(np.uint8) #if not, the image is totally white.
        contours, hierarchy = cv2.findContours(image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        # draw contours on the original image
        image_copy = clean.copy()

        if len(contours) != 0:
            # draw in blue the contours that were founded
            # find the biggest countour (c) by the area
            c = max(contours, key=cv2.contourArea)
            cv2.drawContours(image_copy, [c], -1, 255, 3)
            # x, y, w, h = cv2.boundingRect(c)
            # draw the biggest contour (c) in green
            # cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        fruit_ellipse = cv2.fitEllipse(c)
        (center, axes, angle) = fruit_ellipse
        x_center = center[0]
        y_center = center[1]
        a = axes[0]
        b = axes[1]
        fruit_width = min(a, b)
        fruit_length = a + b - fruit_width

        fruit_width_list.append(fruit_width)
        fruit_length_list.append(fruit_length)

        #1133 -> 3495 = 10cm : 2362 pixel = 100mm height
        #85-->790 = 3cm : 705 pixel = 30 width

        fruit_area = (mask == 255).sum()
        fruit_area_list.append(fruit_area)

        cv2.imwrite(img_path.replace(data_folder, replace_folder_clean), clean)

        #Counting the number of segments
        #calculate distance between center of eclipse of Central segmentation and the contour point on Central segmenation boundary.
        (center, axes, angle) = central_ellipse
        list_distance_central = []
        for i in range(central_points.shape[0]):
            list_distance_central.append(math.sqrt(((central_points[i][0] - center[0]) * w) ** 2 + ((central_points[i][1] - center[1]) * h) ** 2))

        array_data = np.asarray(list_distance_central)
        peaks = find_peaks(array_data, prominence=10)
        add_index = int(peaks[0][0] * 2 / 3)

        array_data = np.concatenate([array_data, array_data[:add_index]])
        peaks = find_peaks(array_data, prominence=10)
        print("Peaks position:", peaks[0])
        """x = np.arange(1, len(array_data)+1)
        y = array_data
        # plotting
        plt.title("Line graph")
        plt.xlabel("X axis")
        plt.ylabel("Y axis")
        plt.plot(x, y, color="red")
        plt.show()"""
        cv2.drawContours(clean, [c], -1, 255, 3)
        segment_Num_list.append(len(peaks[0]))

        # cv2.circle(image_fruit, (int(central_points[0][0] * w), int(central_points[0][1] * h)), 10, (0, 255, 0), -1)
        for index in peaks[0]:
            if(index < central_points.shape[0]):
                cv2.circle(clean, (int(central_points[index][0] * w), int(central_points[index][1]* h)), 10, (255, 0, 0), -1)
            else:
                cv2.circle(clean, (int(central_points[index - central_points.shape[0]][0] * w), int(central_points[index - central_points.shape[0]][1] * h)), 10,
                           (255, 0, 0), -1)


        # calculate distance between center of eclipse of Segment segmentation and the contour point on Segment segmenation boundary.
        (center, axes, angle) = segment_ellipse
        list_distance_segment = []
        for i in range(segment_points.shape[0]):
            list_distance_segment.append(-math.sqrt(
                ((segment_points[i][0] - center[0]) * w) ** 2 + ((segment_points[i][1] - center[1]) * h) ** 2))
        array_data = np.asarray(list_distance_segment)
        peaks = find_peaks(array_data, prominence=5)
        add_index = int(peaks[0][0] * 2 / 3)

        array_data = np.concatenate([array_data, array_data[:add_index]])
        peaks = find_peaks(array_data, prominence=5)
        print("Peaks position:", peaks[0])

        # cv2.circle(image_fruit, (int(segment_points[0][0] * w), int(segment_points[0][1] * h)), 10, (0, 255, 0), -1)
        for index in peaks[0]:
            if (index < segment_points.shape[0]):
                cv2.circle(clean, (int(segment_points[index][0] * w), int(segment_points[index][1] * h)), 10,
                           (255, 0, 0), -1)
            else:
                cv2.circle(clean, (int(segment_points[index - segment_points.shape[0]][0] * w), int(segment_points[index - segment_points.shape[0]][1] * h)), 10,
                           (255, 0, 0), -1)
        cv2.imwrite(img_path.replace(data_folder, replace_folder_peak), clean)
        """half = cv2.resize(image, (0, 0), fx=0.3, fy=0.3)
        cv2.imshow("peaks", half)
        cv2.waitKey(0)"""

    for i in range(len(fruit_area_list)):
        line = [i+1, image_list[i], image_width_list[i], image_height_list[i],central_area_list[i], segment_area_list[i], fruit_area_list[i], segment_Num_list[i], fruit_width_list[i], fruit_length_list[i]]
        writer.writerow(line)
