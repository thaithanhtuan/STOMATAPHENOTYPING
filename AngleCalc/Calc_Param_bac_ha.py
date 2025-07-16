import glob
import cv2
import numpy as np
import json
import math
import csv


seg_dir = 'D:\\Jeju\\Thai\\Research\\Stomata\\Dataset\\20230812 Bac ha_Tuan Thesis_Gap\\'

stomata_list = []
stomata_ellipse_list = []

pore_list = []
pore_ellipse_list = []
pore_area_list = []

points_list = []
THRESHOLD_DISTANCE_STOMATA_PORE = 200
PIXEL2MICROMET = 0.17

# open the file in the write mode
with open('Data.csv', 'w') as csv_f:
    # create the csv writer
    writer = csv.writer(csv_f)
    # image path, image height, image width
    # stomata width, stomata length, stomata angle, stomata area, sto ellipse area
    # pore width, pore length, pore angle, pore area, pore ellipse area
    fields = ['No.', 'Image Path', 'Height', 'Width', 'Sto Width', 'Sto Length', 'Sto Angle', 'Sto Area', 'Sto Elip Area', 'Sto center X', 'Sto center Y', 'Pore Width', 'Pore Length', 'Pore Angle', 'Pore Area', 'Pore Elip Area', 'Pore center X', 'Pore center Y', 'Height(micromet)', 'Width(micromet)', 'Sto Width(micromet)', 'Sto Length(micromet)', 'Sto Area(micromet)', 'Sto Elip Area(micromet)', 'Sto center X(micromet)', 'Sto center Y(micromet)', 'Pore Width(micromet)', 'Pore Length(micromet)', 'Pore Area(micromet)', 'Pore Elip Area(micromet)', 'Pore center X(micromet)', 'Pore center Y(micromet)']
    writer.writerow(fields)

    for file_path in glob.glob(seg_dir + '*.json'):
        line_data = ""
        img_path = file_path.replace(".json",".JPG")
        image = cv2.imread(img_path)

        h, w = image.shape[:2]
        # print (h , w)
        # cv2.imshow("Hello", image)
        # cv2.waitKey(0)
        stomata_list = []
        stomata_ellipse_list = []

        pore_list = []
        pore_ellipse_list = []
        pore_area_list = []

        points_list = []

        with open(file_path, 'r') as file:
            data = json.load(file)
            print(data["imagePath"])
            # if(data["imagePath"] != "B (11).JPG"):
            #      continue
            line_data = data["imagePath"] + ", " + str(data["imageHeight"]) + ", " + str(data["imageWidth"])
            for item in data['shapes']:

                label = item["label"]
                if(label=="stomata"):
                    stomata_list.append(item["points"])
                elif (label == "pore"):
                    pore_list.append(item["points"])
                points_list.append(item["points"])
            # print("stomatas:", len(stomata_list))
            # print("pores:", len(pore_list))

            #find corresponding stomata and pore
            if (len(stomata_list) != len(pore_list)):
                print("WARNING: " + data["imagePath"])
                print("stomatas:", len(stomata_list))
                print("pores:", len(pore_list))
                # exit(0)
            # continue
            for pore in pore_list:
                # Convert list to list of tuples
                points_tuples = [tuple(point) for point in pore]
                # print('points_tuples: ', points_tuples)

                # Make the list to array
                points_arr = np.array(points_tuples, dtype=np.float32)
                # print('points_arr: ', points_arr)

                # Normalized points array to 8bit one
                # points_arr = normalize_array_to_8bit(points_arr)
                # print('points_arr: ', points_arr)

                ## Fit an ellipse to the points
                if (len(points_arr) == 4):
                    points_arr = np.append(points_arr, [[points_arr[0][0] + 1, points_arr[0][1] + 1]]).reshape(-1, 2)
                points_arr = np.array(points_arr, dtype=np.float32)
                ellipse = cv2.fitEllipse(points_arr)
                # print(ellipse)

                # Retrieve ellipse parameters
                (center, axes, angle) = ellipse
                # pore_ellipse_list.append(ellipse)

                # Draw the ellipse on an image
                # cv2.ellipse(image, ellipse, (0, 255, 0), 2)

                # Schoelace formula.
                # x,y are arrays containing coordinates of the polygon vertices
                x = np.array(points_arr[:, 0])
                y = np.array(points_arr[:, 1])
                i = np.arange(len(x))
                # Area=np.sum(x[i-1]*y[i]-x[i]*y[i-1])*0.5 # signed area, positive if the vertex sequence is counterclockwise
                Area = np.abs(np.sum(x[i - 1] * y[i] - x[i] * y[i - 1]) * 0.5)  # one line of code for the shoelace formula
                # ellipse_area = math.pi * axes[0] * axes[1]
                pore_ellipse_list.append((ellipse, Area))
                pore_area_list.append(Area)
            index = 0
            for stomata in stomata_list:
                index = index + 1
                # Convert list to list of tuples
                points_tuples = [tuple(point) for point in stomata]
                # print('points_tuples: ', points_tuples)

                # Make the list to array
                points_arr = np.array(points_tuples, dtype=np.float32)
                # print('points_arr: ', points_arr)

                # Normalized points array to 8bit one
                # points_arr = normalize_array_to_8bit(points_arr)
                # print('points_arr: ', points_arr)

                ## Fit an ellipse to the points
                ellipse = cv2.fitEllipse(points_arr)
                # print(ellipse)

                # Retrieve ellipse parameters
                (center, axes, angle) = ellipse
                stomata_ellipse_list.append(ellipse)
                flag = False
                i = 0
                for pore_ellipse in pore_ellipse_list:
                    (pcenter, paxes, pangle) = pore_ellipse[0]
                    i = i + 1
                    if(math.sqrt(math.pow(center[0]-pcenter[0],2) + math.pow(center[1]-pcenter[1],2)) < THRESHOLD_DISTANCE_STOMATA_PORE):
                        flag = True
                        pArea = pore_ellipse[1]
                        pellipse_area = math.pi * paxes[0] * paxes[1] / 4
                        break

                # Draw the ellipse on an image
                cv2.ellipse(image, ellipse, (0, 255, 0), 2)
                cv2.ellipse(image, pore_ellipse[0], (255, 255, 0), 2)
                # cv2.imshow("hello", image)
                # cv2.waitKey(0)

                #Schoelace formula.
                # x,y are arrays containing coordinates of the polygon vertices
                x = np.array(points_arr[:, 0])
                y = np.array(points_arr[:, 1])
                i = np.arange(len(x))
                # Area=np.sum(x[i-1]*y[i]-x[i]*y[i-1])*0.5 # signed area, positive if the vertex sequence is counterclockwise
                Area = np.abs(np.sum(x[i - 1] * y[i] - x[i] * y[i - 1]) * 0.5)  # one line of code for the shoelace formula
                ellipse_area = math.pi * axes[0] * axes[1] / 4
                # image path, image height, image width

                line_data = [str(index), data["imagePath"] , str(data["imageHeight"]) , str(data["imageWidth"])]
                # stomata width, stomata length, stomata angle, stomata area, sto ellipse area
                line_data = line_data + [str(axes[0]) , str(axes[1]) , str(angle) , str(Area) , str(ellipse_area), str(center[0]), str(center[1])]
                # pore width, pore length, pore angle, pore area, pore ellipse area
                if(flag == False):
                    line_data = line_data + [ 0, 0, 0, 0, 0]
                else:
                    line_data = line_data + [ str(paxes[0]) , str(paxes[1]) , str(pangle) , str(pArea) , str(pellipse_area), str(pcenter[0]), str(pcenter[1])]
                #, 'Height(micromet)', 'Width(micromet)', 'Sto Width(micromet)', 'Sto Length(micromet)', 'Sto Angle(micromet)', 'Sto Area(micromet)', 'Sto Elip Area(micromet)', 'Sto center X(micromet)', 'Sto center Y(micromet)', 'Pore Width(micromet)', 'Pore Length(micromet)', 'Pore Angle(micromet)', 'Pore Area(micromet)', 'Pore Elip Area(micromet)', 'Pore center X(micromet)', 'Pore center Y(micromet)']
                line_data = line_data + [str(PIXEL2MICROMET * data["imageHeight"]) , str(PIXEL2MICROMET * data["imageWidth"]) , str(PIXEL2MICROMET * axes[0]) , str(PIXEL2MICROMET * axes[1]) , str(PIXEL2MICROMET * Area) , str(PIXEL2MICROMET * ellipse_area), str(PIXEL2MICROMET * center[0]), str(PIXEL2MICROMET * center[1])]
                if (flag == False):
                    line_data = line_data + [0, 0, 0, 0]
                else:
                    line_data = line_data + [str(PIXEL2MICROMET * paxes[0]), str(PIXEL2MICROMET * paxes[1]), str(PIXEL2MICROMET * pArea), str(PIXEL2MICROMET * pellipse_area),
                                             str(PIXEL2MICROMET * pcenter[0]), str(PIXEL2MICROMET * pcenter[1])]
                writer.writerow(line_data)

                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (int(center[0] + 10), int(center[1] + 10))
                fontScale = 1
                fontColor = (0, 0, 255)
                thickness = 2
                lineType = 2
                fields = ['Sto Width', 'Sto Length', 'Sto Angle', 'Sto Area',
                          'Sto Elip Area', 'Sto center X', 'Sto center Y', 'Pore Width', 'Pore Length', 'Pore Angle',
                          'Pore Area', 'Pore Elip Area', 'Pore center X', 'Pore center Y']
                string = "Sto Width: " + str(int(axes[0])) + ", " + "Sto Length: " + str(int(axes[1]))

                cv2.putText(image, string,
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            thickness,
                            lineType)
                bottomLeftCornerOfText = (int(center[0] + 10), int(center[1] + 40))
                string = "Sto Angle: " + str(int(angle)) + ", " + "Sto Area: " + str(int(Area))
                cv2.putText(image, string,
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            thickness,
                            lineType)
                bottomLeftCornerOfText = (int(center[0] - 80), int(center[1] - 80))
                string = str(index)
                fontScale = 3
                fontColor = (255, 0, 0)
                thickness = 5
                cv2.putText(image, string,
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            thickness,
                            lineType)

            bottomLeftCornerOfText = (100, 100)
            string = "Number of Stomatas: " + str(index)
            fontScale = 3
            fontColor = (255, 0, 0)
            thickness = 5
            cv2.putText(image, string,
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)
            # Display the image
            scale_percent = 40  # percent of original size
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
            dim = (width, height)

            # resize image
            cv2.imwrite(file_path.replace(".json","_viz.jpg"), image)
            # imgresized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

            # cv2.imshow("Ellipse", imgresized)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()