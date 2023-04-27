
import time

# import os
import numpy as np
import cv2
import argparse
from ultralytics import YOLO
from shapely.geometry import Polygon   # as shapely_poly
from shapely.geometry import box


def calculate_iou(parking_spaces, car_boxes):
    
    new_car_boxes = []
    for box in car_boxes:
        x1, y2, x2, y1 = box 
        new_car_boxes.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

    iou_matrix = np.zeros((len(parking_spaces), len(new_car_boxes)))   # creates matrix, where ccolumns are the parking slots
                                                                     # and the rows are the detected cars
    for i in range(len(parking_spaces)):                             # for each parking slot
        for j in range(len(new_car_boxes)):                          # for each detected car
            polygon1 = Polygon(parking_spaces[i])                     # create polygon for parking space
            polygon2 = Polygon(new_car_boxes[j])                     # create polygon for bounding box of a car
            polygon_intersection = polygon1.intersection(polygon2).area
            polygon_union = polygon1.union(polygon2).area
            iou_matrix[i][j] = polygon_intersection/polygon_union
    return iou_matrix


if __name__ == "__main__":
    start = time.time()   # +
    
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', help="Path of the video file")
    parser.add_argument('parking_spaces_path', help="File of the parking spaces' coordinates",
                        default="regions.p")
    args = parser.parse_args()

#     regions_start = time.time()
    parking_spaces_path = args.parking_spaces_path
    with open(parking_spaces_path, 'rb') as f:
        parking_spaces = np.load(f)
#     regions_end = time.time()
#     print(f'Reading Regions: {regions_end-regions_start}')
#     print('----------------------------------')

    alpha = 0.6
    video_source = args.video_path
    cap = cv2.VideoCapture(video_source)
    video_FourCC = cv2.VideoWriter_fourcc(*'mp4v')  #
    video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video_fps = 4        # !! Adjust with count !!                     #cap.get(cv2.CAP_PROP_FPS)
    output = cv2.VideoWriter("_Videos_/output/yolo/video32_5fps.mp4", video_FourCC, video_fps, video_size)

    model = YOLO('runs/detect/train/weights/yolov8_cars.pt')

    count = 0      # +
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_copy = frame.copy()   #
#         cv2.imwrite(f"frame{count}.jpg", overlay)   #####

        count += 6      # +
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        
#         rgb_image = frame[:, :, ::-1]

        # Detection 
        
#         detect_start = time.time()
        cars = model.predict(frame)[0].boxes.xyxy
#         detect_end = time.time()
        
        # IoU calculation
#         iou_start = time.time()
        iou_matrix = calculate_iou(parking_spaces, cars)
#         iou_end = time.time()
        
        # Decision
#         decision_start = time.time()
        for parking_space, cars_iou in zip(parking_spaces, iou_matrix):
            max_iou = np.max(cars_iou)
            if max_iou < 0.1:                                          # Treshold of miou for parking space detection
                cv2.fillPoly(frame_copy, [np.array(parking_space)], (71, 27, 92))   # (71, 27, 92) is the color for free parking space
#                 free_space = True
        cv2.addWeighted(frame_copy, alpha, frame, 1 - alpha, 0, frame)
#         decision_end = time.time()
        
#         print(f'Yolov8 Detection Time: {detect_end-detect_start}')
#         print(f'IoU Matrix:            {iou_end-iou_start}')
#         print(f'Decison Time:          {decision_end - decision_start}')
#         print('-----------------------------------------------')
        
        cv2.imshow('Parking Occupancy Detection', frame)
        output.write(frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    output.release()
    cv2.destroyAllWindows()
    
    end = time.time()   # +
    print(f'Execution Time:  {end-start}')    #
    print("output saved as out_test_yolo.mp4")    #
