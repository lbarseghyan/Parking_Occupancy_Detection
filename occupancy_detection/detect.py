import time
import numpy as np
import cv2
import argparse
from ultralytics import YOLO
from shapely.geometry import Polygon


class ParkingOccupancyDetection:

    @staticmethod
    def calculate_iou(parking_spaces, car_boxes):

        new_car_boxes = []
        for box in car_boxes:
            x1, y2, x2, y1 = box
            new_car_boxes.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

        iou_matrix = np.zeros((len(parking_spaces), len(new_car_boxes)))
        for i in range(len(parking_spaces)):
            for j in range(len(new_car_boxes)):
                polygon1 = Polygon(parking_spaces[i])
                polygon2 = Polygon(new_car_boxes[j])
                polygon_intersection = polygon1.intersection(polygon2).area
                polygon_union = polygon1.union(polygon2).area
                iou_matrix[i][j] = polygon_intersection/polygon_union
        return iou_matrix

    @staticmethod
    def frame_detect(model, frame, parking_spaces, threshold=0.1, alpha=0.5, color=(81, 37, 21)):
        frame_copy = frame.copy()  #
        cars = model.predict(frame)[0].boxes.xyxy

        # IoU calculation
        iou_matrix = ParkingOccupancyDetection.calculate_iou(parking_spaces, cars)

        # Decision
        for parking_space, cars_iou in zip(parking_spaces, iou_matrix):
            max_iou = np.max(cars_iou)
            if max_iou < threshold:
                cv2.fillPoly(frame_copy, [np.array(parking_space, dtype=np.int32)], color)

        cv2.addWeighted(frame_copy, alpha, frame, 1 - alpha, 0, frame)
        return frame

    @staticmethod
    def video_detect(model, video_path, output_fps=1):
        cap = cv2.VideoCapture(video_path)
        video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        output_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter(output_path, output_fourcc, output_fps, video_size)

        count = 0
        detect_start = time.time()
        while cap.isOpened():
            frame_start = time.time()
            success, frame = cap.read()
            if not success:
                break
            count += cap.get(cv2.CAP_PROP_FPS)//output_fps
            cap.set(cv2.CAP_PROP_POS_FRAMES, count)

            # Detection
            frame_detected = ParkingOccupancyDetection.frame_detect(model, frame, parking_spaces, threshold=0.1, alpha=0.5)

            cv2.imshow('Parking Occupancy Detection', frame_detected)
            output.write(frame_detected)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_end = time.time()
            time.sleep((1/output_fps) - (frame_end-frame_start))
        detect_end = time.time()
        print('Detection time: ' + str(detect_end-detect_start))

        cap.release()
        output.release()
        cv2.destroyAllWindows()
        print("Output saved")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', help="Path of the video file")
    parser.add_argument('parking_spaces_path', help="File of the parking spaces' coordinates",
                        default="regions.p")
    parser.add_argument('--output_path', help="Path of the output file",
                        default="result.mp4")
    args = parser.parse_args()

    output_path = args.output_path
    if not output_path.endswith(".mp4"):
        output_path += ".mp4"

    parking_spaces_path = args.parking_spaces_path
    with open(parking_spaces_path, 'rb') as f:
        parking_spaces = np.load(f)

    video_path = args.video_path
    model = YOLO('Object_Detection/models/detect/singleclass/weights/yolov8_cars.pt')

    ParkingOccupancyDetection.video_detect(model, video_path, output_fps=1)


