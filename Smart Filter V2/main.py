import os
import time
import csv
import cv2
from concurrent.futures import ThreadPoolExecutor
import argparse
from ultralytics import YOLO
import cvzone
import math
from sort import *
from PIL import Image
from Cos_sim import Pycos


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--VIDEO_DIR", required=True, help="path to the directory containing the videos")
ap.add_argument("-o", "--OUTPUT_DIR", required=True, help="path to the output directory")
ap.add_argument("-n", "--MAX_WORKERS", default=os.cpu_count(), type=int,
                help="number of processes to use for processing the videos")
args = vars(ap.parse_args())

if not os.path.exists(args["OUTPUT_DIR"]):
    os.makedirs(args["OUTPUT_DIR"])

model = YOLO('yolov8n.pt')

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)


totalCount = []
classNames = []

with open('labels.txt', 'r') as f:
    classNames = f.read().splitlines()

def process_video(video_file):
    # Load the video file
    cap = cv2.VideoCapture(video_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detections = np.empty((0, 5))
    pframe = 0
    final = 0
    ret = True  # creates a boolean
    ret, old_frame = cap.read()  # ret is true and the first frame of video saved in old_frame
    old_frame_rgb = cv2.cvtColor(old_frame, cv2.COLOR_BGR2RGB)
    old_frame_pil = Image.fromarray(old_frame_rgb)

    # Create output directory for the video
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    video_output_dir = os.path.join(args["OUTPUT_DIR"], video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    # Create output data file for the video
    data_file = os.path.join(video_output_dir, f"{video_name}.csv")

    # Open CSV file for writing
    with open(data_file, mode='w') as csv_file:
        fieldnames = ['frame', 'person', 'x', 'y', 'w', 'h']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # Process each frame in the video
        for i in range(frame_count):
            # Read the frame
            ret, frame = cap.read()
            if pframe != (frame_count-1):
             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
             frame_pil = Image.fromarray(frame_rgb)

             sem = Pycos(frame_pil, old_frame_pil)
             print(sem)

             old_frame_pil = frame_pil
             pframe += 1



            if not ret:
                break

            start = time.time()
            results = model(frame, batch = 16)
            end = time.time()
            elap = end - start
            fps = 1 / (end - start)
            start = end
            print('Frame' , pframe , "{:.2f}".format(elap), 'Seconds')
            cv2.putText(frame, "FPS :" + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 255), 2,
                        cv2.LINE_AA)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Bounding Box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
                    w, h = x2 - x1, y2 - y1

                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    # Class Name
                    cls = int(box.cls[0])
                    currentClass = classNames[cls]

                    if currentClass == "person" and conf > 0.6 and sem < 0.98:
                        currentArray = np.array([x1, y1, x2, y2, conf])
                        detections = np.vstack((detections, currentArray))
                        cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
                        frame_file = os.path.join(video_output_dir, f"{i:04d}.jpg")
                        cv2.imwrite(frame_file, frame)
                        resultsTracker = tracker.update(detections)

                        for result in resultsTracker:
                            x1, y1, x2, y2, id = result
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            print(result)
                            w, h = x2 - x1, y2 - y1
                            j = int(id)
                            # cvzone.putTextRect(frame, f' {int(id)}', (max(0, x1), max(35, y1)),
                            #                    scale=2, thickness=3, offset=10)
                            writer.writerow({
                                "frame": i,
                                "person": j,
                                "x": x1,
                                "y": y1,
                                "w": w,
                                "h": h
                            })
        final = elap * frame_count
        print("Total Time taken", "{:.2f}".format(final), 'Seconds')
    cap.release()

def process_videos():
    video_files = [os.path.join(args["VIDEO_DIR"], f) for f in os.listdir(args["VIDEO_DIR"]) if f.endswith(".mp4")]

    # Create a thread pool executor with a maximum number of worker threads
    with ThreadPoolExecutor(max_workers=args["MAX_WORKERS"]) as executor:
        # Submit the video processing tasks to the thread pool
        futures = [executor.submit(process_video, video_file) for video_file in video_files]

        # Wait for all tasks to complete
        for future in futures:
            future.result()

process_videos()
