import os
import csv
import cv2
from concurrent.futures import ThreadPoolExecutor

VIDEO_DIR = "/path/to/video/directory"
OUTPUT_DIR = "/path/to/output/directory"
MAX_WORKERS = os.cpu_count() # Maximum number of worker threads to use for multithreading

def process_video(video_file):
    # Load the video file
    cap = cv2.VideoCapture(video_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create output directory for the video
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    video_output_dir = os.path.join(OUTPUT_DIR, video_name)
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

            if not ret:
                break

            # Apply person detection algorithm to the frame
            # You can use any pre-trained object detection model in OpenCV here
            # Here's an example using the Haar cascades classifier for detecting pedestrians
            classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            # If people are detected in the frame, save the frame and its data
            if len(detections) > 0:
                frame_file = os.path.join(video_output_dir, f"{i:04d}.jpg")
                cv2.imwrite(frame_file, frame)

                # Save the frame data in the output data file
                for j, (x, y, w, h) in enumerate(detections):
                    writer.writerow({
                        "frame": i,
                        "person": j,
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h
                    })

    cap.release()

def process_videos():
    video_files = [os.path.join(VIDEO_DIR, f) for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]

    # Create a thread pool executor with a maximum number of worker threads
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit the video processing tasks to the thread pool
        futures = [executor.submit(process_video, video_file) for video_file in video_files]

        # Wait for all tasks to complete
        for future in futures:
            future.result()

process_videos()
