import math
import os

import cv2


def capture_frames(video_path, output_folder, video_name, num_frames=10):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(total_frames)

    interval = math.floor(total_frames // num_frames)
    if interval == 0:
        interval = 1
    # print(interval)

    frame_count = 0
    success = True
    frame_number = 0
    while success and frame_number < total_frames:
        # Set the frame number to read
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read a frame from the video
        success, frame = cap.read()

        # Check if its time to capture a frame
        if (
            frame_number != 0
            and interval != 0
            and frame_number % interval == 0
            and success
        ):
            frame_path = os.path.join(output_folder, f"{video_name}_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            print(f"Saved frame {frame_count}")

            frame_count += 1

        frame_number += interval

    cap.release()
    print("Frame capture completed.")


def capture_frames_from_folder(video_folder, output_folder, num_frames=10):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all files in the video folder
    for file_name in os.listdir(video_folder):
        if file_name.endswith((".mp4")):
            video_path = os.path.join(video_folder, file_name)
            video_name = os.path.splitext(file_name)[0]
            print(video_name)
            capture_frames(video_path, output_folder, video_name, num_frames)


# ----------------------------------------------------------------------------------------------------
# Test Run
# ----------------------------------------------------------------------------------------------------

# video_path = "dataset/DeepFakeDetection1/fake"
# output_folder = "dataset/DeepFakeDetection1/fake-frames-temp"
# num_frames_to_capture = 10

# capture_frames_from_folder(video_path, output_folder, num_frames=num_frames_to_capture)
