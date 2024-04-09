import math
import os

import cv2
import face_recognition
from PIL import Image


# extract frames from a single video
def extract_frames_single_video(video_path, output_folder, video_name, num_frames=5):
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
            # print(f"Saved frame {frame_count}")

            frame_count += 1

        frame_number += interval

    cap.release()
    print("Frame capture completed.")


# extract frames from a folder of videos
def extract_frames(video_folder, output_folder, num_frames=5):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all files in the video folder
    for file_name in os.listdir(video_folder):
        if file_name.endswith((".mp4")):
            video_path = os.path.join(video_folder, file_name)
            video_name = os.path.splitext(file_name)[0]
            print(video_name)
            extract_frames_single_video(
                video_path, output_folder, video_name, num_frames
            )


# extract faces from a folder of frames
def extract_faces(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(input_folder, filename)
            image = face_recognition.load_image_file(image_path)

            # Find all face locations in the image
            face_locations = face_recognition.face_locations(image)

            # Extract and save faces
            for i, (top, right, bottom, left) in enumerate(face_locations):
                # only extract faces greater than 75x75 pixels --> helps to ignore wrong faces from background objects
                if (bottom - top > 75) and (right - left > 75):
                    face_roi = image[top:bottom, left:right]
                    face_path = os.path.join(
                        output_folder, f"{os.path.splitext(filename)[0]}_face_{i}.jpg"
                    )
                    face_image = Image.fromarray(face_roi)
                    face_image.save(face_path)
                    print(f"Saved face {i} from {filename}")


# extract frames and faces from a folder of videos
def extract_frames_and_faces(input_folder, class_name_0="real", class_name_1="fake"):
    # extract frames from videos
    extract_frames(input_folder, f"{input_folder}-frames")

    # extract faces from frames
    extract_faces(f"{input_folder}-frames", f"{input_folder}-faces")


# ----------------------------------------------------------------------------------------------------
# Execute
# ----------------------------------------------------------------------------------------------------

# extract_frames_and_faces("dataset/train/real")
# extract_frames_and_faces("dataset/train/fake")
