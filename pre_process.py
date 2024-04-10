import math
import os

import cv2
import face_recognition
from PIL import Image


# Extract frames from a single video
def extract_frames_single_video(video_path, output_folder, video_name, num_frames=5):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Calculate the interval size for getting the desired number of evenly spaced frames between and excluding the first and last frames
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = math.floor(total_frames // (num_frames + 1))

    if interval == 0:
        interval = 1

    # Iterate through the frames using the calculated step size and write the frame to output_folder
    frame_index = interval
    frame_count = 0

    while frame_index < total_frames and frame_count < num_frames:
        # Read the frame at the current frame_index
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()

        # Write the frame to output_folder
        if success:
            frame_path = os.path.join(output_folder, f"{video_name}_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1

        frame_index += interval

    cap.release()


# Extract frames from a folder of videos
def extract_frames(input_folder, output_folder, num_frames=5):
    print(f'Extracting frames from "{input_folder}"...')

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all files in the video folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith((".mp4")):
            video_path = os.path.join(input_folder, file_name)
            video_name = os.path.splitext(file_name)[0]
            extract_frames_single_video(
                video_path, output_folder, video_name, num_frames
            )
            print(f'Frames have been extracted from "{file_name}".')

    print(
        f"A total of {len(os.listdir(output_folder))} frames have been extracted from {len(os.listdir(input_folder))} videos.\n"
    )


# Extract faces from a folder of frames
def extract_faces(input_folder, output_folder):
    print(f'Extracting faces from "{input_folder}"...')
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
                # Only extract faces greater than 75x75 pixels --> helps to ignore wrong faces from background objects
                if (bottom - top > 75) and (right - left > 75):
                    face_roi = image[top:bottom, left:right]
                    face_path = os.path.join(
                        output_folder, f"{os.path.splitext(filename)[0]}_face_{i}.jpg"
                    )
                    face_image = Image.fromarray(face_roi)
                    face_image.save(face_path)

                    print(f'Face {i} has been extracted from "{filename}".')

    print(
        f"A total of {len(os.listdir(output_folder))} faces have been extracted from {len(os.listdir(input_folder))} frames.\n"
    )


# Extract frames and faces from the dataset root directory, which is split into real and fake subdirectories
def extract_frames_and_faces(
    dataset_root_dir, real_subdirectory="real", fake_subdirectory="fake"
):
    # Extract frames from videos
    extract_frames(
        f"{dataset_root_dir}/{real_subdirectory}",
        f"{dataset_root_dir}/{real_subdirectory}_frames",
    )
    extract_frames(
        f"{dataset_root_dir}/{fake_subdirectory}",
        f"{dataset_root_dir}/{fake_subdirectory}_frames",
    )

    # Extract faces from frames
    extract_faces(
        f"{dataset_root_dir}/{real_subdirectory}_frames",
        f"{dataset_root_dir}/{real_subdirectory}_faces",
    )
    extract_faces(
        f"{dataset_root_dir}/{fake_subdirectory}_frames",
        f"{dataset_root_dir}/{fake_subdirectory}_faces",
    )


if __name__ == "__main__":
    extract_frames_and_faces("dataset/train")
