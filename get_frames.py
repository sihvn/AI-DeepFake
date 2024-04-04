import cv2
import os
import math

def capture_frames(video_path, output_folder,video_name, num_frames=10):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(total_frames)
    
    # Calculate the interval between frames
    interval = math.floor(total_frames // num_frames)
    print(interval)
    
    frame_count = 0
    success = True
    frame_number = 0
    while success and frame_number < total_frames:
        # Set the frame number to read
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read a frame from the video
        success, frame = cap.read()
        
        # Check if it's time to capture a frame
        if frame_number % interval == 0 and success:
            # Save the frame to the output folder with video name and count
            frame_path = os.path.join(output_folder, f"{video_name}_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            print(f"Saved frame {frame_count}")
            
            frame_count += 1

        frame_number += 1

    cap.release()
    print("Frame capture completed.")

def capture_frames_from_folder(video_folder, output_folder, num_frames=10):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all files in the video folder
    for file_name in os.listdir(video_folder):
        if file_name.endswith(('.mp4')):  # Filter video files
            video_path = os.path.join(video_folder, file_name)
            video_name = os.path.splitext(file_name)[0]
            capture_frames(video_path, output_folder, video_name, num_frames)

# Example usage
video_path = "dataset/train_micro"
output_folder = "train/fake"
num_frames_to_capture = 10

capture_frames_from_folder(video_path, output_folder, num_frames=num_frames_to_capture)
