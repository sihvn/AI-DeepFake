import os

import face_recognition
from PIL import Image


def extract_faces_from_frames(input_folder, output_folder):
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


input_folder = "dataset\\DeepFakeDetection1\\real-frames"
output_folder = "dataset\\DeepFakeDetection1-faces\\real"

extract_faces_from_frames(input_folder, output_folder)
