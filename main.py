import sys

from keras.preprocessing.image import ImageDataGenerator

from evaluate import *
from model import *
from pre_process import *
from train import *


def main(train_data_path: str, test_data_path: str):
    print("\nAI Project - DeepFake Detection\n")

    print("Train data path:", train_data_path)
    print("Test data path:", test_data_path)

    # Define data augmentation and preprocessing
    data_generator = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,  # Split data into training and validation
    )

    print("Pre-processing data...")
    train_generator, validation_generator = pre_process(train_data_path, data_generator)

    print("Building the model...")
    model = get_model()

    print("Training...")
    history = train(model, train_generator, validation_generator)

    print("Evaluating model on test data...")
    evaluate(model, test_data_path, data_generator)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Arguments missing.")
        print("Usage: main.py [train_data_path] [test_data_path]")

    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]

    main(train_data_path, test_data_path)
