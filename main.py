import sys

from keras.applications import EfficientNetB0, ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

from evaluate import *
from model import *
from pre_process import *
from train import *


def main(train_data_path: str, test_data_path: str):
    print("\nAI Project - DeepFake Detection\n")

    # # Define data augmentation and preprocessing
    # data_generator = ImageDataGenerator(
    #     # rescale=1.0 / 255,
    #     rotation_range=20,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     shear_range=0.2,
    #     zoom_range=0.2,
    #     horizontal_flip=True,
    #     validation_split=0.2,  # Split data into training and validation
    # )

    # print("Pre-processing data...")
    # train_generator, validation_generator = pre_process(train_data_path, data_generator)

    # print("Building the model...")
    # model = get_model()

    # print("Training...")
    # history = train(model, train_generator, validation_generator)

    # print("Evaluating model on test data...")
    # evaluate(model, test_data_path, data_generator)

    # Define data augmentation and preprocessing
    datagen = ImageDataGenerator(
        # rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,  # Split data into training and validation
    )

    # Load and preprocess data
    train_generator = datagen.flow_from_directory(
        train_data_path,
        target_size=(224, 224),
        batch_size=10,
        class_mode="binary",
        subset="training",
    )

    validation_generator = datagen.flow_from_directory(
        train_data_path,
        target_size=(224, 224),
        batch_size=10,
        class_mode="binary",
        subset="validation",
    )

    test_generator = datagen.flow_from_directory(
        test_data_path, target_size=(224, 224), batch_size=10, class_mode="binary"
    )

    # Load pre-trained EfficientNetB0
    base_model = EfficientNetB0(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )

    # # Load pre-trained ResNet50
    # base_model = ResNet50(
    #     weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    # )

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=10,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
    )

    # loss, accuracy = model.evaluate(test_generator)
    # print(f"Test loss: {loss:.4f}")
    # print(f"Test accuracy: {accuracy*100:.2f}%")

    evaluation = model.evaluate(test_generator)
    print(evaluation)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Arguments missing.")
        print("Usage: main.py [train_data_path] [test_data_path]")

    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]

    main(train_data_path, test_data_path)
