from keras.preprocessing.image import ImageDataGenerator


def pre_process():
    # Define data augmentation and preprocessing
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
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
        "data/train",
        target_size=(224, 224),
        batch_size=32,
        class_mode="binary",
        subset="training",
    )

    validation_generator = datagen.flow_from_directory(
        "data/train",
        target_size=(224, 224),
        batch_size=32,
        class_mode="binary",
        subset="validation",
    )
