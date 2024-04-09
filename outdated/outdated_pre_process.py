from keras.preprocessing.image import ImageDataGenerator


def pre_process(train_data_path: str, data_generator: ImageDataGenerator):

    # Load and preprocess data
    train_generator = data_generator.flow_from_directory(
        train_data_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode="binary",
        subset="training",
    )

    validation_generator = data_generator.flow_from_directory(
        train_data_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode="binary",
        subset="validation",
    )

    return train_generator, validation_generator
