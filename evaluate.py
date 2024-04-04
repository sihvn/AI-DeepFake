from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator


def evaluate(model: Model, test_data_path: str, data_generator: ImageDataGenerator):
    test_generator = data_generator.flow_from_directory(
        test_data_path, target_size=(224, 224), batch_size=32, class_mode="binary"
    )

    loss, accuracy = model.evaluate(test_generator)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy*100:.2f}%")
