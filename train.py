from keras.models import Model
from keras.preprocessing.image import DirectoryIterator


def train(
    model: Model,
    train_generator: DirectoryIterator,
    validation_generator: DirectoryIterator,
):
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=10,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
    )

    return history
