from keras.applications import EfficientNetB0
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model


def get_model():
    # Load pre-trained EfficientNetB0
    base_model = EfficientNetB0(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )

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

    return model
