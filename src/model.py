"""Model architectures for pneumonia detection."""

from __future__ import annotations

import logging
from typing import Literal

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.optimizers import Adam

from src import config


class ModelBuilder:
    """Build custom and transfer-learning image classification models."""

    def __init__(self) -> None:
        """Initialize builder."""
        self.base_model: tf.keras.Model | None = None

    def build_custom_cnn(self, input_shape: tuple = (config.IMG_SIZE, config.IMG_SIZE, 3)) -> Model:
        """Build and compile a baseline custom CNN model."""
        model = Sequential(
            [
                Input(shape=input_shape),
                Conv2D(32, (3, 3), activation="relu", padding="same"),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), activation="relu", padding="same"),
                MaxPooling2D((2, 2)),
                Conv2D(128, (3, 3), activation="relu", padding="same"),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(256, activation="relu"),
                Dropout(0.5),
                Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=config.LEARNING_RATE),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def build_transfer_model(
        self,
        base: Literal["resnet50", "vgg16"] = "resnet50",
        input_shape: tuple = (config.IMG_SIZE, config.IMG_SIZE, 3),
    ) -> Model:
        """Build a transfer learning model with ResNet50 or VGG16 backbone."""
        base = base.lower()
        if base == "resnet50":
            backbone = tf.keras.applications.ResNet50(
                weights="imagenet",
                include_top=False,
                input_shape=input_shape,
            )
        elif base == "vgg16":
            backbone = tf.keras.applications.VGG16(
                weights="imagenet",
                include_top=False,
                input_shape=input_shape,
            )
        else:
            raise ValueError("Supported bases are 'resnet50' and 'vgg16'.")

        backbone.trainable = False
        self.base_model = backbone

        x = GlobalAveragePooling2D()(backbone.output)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.5)(x)
        outputs = Dense(1, activation="sigmoid")(x)
        model = Model(inputs=backbone.input, outputs=outputs, name=f"{base}_transfer")
        model.compile(
            optimizer=Adam(learning_rate=config.LEARNING_RATE),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def unfreeze_and_finetune(self, model: Model, num_layers: int = 20) -> Model:
        """Unfreeze the last `num_layers` of the base model and recompile for fine-tuning."""
        base_model = self.base_model
        if base_model is None:
            for layer in model.layers:
                if isinstance(layer, tf.keras.Model):
                    base_model = layer
                    break
        if base_model is None:
            raise ValueError("Base model not found for fine-tuning.")

        for layer in base_model.layers:
            layer.trainable = False
        for layer in base_model.layers[-num_layers:]:
            layer.trainable = True

        model.compile(
            optimizer=Adam(learning_rate=config.FINE_TUNE_LR),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        logging.info("Unfroze last %d layers for fine-tuning.", num_layers)
        return model


if __name__ == "__main__":
    builder = ModelBuilder()
    cnn = builder.build_custom_cnn()
    resnet = builder.build_transfer_model(base="resnet50")
    finetuned = builder.unfreeze_and_finetune(resnet, num_layers=20)
    print(cnn.summary())
    print(finetuned.summary())
