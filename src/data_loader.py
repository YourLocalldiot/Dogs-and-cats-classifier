from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


DATA_DIR = Path(__file__).resolve().parent.parent / "PetImages"


def build_generators(batch_size=32, image_size=(150, 150)):
    """Create training and validation generators from the PetImages directory."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,
    )

    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="training",
    )

    validation_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="validation",
    )

    return train_generator, validation_generator
