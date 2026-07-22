from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model
from PIL import Image


def load_model_artifact(model_path: Path) -> tf.keras.Model:
    """Load a trained Keras model from disk."""
    return load_model(model_path)


def load_class_names(class_names_path: Path) -> np.ndarray:
    """Load class labels from a NumPy file."""
    return np.load(class_names_path, allow_pickle=True)


def read_image(image_path: Path) -> np.ndarray:
    """Read and convert an image to a NumPy array in RGB format."""
    with Image.open(image_path) as image:
        return np.array(image.convert("RGB"), dtype=np.float32)


def preprocess_for_model(image_array: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize and preprocess an image for EfficientNetB0."""
    resized = tf.image.resize(image_array, target_size)
    resized = tf.cast(resized, tf.float32)
    return preprocess_input(resized.numpy())


def predict_image(image_path: Path, model_path: Path, class_names_path: Path) -> Tuple[str, float, float]:
    """Return the predicted label, confidence percentage, and raw probability for an image."""
    model = load_model_artifact(model_path)
    class_names = load_class_names(class_names_path)

    image_array = read_image(image_path)
    prepared_image = preprocess_for_model(image_array)
    image_batch = np.expand_dims(prepared_image, axis=0)

    probability = float(model.predict(image_batch, verbose=0)[0][0])
    predicted_index = int(probability >= 0.5)
    predicted_label = str(class_names[predicted_index])
    confidence_percentage = float(probability * 100.0 if predicted_index == 1 else (1.0 - probability) * 100.0)

    return predicted_label, confidence_percentage, probability
