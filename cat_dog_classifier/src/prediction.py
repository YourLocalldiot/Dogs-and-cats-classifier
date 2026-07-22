from pathlib import Path
from typing import Tuple

import numpy as np
from tensorflow.keras.models import load_model

from src.data_utils import load_image_array


def predict_image(image_path: Path, model_path: Path, class_names_path: Path) -> Tuple[str, float]:
    """Return the predicted class label and confidence for a single image."""
    model = load_model(model_path)
    class_names = np.load(class_names_path, allow_pickle=True)

    image_array = load_image_array(image_path)
    image_batch = np.expand_dims(image_array, axis=0)
    probability = float(model.predict(image_batch, verbose=0)[0][0])

    predicted_index = int(probability >= 0.5)
    class_name = str(class_names[predicted_index])
    confidence = probability if predicted_index == 1 else 1.0 - probability
    return class_name, confidence
