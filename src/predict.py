from pathlib import Path
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "catdog_classifier.keras"


def predict_image(image_path):
    """Predict whether an image is a cat or a dog."""
    model = load_model(MODEL_PATH)
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array, verbose=0)[0][0]
    label = "Dog" if prediction >= 0.5 else "Cat"
    return label, float(prediction)
