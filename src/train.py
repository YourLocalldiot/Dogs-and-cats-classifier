from pathlib import Path
from tensorflow.keras.callbacks import ModelCheckpoint
from .data_loader import build_generators
from .model import build_model


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "catdog_classifier.keras"


def train_model(epochs=10):
    """Train the cat vs dog classifier and save the model."""
    train_generator, validation_generator = build_generators()
    model = build_model()

    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
    )

    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=[checkpoint],
    )

    return model


if __name__ == "__main__":
    train_model()
