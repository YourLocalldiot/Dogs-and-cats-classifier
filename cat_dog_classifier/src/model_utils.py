from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import confusion_matrix, f1_score


def create_base_model(input_shape: Tuple[int, int, int] = (224, 224, 3)) -> tf.keras.Model:
    """Create an EfficientNetB0-based transfer learning model for binary classification."""
    base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=input_shape)
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="cat_dog_efficientnet")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", Precision(name="precision"), Recall(name="recall")],
    )
    return model


def build_callbacks(model_path: Path) -> list[tf.keras.callbacks.Callback]:
    """Create common training callbacks for early stopping, LR reduction, and checkpointing."""
    return [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6),
        ModelCheckpoint(filepath=model_path, monitor="val_loss", save_best_only=True),
    ]


def plot_training_history(history, output_path: Path) -> None:
    """Save accuracy and loss plots for the training history."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="train accuracy")
    axes[0].plot(history.history["val_accuracy"], label="val accuracy")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train loss")
    axes[1].plot(history.history["val_loss"], label="val loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def evaluate_model(model: tf.keras.Model, test_ds: tf.data.Dataset) -> dict[str, float]:
    """Evaluate the trained model on the test dataset and return key metrics."""
    results = model.evaluate(test_ds, verbose=0)
    loss = float(results[0])
    accuracy = float(results[1])
    precision = float(results[2])
    recall = float(results[3])
    f1 = float(2 * precision * recall / (precision + recall + 1e-8))

    return {
        "test_loss": loss,
        "test_accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def save_model_artifacts(model: tf.keras.Model, output_dir: Path, class_names: np.ndarray) -> None:
    """Save the trained model and class labels."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "final_model.keras")
    np.save(output_dir / "class_names.npy", class_names)


def make_confusion_matrix(model: tf.keras.Model, test_ds: tf.data.Dataset, class_names: np.ndarray) -> np.ndarray:
    """Generate a confusion matrix for the test set."""
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        predictions = (model.predict(images, verbose=0).ravel() >= 0.5).astype(int)
        y_true.extend(labels.numpy().astype(int))
        y_pred.extend(predictions.tolist())

    return confusion_matrix(y_true, y_pred)
