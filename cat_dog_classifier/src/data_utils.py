from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "PetImages"


def set_seed(seed: int = 42) -> None:
    """Set global random seeds."""

    import os
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def remove_corrupted_images(dataset_dir: Path) -> List[Path]:
    """Remove images TensorFlow cannot decode."""

    removed_files = []

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    for image_path in dataset_dir.rglob("*"):

        if (
            not image_path.is_file()
            or image_path.suffix.lower() not in valid_extensions
            or image_path.name.startswith(".")
        ):
            continue

        try:
            image_bytes = tf.io.read_file(str(image_path))
            tf.image.decode_image(image_bytes, channels=3)

        except Exception:
            print(f"Removing {image_path}")
            image_path.unlink(missing_ok=True)
            removed_files.append(image_path)

    return removed_files

def build_datasets(
    dataset_dir: Path,
    image_size=(224, 224),
    batch_size=32,
    seed=42,
):

    class_names = sorted(
        [
            folder.name
            for folder in dataset_dir.iterdir()
            if folder.is_dir()
        ]
    )

    image_paths = []
    labels = []

    for label, class_name in enumerate(class_names):

        class_folder = dataset_dir / class_name

        for image_path in class_folder.glob("*"):

            if image_path.suffix.lower() not in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
            }:
                continue

            image_paths.append(str(image_path))
            labels.append(label)

    image_paths = np.array(image_paths)
    labels = np.array(labels)

    # 70% train / 30% temp
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths,
        labels,
        test_size=0.30,
        random_state=seed,
        stratify=labels,
    )

    # Split remaining 30% into 15% validation / 15% test
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.50,
        random_state=seed,
        stratify=temp_labels,
    )

    def load_image(path, label):
        image = tf.io.read_file(path)

        image = tf.image.decode_image(
            image,
            channels=3,
            expand_animations=False,
        )

        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32)

        # Preprocess for EfficientNet
        image = preprocess_input(image)

        return image, label

    def make_dataset(paths, labels, shuffle=False):

        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

        ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

        if shuffle:
            ds = ds.shuffle(len(paths), seed=seed)

        ds = ds.batch(batch_size)

        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds

    train_ds = make_dataset(train_paths, train_labels, shuffle=True)
    val_ds = make_dataset(val_paths, val_labels)
    test_ds = make_dataset(test_paths, test_labels)

    return train_ds, val_ds, test_ds, class_names