from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.data_utils import load_image_array


def build_dataset(image_dirs: List[Path], labels: List[int], target_size: Tuple[int, int] = (224, 224)) -> Tuple[np.ndarray, np.ndarray]:
    """Build a simple dataset array from image folders."""
    images = []
    targets = []

    for image_dir, label in zip(image_dirs, labels):
        for image_path in sorted(image_dir.glob("*")):
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                images.append(load_image_array(image_path, target_size))
                targets.append(label)

    return np.stack(images, axis=0), np.array(targets, dtype=np.int32)
