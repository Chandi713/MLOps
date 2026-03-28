"""
Load raw train images, resize, shuffle, and write X/y pickle artifacts.
Corresponds to the data preparation cells in Save_Model_Pickle.ipynb.
"""

from __future__ import annotations

import argparse
import os
import pickle
import random

import cv2
import numpy as np

from .constants import (
    CATEGORY,
    DEFAULT_TRAIN_DIR,
    DEFAULT_X_PICKLE,
    DEFAULT_Y_PICKLE,
    IMG_SIZE,
)


def build_training_list(train_dir: str, seed: int | None) -> list:
    training_array = []
    for file in sorted(os.listdir(train_dir)):
        path = os.path.join(train_dir, file)
        if not os.path.isfile(path):
            continue
        parts = file.strip().split(".")
        if len(parts) < 2:
            continue
        label_key = parts[0]
        if label_key not in CATEGORY:
            continue
        img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img_array is None:
            continue
        resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        training_array.append([resized, CATEGORY[label_key]])
    if seed is not None:
        random.seed(seed)
    random.shuffle(training_array)
    return training_array


def to_xy_arrays(training_array: list) -> tuple[np.ndarray, np.ndarray]:
    X_train = []
    y_train = []
    for feature, label in training_array:
        X_train.append(feature)
        y_train.append(label)
    X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y_train = np.array(y_train)
    return X_train, y_train


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare X/y pickle files from image folder.")
    parser.add_argument("--train-dir", default=DEFAULT_TRAIN_DIR, help="Directory of cat.* / dog.* images")
    parser.add_argument("--out-x", default=DEFAULT_X_PICKLE, help="Output path for X_train pickle")
    parser.add_argument("--out-y", default=DEFAULT_Y_PICKLE, help="Output path for y_train pickle")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed (use -1 for nondeterministic)")
    args = parser.parse_args()

    seed = None if args.seed is not None and args.seed < 0 else args.seed
    training_array = build_training_list(args.train_dir, seed)
    if not training_array:
        raise SystemExit(f"No images loaded from {args.train_dir!r}")

    X_train, y_train = to_xy_arrays(training_array)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_x)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_y)) or ".", exist_ok=True)

    with open(args.out_x, "wb") as f:
        pickle.dump(X_train, f)
    with open(args.out_y, "wb") as f:
        pickle.dump(y_train, f)

    print(f"Wrote {X_train.shape[0]} samples: {args.out_x}, {args.out_y}")


if __name__ == "__main__":
    main()
