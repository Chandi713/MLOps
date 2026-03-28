"""
Load pickles, split holdout for evaluation, normalize, train CNN.
Writes trained model, train metrics, and test-set pickles for the evaluate stage.
Corresponds to training cells in Save_Model_Pickle.ipynb.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np
from tensorflow.keras.layers import (
    Activation,
    Conv2D,
    Dense,
    Flatten,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential

from .constants import (
    DEFAULT_TRAIN_METRICS_PATH,
    DEFAULT_X_PICKLE,
    DEFAULT_X_TEST_PICKLE,
    DEFAULT_Y_PICKLE,
    DEFAULT_Y_TEST_PICKLE,
    TRAINED_MODEL_PATH,
)


def build_model(input_shape: tuple[int, int, int]) -> Sequential:
    model = Sequential(
        [
            Input(shape=input_shape),
            Conv2D(32, (3, 3)),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(32, (3, 3)),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(32),
            Dense(8),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train cats vs. dogs CNN from pickle artifacts.")
    parser.add_argument("--x-pickle", default=DEFAULT_X_PICKLE)
    parser.add_argument("--y-pickle", default=DEFAULT_Y_PICKLE)
    parser.add_argument("--out-x-test", default=DEFAULT_X_TEST_PICKLE)
    parser.add_argument("--out-y-test", default=DEFAULT_Y_TEST_PICKLE)
    parser.add_argument("--model-out", default=TRAINED_MODEL_PATH)
    parser.add_argument("--metrics-out", default=DEFAULT_TRAIN_METRICS_PATH)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--validation-split", type=float, default=0.2)
    args = parser.parse_args()

    with open(args.x_pickle, "rb") as f:
        X = pickle.load(f)
    with open(args.y_pickle, "rb") as f:
        y = pickle.load(f)

    X = np.asarray(X)
    y = np.asarray(y)
    n = len(y)
    rng = np.random.default_rng(args.split_seed)
    perm = rng.permutation(n)
    n_test = max(1, int(args.test_fraction * n))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    X_test = X[test_idx]
    y_test = y[test_idx]
    X_tr = X[train_idx]
    y_tr = y[train_idx]

    os.makedirs(os.path.dirname(os.path.abspath(args.out_x_test)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_y_test)) or ".", exist_ok=True)
    with open(args.out_x_test, "wb") as f:
        pickle.dump(X_test, f)
    with open(args.out_y_test, "wb") as f:
        pickle.dump(y_test, f)

    X_tr = X_tr.astype(np.float32) / 255.0

    model = build_model(tuple(X_tr.shape[1:]))
    history = model.fit(
        X_tr,
        y_tr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=args.validation_split,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.model_out)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.metrics_out)) or ".", exist_ok=True)

    model.save(args.model_out)

    last = {k: float(v[-1]) for k, v in history.history.items()}
    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(last, f, indent=2)

    print(f"Saved model to {args.model_out}")
    print(f"Train metrics: {last}")


if __name__ == "__main__":
    main()
