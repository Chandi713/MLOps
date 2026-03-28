"""
Load trained model and held-out pickles from the train stage; write eval metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np
from tensorflow.keras.models import load_model

from .constants import (
    DEFAULT_EVAL_METRICS_PATH,
    DEFAULT_X_TEST_PICKLE,
    DEFAULT_Y_TEST_PICKLE,
    TRAINED_MODEL_PATH,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model on held-out test pickles.")
    parser.add_argument("--model-in", default=TRAINED_MODEL_PATH)
    parser.add_argument("--x-test-pickle", default=DEFAULT_X_TEST_PICKLE)
    parser.add_argument("--y-test-pickle", default=DEFAULT_Y_TEST_PICKLE)
    parser.add_argument("--metrics-out", default=DEFAULT_EVAL_METRICS_PATH)
    args = parser.parse_args()

    model = load_model(args.model_in)

    with open(args.x_test_pickle, "rb") as f:
        X_test = pickle.load(f)
    with open(args.y_test_pickle, "rb") as f:
        y_test = pickle.load(f)

    X_test = np.asarray(X_test, dtype=np.float32) / 255.0
    y_test = np.asarray(y_test)

    result = model.evaluate(X_test, y_test, batch_size=32, verbose=0, return_dict=True)
    out = {k: float(v) for k, v in result.items()}

    os.makedirs(os.path.dirname(os.path.abspath(args.metrics_out)) or ".", exist_ok=True)
    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Eval metrics: {out}")


if __name__ == "__main__":
    main()
