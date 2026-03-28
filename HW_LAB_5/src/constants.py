"""Shared defaults for the cats vs. dogs preprocessing and training pipeline."""

import os

# Label mapping from image filename prefix (e.g. cat.0.jpg -> 0)
CATEGORY = {"cat": 0, "dog": 1}

IMG_SIZE = 75

# Paths relative to project root (HW_LAB_5)
DEFAULT_TRAIN_DIR = os.path.join("data", "train")
DEFAULT_ARTIFACTS_DIR = "artifacts"
DEFAULT_MODEL_DIR = "models"

DEFAULT_X_PICKLE = os.path.join(DEFAULT_ARTIFACTS_DIR, "X_train.pickle")
DEFAULT_Y_PICKLE = os.path.join(DEFAULT_ARTIFACTS_DIR, "y_train.pickle")
DEFAULT_X_TEST_PICKLE = os.path.join(DEFAULT_ARTIFACTS_DIR, "X_test.pickle")
DEFAULT_Y_TEST_PICKLE = os.path.join(DEFAULT_ARTIFACTS_DIR, "y_test.pickle")

TRAINED_MODEL_PATH = os.path.join(DEFAULT_ARTIFACTS_DIR, "model.keras")
FINAL_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "model.keras")

DEFAULT_TRAIN_METRICS_PATH = os.path.join(DEFAULT_ARTIFACTS_DIR, "train_metrics.json")
DEFAULT_EVAL_METRICS_PATH = os.path.join(DEFAULT_ARTIFACTS_DIR, "eval_metrics.json")
