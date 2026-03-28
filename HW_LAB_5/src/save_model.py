"""
Copy trained Keras model from artifacts/ to models/ for release.
"""

from __future__ import annotations

import argparse
import os
import shutil

from .constants import FINAL_MODEL_PATH, TRAINED_MODEL_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy trained model to models/.")
    parser.add_argument("--src", default=TRAINED_MODEL_PATH)
    parser.add_argument("--dest", default=FINAL_MODEL_PATH)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.dest)) or ".", exist_ok=True)
    shutil.copy2(args.src, args.dest)
    print(f"Copied {args.src} -> {args.dest}")


if __name__ == "__main__":
    main()
