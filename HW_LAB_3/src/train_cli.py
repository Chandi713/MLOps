#!/usr/bin/env python
"""CLI for training polynomial regression models."""
import argparse
import sys
import os
import json
from datetime import datetime

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

from data import generate_data
from train import train_model
from predict import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a polynomial regression model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--points", "-p",
        type=int,
        default=int(os.getenv("DEFAULT_POINTS", 80)),
        help="Number of training data points"
    )
    
    parser.add_argument(
        "--degree", "-d",
        type=int,
        default=int(os.getenv("DEFAULT_DEGREE", 2)),
        help="Polynomial degree"
    )
    
    parser.add_argument(
        "--timestamp", "-t",
        type=str,
        default=None,
        help="Timestamp for versioning (auto-generated if not provided)"
    )
    
    parser.add_argument(
        "--evaluate", "-e",
        action="store_true",
        help="Run evaluation after training"
    )
    
    parser.add_argument(
        "--save-metrics",
        action="store_true",
        default=True,
        help="Save metrics to JSON file"
    )
    
    parser.add_argument(
        "--save-to-gcs",
        action="store_true",
        help="Upload model to Google Cloud Storage"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed output"
    )
    
    return parser.parse_args()


def save_metrics_to_json(metrics, timestamp, degree, points):
    os.makedirs("metrics", exist_ok=True)
    
    metrics_data = {
        "timestamp": timestamp,
        "model_config": {
            "degree": degree,
            "training_points": points
        },
        "training": {
            "mse": float(metrics["train_mse"]),
            "r2": float(metrics["train_r2"])
        },
        "testing": {
            "mse": float(metrics["test_mse"]),
            "r2": float(metrics["test_r2"])
        }
    }
    
    filename = f"metrics/{timestamp}_degree{degree}_metrics.json"
    with open(filename, 'w') as f:
        json.dump(metrics_data, f, indent=4)
    
    return filename


def upload_to_gcs(model, degree, timestamp):
    from gcs_utils import save_model_to_gcs, get_model_version, update_model_version

    bucket_name = os.getenv("GCS_BUCKET_NAME")
    version_file = os.getenv("VERSION_FILE_NAME", "model_version.txt")

    if not bucket_name:
        print("GCS_BUCKET_NAME not set in .env file")
        return None

    try:
        current_version = get_model_version(bucket_name, version_file)
        new_version = current_version + 1
        blob_name = save_model_to_gcs(model, bucket_name, degree, new_version)
        update_model_version(bucket_name, version_file, new_version)
        print(f"Model uploaded to GCS: gs://{bucket_name}/{blob_name}")
        print(f"Version updated: {current_version} -> {new_version}")
        return blob_name
    except Exception as e:
        print(f"GCS upload failed: {e}")
        return None


def main():
    args = parse_args()
    
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d%H%M%S")
    
    if args.verbose:
        print(f"Training Configuration:")
        print(f"   Points: {args.points}")
        print(f"   Degree: {args.degree}")
        print(f"   Timestamp: {timestamp}")
        print(f"   Save to GCS: {args.save_to_gcs}")
        print()

    x_train, y_train, x_test, y_test = generate_data(args.points)

    if args.verbose:
        print(f"Data generated: {len(x_train)} training, {len(x_test)} test points")

    model = train_model(x_train, y_train, args.degree)
    print(f"Model trained: degree={args.degree}, status={model.status}")

    if args.evaluate:
        metrics = evaluate_model(model, x_train, y_train, x_test, y_test)
        print(f"\nEvaluation Results:")
        print(f"   Train MSE: {metrics['train_mse']:.6f}")
        print(f"   Train R²:  {metrics['train_r2']:.6f}")
        print(f"   Test MSE:  {metrics['test_mse']:.6f}")
        print(f"   Test R²:   {metrics['test_r2']:.6f}")
        if args.save_metrics:
            metrics_file = save_metrics_to_json(metrics, timestamp, args.degree, args.points)
            print(f"\nMetrics saved to: {metrics_file}")

    if args.save_to_gcs:
        print(f"\nUploading to Google Cloud Storage...")
        upload_to_gcs(model, args.degree, timestamp)

    print(f"\nTraining complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
