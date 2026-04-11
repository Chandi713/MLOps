import os
import sys

import yaml

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import LOCAL_PATHS, NUM_CLASSES, YOLO_DATASET_DIR, class_names
from src.utils.utils import get_disk_free_gb


def labels_exist() -> bool:
    """Check whether the label .txt files (the real conversion output) exist."""
    for split in ["train", "val"]:
        ldir = os.path.join(YOLO_DATASET_DIR, "labels", split)
        if not os.path.exists(ldir) or not any(f.endswith(".txt") for f in os.listdir(ldir)):
            return False
    return True


def test_labels_exist() -> bool:
    """Check whether test split labels have been created."""
    ldir = os.path.join(YOLO_DATASET_DIR, "labels", "test")
    return os.path.exists(ldir) and any(f.endswith(".txt") for f in os.listdir(ldir))


def symlinks_valid() -> bool:
    """Check whether image symlinks exist and their targets are reachable."""
    for split in ["train", "val"]:
        idir = os.path.join(YOLO_DATASET_DIR, "images", split)
        if not os.path.exists(idir):
            return False
        if not any(
            os.path.exists(os.path.join(idir, f))
            for f in os.listdir(idir)
            if f.endswith((".jpg", ".jpeg", ".png"))
        ):
            return False
    # Also check test if it has been created
    if test_labels_exist():
        idir = os.path.join(YOLO_DATASET_DIR, "images", "test")
        if not os.path.exists(idir):
            return False
        if not any(
            os.path.exists(os.path.join(idir, f))
            for f in os.listdir(idir)
            if f.endswith((".jpg", ".jpeg", ".png"))
        ):
            return False
    return True


def restore_symlinks():
    """Re-create broken image symlinks from existing label files. No re-conversion needed."""
    from src.data.data_loader import loader

    for split in ["train", "val"]:
        idir = os.path.join(YOLO_DATASET_DIR, "images", split)
        ldir = os.path.join(YOLO_DATASET_DIR, "labels", split)
        os.makedirs(idir, exist_ok=True)
        src_index = loader.build_image_index(LOCAL_PATHS[f"{split}_images"])
        label_stems = {os.path.splitext(f)[0] for f in os.listdir(ldir) if f.endswith(".txt")}
        fnames = [fn for fn in src_index if os.path.splitext(fn)[0] in label_stems]
        linked, _ = loader.link_images(fnames, src_index, idir)
        print(f"  {split}: {linked} symlinks restored.")

    # Restore test symlinks (test images live in train_images source dir)
    _test_list = os.path.join(YOLO_DATASET_DIR, "test_filenames.txt")
    if os.path.exists(_test_list) and test_labels_exist():
        with open(_test_list) as _f:
            test_fnames_set = set(_f.read().splitlines())
        idir = os.path.join(YOLO_DATASET_DIR, "images", "test")
        ldir = os.path.join(YOLO_DATASET_DIR, "labels", "test")
        os.makedirs(idir, exist_ok=True)
        src_index   = loader.build_image_index(LOCAL_PATHS["train_images"])
        label_stems = {os.path.splitext(f)[0] for f in os.listdir(ldir) if f.endswith(".txt")}
        fnames      = [fn for fn in src_index
                       if os.path.splitext(fn)[0] in label_stems and fn in test_fnames_set]
        linked, _   = loader.link_images(fnames, src_index, idir)
        print(f"  test: {linked} symlinks restored.")


def verify_dataset():
    print("\nYOLO dataset:")
    for split in ["train", "val", "test"]:
        idir = os.path.join(YOLO_DATASET_DIR, "images", split)
        ldir = os.path.join(YOLO_DATASET_DIR, "labels", split)
        ni = len([f for f in os.listdir(idir) if f.endswith((".jpg", ".jpeg", ".png"))]) if os.path.exists(idir) else 0
        nl = len([f for f in os.listdir(ldir) if f.endswith(".txt")]) if os.path.exists(ldir) else 0
        print(f"  {split:6s}: {ni:,} images, {nl:,} labels")
    print(f"\nDisk free: {get_disk_free_gb(YOLO_DATASET_DIR):.1f} GB")


def create_dataset_yaml() -> str:
    yaml_config = {
        "path":  os.path.abspath(YOLO_DATASET_DIR),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    NUM_CLASSES,
        "names": class_names,
    }
    yaml_path = os.path.join(YOLO_DATASET_DIR, "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_config, f, default_flow_style=False)
    print(f"YAML: {yaml_path}")
    with open(yaml_path) as f:
        print(f.read())
    return yaml_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify YOLO dataset or generate dataset YAML.")
    parser.add_argument("--action", required=True, choices=["verify", "create-yaml"],
                        help="'verify' — print image/label counts. 'create-yaml' — write dataset.yaml.")
    args = parser.parse_args()

    if args.action == "verify":
        verify_dataset()
    elif args.action == "create-yaml":
        create_dataset_yaml()
