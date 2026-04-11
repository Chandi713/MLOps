import gc
import os
import random
import sys
import time
from ast import literal_eval

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import NUM_CLASSES, TEST_SPLIT_RATIO, YOLO_DATASET_DIR
from src.data.data_loader import loader
from src.utils.utils import get_disk_free_gb

_TEST_FILENAMES_PATH = os.path.join(YOLO_DATASET_DIR, "test_filenames.txt")


def carve_test_split(train_csv_path: str, test_ratio: float = None, seed: int = 42):
    """Split training images into train/test by filename (deterministic, run once)."""
    if test_ratio is None:
        test_ratio = TEST_SPLIT_RATIO

    print("Reading unique image filenames from training CSV...")
    all_fnames = set()
    for _, chunk_df in loader.stream_csv_batches(train_csv_path):
        all_fnames.update(os.path.basename(p) for p in chunk_df["path"])
        del chunk_df
        gc.collect()

    all_fnames = sorted(all_fnames)
    random.seed(seed)
    random.shuffle(all_fnames)

    n_test = max(1, int(len(all_fnames) * test_ratio))
    test_fnames  = set(all_fnames[:n_test])
    train_fnames = set(all_fnames[n_test:])

    with open(_TEST_FILENAMES_PATH, "w") as f:
        f.write("\n".join(sorted(test_fnames)))

    print(f"Test split  : {len(test_fnames):,} images ({test_ratio:.0%})")
    print(f"Train retain: {len(train_fnames):,} images")
    print(f"Saved to    : {_TEST_FILENAMES_PATH}")
    return train_fnames, test_fnames


def load_test_filenames():
    """Load the persisted test filename set, or None if not yet carved."""
    if not os.path.exists(_TEST_FILENAMES_PATH):
        return None
    with open(_TEST_FILENAMES_PATH) as f:
        return set(f.read().splitlines())


def convert_bbox_to_yolo(bbox_str, img_w, img_h):
    try:
        bbox = literal_eval(bbox_str) if isinstance(bbox_str, str) else bbox_str
        x1, y1, x2, y2 = bbox
        x1, x2 = max(0, min(x1, img_w)), max(0, min(x2, img_w))
        y1, y2 = max(0, min(y1, img_h)), max(0, min(y2, img_h))
        w, h = (x2 - x1) / img_w, (y2 - y1) / img_h
        if w <= 0 or h <= 0:
            return None
        return [((x1 + x2) / 2) / img_w, ((y1 + y2) / 2) / img_h, w, h]
    except:
        return None


def convert_split(split, csv_path, images_dir, allowed_fnames=None):
    img_dir = os.path.join(YOLO_DATASET_DIR, "images", split)
    lbl_dir = os.path.join(YOLO_DATASET_DIR, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    stats = dict(total=0, converted=0, skipped=0, linked=0,
                 cached=0, labels=0, batches=0, disk_stop=False)

    src_index = loader.build_image_index(images_dir)

    print(f"\nConverting {split} in batches...")
    print("=" * 60)

    for batch_num, chunk_df in loader.stream_csv_batches(csv_path):
        stats["batches"] = batch_num
        t0 = time.time()

        # Filter to allowed filenames when carving a subset (train or test)
        if allowed_fnames is not None:
            chunk_df = chunk_df[chunk_df["path"].apply(os.path.basename).isin(allowed_fnames)]
            if chunk_df.empty:
                gc.collect()
                continue

        fnames = [os.path.basename(p) for p in chunk_df["path"].unique()]
        cached = sum(1 for fn in fnames if os.path.exists(os.path.join(img_dir, fn)))
        stats["cached"] += cached

        n_linked, disk_ok = loader.link_images(fnames, src_index, img_dir)
        stats["linked"] += n_linked

        if not disk_ok:
            stats["disk_stop"] = True
            print(f"  Batch {batch_num}: DISK FULL — stopping. Will train on partial data.")
            del chunk_df
            gc.collect()
            break

        # Annotation conversion
        for path, group in chunk_df.groupby("path"):
            fn = os.path.basename(path)
            if not os.path.exists(os.path.join(img_dir, fn)):
                stats["skipped"] += len(group)
                continue

            lines = []
            for _, row in group.iterrows():
                stats["total"] += 1
                cid = int(row["category_id"]) - 1
                if cid < 0 or cid >= NUM_CLASSES:
                    stats["skipped"] += 1
                    continue
                yolo = convert_bbox_to_yolo(
                    row["b_box"],
                    int(row.get("img_width", 640)),
                    int(row.get("img_height", 640)),
                )
                if yolo is None:
                    stats["skipped"] += 1
                    continue
                lines.append(
                    f"{cid} {yolo[0]:.6f} {yolo[1]:.6f} {yolo[2]:.6f} {yolo[3]:.6f}"
                )
                stats["converted"] += 1

            if lines:
                lbl_path = os.path.join(lbl_dir, os.path.splitext(fn)[0] + ".txt")
                mode = "a" if os.path.exists(lbl_path) else "w"
                with open(lbl_path, mode) as f:
                    f.write("\n".join(lines) + "\n")
                stats["labels"] += 1

        free = get_disk_free_gb(YOLO_DATASET_DIR)
        print(
            f"  Batch {batch_num}: {len(chunk_df):,} annot | "
            f"+{n_linked} linked | {time.time()-t0:.1f}s | {free:.1f} GB free"
        )
        del chunk_df
        gc.collect()

    print(f"\n{'='*60}")
    print(
        f"{split.upper()} DONE — {stats['converted']:,}/{stats['total']:,} annotations, "
        f"{stats['linked']:,} linked, {stats['labels']:,} label files"
    )
    if stats["disk_stop"]:
        print("  ⚠ Stopped early due to disk space. Training will use partial data.")
    print("=" * 60)
    return stats


if __name__ == "__main__":
    import argparse
    from config import LOCAL_PATHS

    parser = argparse.ArgumentParser(description="Convert DeepFashion2 split to YOLO format.")
    parser.add_argument("--split", required=True, choices=["train", "val"],
                        help="Dataset split to convert.")
    args = parser.parse_args()

    convert_split(
        args.split,
        LOCAL_PATHS[f"{args.split}_csv"],
        LOCAL_PATHS[f"{args.split}_images"],
    )
