import os
from typing import Dict, List, Tuple

import pandas as pd
from tqdm.auto import tqdm

from config import CSV_BATCH_SIZE, IMAGE_DOWNLOAD_BATCH, MIN_DISK_FREE_GB, YOLO_DATASET_DIR
from src.utils.utils import get_disk_free_gb


class LocalDataLoader:
    """
    Memory-efficient data loader that reads from the local filesystem.

    - CSV metadata is streamed in chunks (bounded RAM)
    - Images are symlinked per-chunk (zero extra disk space)
    - Disk space checked before each batch (safe stop)
    """

    def __init__(self, csv_batch_size=20000, image_batch=200, min_disk_free_gb=5.0):
        self.csv_batch_size   = csv_batch_size
        self.image_batch      = image_batch
        self.min_disk_free_gb = min_disk_free_gb

    def stream_csv_batches(self, csv_path: str):
        """Read local CSV in chunks; yield (batch_num, DataFrame)."""
        print(f"  Reading CSV: {csv_path}")
        size_mb = os.path.getsize(csv_path) / 1024**2
        total_rows = sum(1 for _ in open(csv_path)) - 1  # fast line count
        total_batches = (total_rows // self.csv_batch_size) + 1
        print(f"  CSV size: {size_mb:.1f} MB | ~{total_rows:,} rows | {total_batches} batches")

        batch_num = 0
        for chunk in tqdm(
            pd.read_csv(csv_path, chunksize=self.csv_batch_size),
            total=total_batches,
            desc="  CSV batches",
            unit="batch",
        ):
            batch_num += 1
            yield batch_num, chunk

        print(f"  Streamed {batch_num} CSV batches")

    def load_csv(self, csv_path: str) -> pd.DataFrame:
        """Load full CSV into DataFrame (for exploration)."""
        print(f"  Loading: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df):,} rows ({df.memory_usage(deep=True).sum()/1024**2:.1f} MB)")
        return df

    def build_image_index(self, images_dir: str) -> Dict[str, str]:
        """Return {filename: full_path} for all images in a directory."""
        images_dir = os.path.abspath(images_dir)
        print(f"  Indexing: {images_dir}")
        index = {
            fname: os.path.join(images_dir, fname)
            for fname in os.listdir(images_dir)
            if fname.lower().endswith((".jpg", ".jpeg", ".png"))
        }
        print(f"  Found {len(index):,} images")
        return index

    def link_images(self, filenames: List[str], src_index: Dict[str, str],
                    dest_dir: str) -> Tuple[int, bool]:
        """
        Symlink images into dest_dir.
        Returns (count_linked, disk_ok).
        """
        os.makedirs(dest_dir, exist_ok=True)
        linked = 0

        to_link = [
            fn for fn in filenames
            if fn in src_index and not os.path.exists(os.path.join(dest_dir, fn))
        ]

        for i in range(0, len(to_link), self.image_batch):
            free_gb = get_disk_free_gb(YOLO_DATASET_DIR)
            if free_gb < self.min_disk_free_gb:
                print(f"\n  ⚠ Disk low ({free_gb:.1f} GB). Stopping.")
                return linked, False

            for fn in to_link[i : i + self.image_batch]:
                dest = os.path.join(dest_dir, fn)
                try:
                    os.symlink(src_index[fn], dest)
                    linked += 1
                except OSError as e:
                    print(f"  Warning: {fn}: {e}")

        return linked, True


loader = LocalDataLoader(
    csv_batch_size=CSV_BATCH_SIZE,
    image_batch=IMAGE_DOWNLOAD_BATCH,
    min_disk_free_gb=MIN_DISK_FREE_GB,
)
