"""
Case 1 (CLASE_1) Real-World Microplastic Filter Extraction & Cleaning Pipeline
Streams all 3,000 real-world microplastic filter paper TIFFs directly from CLASE_1.zip,
converts them into clean, standardized RGB JPEGs, audits exposure and contrast,
and partitions them into Validation (500) and Test (2,500) benchmark suites.
Multi-threaded across CPU cores.
"""

import os
import io
import sys
import zipfile
import json
import csv
import time
import random
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image

CLEANED_ROOT = os.path.join("data", "cleaned")
BENCHMARK_DIR = os.path.join(CLEANED_ROOT, "benchmark_real")
VAL_DIR = os.path.join(BENCHMARK_DIR, "val")
TEST_DIR = os.path.join(BENCHMARK_DIR, "test")
MANIFEST_PATH = os.path.join(CLEANED_ROOT, "benchmark_real_manifest.csv")
REPORT_PATH = os.path.join(CLEANED_ROOT, "benchmark_real_report.json")
ZIP_PATH = os.path.join("data", "raw", "CLASE_1.zip")

thread_local = threading.local()

def get_zip_handle(zip_path: str):
    if not hasattr(thread_local, "zip_handle"):
        thread_local.zip_handle = zipfile.ZipFile(zip_path, "r")
    return thread_local.zip_handle


def process_real_filter(task):
    idx, entry_name, subset, zip_path, quality = task
    z = get_zip_handle(zip_path)
    try:
        raw_bytes = z.read(entry_name)
        with Image.open(io.BytesIO(raw_bytes)) as img:
            img.load()
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            w, h = img.size
            np_img = np.array(img)
            mean_val = float(np.mean(np_img))
            std_val = float(np.std(np_img))

            # Quality filters
            if mean_val < 10.0:
                return None, {"file": entry_name, "reason": f"Under-exposed / black frame (mean={mean_val:.1f})"}
            if std_val < 2.0:
                return None, {"file": entry_name, "reason": f"Zero contrast / solid color (std={std_val:.1f})"}

            filename = f"real_filter_{idx:05d}.jpg"
            target_folder = VAL_DIR if subset == "val" else TEST_DIR
            dst_path = os.path.join(target_folder, filename)
            img.save(dst_path, "JPEG", quality=quality)

            row = {
                "image_id": filename,
                "subset": subset,
                "original_entry": entry_name,
                "width": w,
                "height": h,
                "resolution": f"{w}x{h}",
                "mean_brightness": round(mean_val, 2),
                "std_brightness": round(std_val, 2)
            }
            return row, None
    except Exception as e:
        return None, {"file": entry_name, "reason": str(e)}


def extract_and_clean_case1(zip_path: str = ZIP_PATH, val_count: int = 500, quality: int = 95, num_workers: int = 12, seed: int = 42):
    start_time = time.time()
    random.seed(seed)
    
    if not os.path.exists(zip_path):
        print(f"[Error] Zip archive not found at: {zip_path}")
        sys.exit(1)

    os.makedirs(VAL_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    print(f"[Init] Reading {zip_path} archive index...")
    with zipfile.ZipFile(zip_path, "r") as z:
        all_entries = z.namelist()
        tif_entries = sorted([n for n in all_entries if n.lower().endswith((".tif", ".tiff"))])
        total_tifs = len(tif_entries)
        print(f"[Info] Found {total_tifs:,} TIF real-world microplastic filter papers in archive.")

    # Deterministic shuffle to split into validation (500) and test (2,500)
    shuffled = list(tif_entries)
    random.shuffle(shuffled)

    val_entries = set(shuffled[:val_count])
    
    tasks = []
    for i, entry in enumerate(tif_entries, 1):
        subset = "val" if entry in val_entries else "test"
        tasks.append((i, entry, subset, zip_path, quality))

    print(f"[Info] Partitioned into {val_count} Validation frames and {total_tifs - val_count} Test frames.")
    print(f"[Info] Starting parallel extraction across {num_workers} worker threads...")

    manifest_rows = []
    rejected = []
    processed_count = 0
    resolution_counts = {}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_real_filter, t): t for t in tasks}
        for future in as_completed(futures):
            processed_count += 1
            row, err = future.result()
            if row:
                manifest_rows.append(row)
                res = row["resolution"]
                resolution_counts[res] = resolution_counts.get(res, 0) + 1
            if err:
                rejected.append(err)

            if processed_count % 500 == 0 or processed_count == total_tifs:
                print(f"  Processed {processed_count:,}/{total_tifs:,} frames ({len(manifest_rows):,} valid)...")

    # Sort manifest rows by image_id
    manifest_rows.sort(key=lambda r: r["image_id"])

    # Write Manifest CSV
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["image_id", "subset", "original_entry", "width", "height", "resolution", "mean_brightness", "std_brightness"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    elapsed_time = round(time.time() - start_time, 2)

    # Write Summary Report JSON
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_archive": zip_path,
        "total_tifs_in_archive": total_tifs,
        "total_valid_saved": len(manifest_rows),
        "validation_frames": len([r for r in manifest_rows if r["subset"] == "val"]),
        "test_frames": len([r for r in manifest_rows if r["subset"] == "test"]),
        "resolution_distribution": resolution_counts,
        "total_rejected": len(rejected),
        "execution_time_seconds": elapsed_time,
        "validation_directory": os.path.abspath(VAL_DIR),
        "test_directory": os.path.abspath(TEST_DIR),
        "manifest_path": os.path.abspath(MANIFEST_PATH),
        "rejected_sample": rejected[:10]
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("COMPLETE CASE 1 EXTRACTION & CLEANING FINISHED")
    print("=" * 60)
    print(f"Total real-world filters saved : {len(manifest_rows):,}")
    print(f"  - Validation Benchmark       : {report['validation_frames']:,} frames")
    print(f"  - Held-Out Test Benchmark    : {report['test_frames']:,} frames")
    print(f"Resolution breakdown           : {resolution_counts}")
    print(f"Total frames rejected          : {len(rejected):,}")
    print(f"Time taken                     : {elapsed_time}s")
    print(f"Saved to directory             : {BENCHMARK_DIR}")
    print(f"Saved manifest to              : {MANIFEST_PATH}")
    print(f"Saved report to                : {REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract all real-world microplastic filter images from CLASE_1.zip")
    parser.add_argument("--val-count", type=int, default=500, help="Number of frames for validation benchmark")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality for saved images")
    parser.add_argument("--workers", type=int, default=12, help="Number of parallel worker threads")
    args = parser.parse_args()

    extract_and_clean_case1(val_count=args.val_count, quality=args.quality, num_workers=args.workers)
