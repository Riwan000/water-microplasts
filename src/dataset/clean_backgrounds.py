"""
High-Speed Background Extraction & Cleaning Pipeline (Multi-Threaded)
Streams and converts uncompressed TIFF filter paper backgrounds directly from CLASE_0.zip,
validates brightness and contrast, and saves clean, lightweight JPEGs to data/cleaned/backgrounds/.
Supports parallel extraction across multi-core CPUs.
"""

import os
import io
import sys
import zipfile
import json
import csv
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image

CLEANED_ROOT = os.path.join("data", "cleaned")
BACKGROUNDS_DIR = os.path.join(CLEANED_ROOT, "backgrounds")
MANIFEST_PATH = os.path.join(CLEANED_ROOT, "backgrounds_manifest.csv")
REPORT_PATH = os.path.join(CLEANED_ROOT, "backgrounds_report.json")
ZIP_PATH = os.path.join("data", "raw", "CLASE_0.zip")

thread_local = threading.local()

def get_zip_handle(zip_path: str):
    if not hasattr(thread_local, "zip_handle"):
        thread_local.zip_handle = zipfile.ZipFile(zip_path, "r")
    return thread_local.zip_handle


def process_single_bg(task):
    idx, entry_name, zip_path, quality = task
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

            bg_filename = f"bg_{idx:05d}.jpg"
            dst_path = os.path.join(BACKGROUNDS_DIR, bg_filename)
            img.save(dst_path, "JPEG", quality=quality)

            row = {
                "bg_id": bg_filename,
                "original_entry": entry_name,
                "width": w,
                "height": h,
                "mean_brightness": round(mean_val, 2),
                "std_brightness": round(std_val, 2)
            }
            return row, None
    except Exception as e:
        return None, {"file": entry_name, "reason": str(e)}


def extract_and_clean_backgrounds(zip_path: str = ZIP_PATH, target_count: int = 3000, quality: int = 95, num_workers: int = 12):
    start_time = time.time()
    
    if not os.path.exists(zip_path):
        print(f"[Error] Zip archive not found at: {zip_path}")
        sys.exit(1)

    os.makedirs(BACKGROUNDS_DIR, exist_ok=True)

    print(f"[Init] Reading {zip_path} archive index...")
    with zipfile.ZipFile(zip_path, "r") as z:
        all_entries = z.namelist()
        tif_entries = sorted([n for n in all_entries if n.lower().endswith((".tif", ".tiff"))])
        total_tifs = len(tif_entries)
        print(f"[Info] Found {total_tifs:,} TIF filter paper backgrounds in archive.")

        if target_count < total_tifs:
            step = total_tifs / target_count
            selected_indices = [int(i * step) for i in range(target_count)]
            selected_entries = [tif_entries[i] for i in selected_indices]
        else:
            selected_entries = tif_entries

    total_selected = len(selected_entries)
    print(f"[Info] Selected all {total_selected:,} frames for extraction and validation using {num_workers} parallel workers...")

    tasks = [(i + 1, entry, zip_path, quality) for i, entry in enumerate(selected_entries)]
    manifest_rows = []
    rejected = []
    processed_count = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_bg, t): t for t in tasks}
        for future in as_completed(futures):
            processed_count += 1
            row, err = future.result()
            if row:
                manifest_rows.append(row)
            if err:
                rejected.append(err)

            if processed_count % 500 == 0 or processed_count == total_selected:
                print(f"  Processed {processed_count:,}/{total_selected:,} backgrounds ({len(manifest_rows):,} valid)...")

    # Sort manifest rows by bg_id for clean ordering
    manifest_rows.sort(key=lambda r: r["bg_id"])

    # Write Manifest CSV
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["bg_id", "original_entry", "width", "height", "mean_brightness", "std_brightness"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    elapsed_time = round(time.time() - start_time, 2)

    # Write Summary Report JSON
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_archive": zip_path,
        "total_tifs_in_archive": total_tifs,
        "selected_for_processing": total_selected,
        "total_valid_saved": len(manifest_rows),
        "total_rejected": len(rejected),
        "execution_time_seconds": elapsed_time,
        "output_directory": os.path.abspath(BACKGROUNDS_DIR),
        "manifest_path": os.path.abspath(MANIFEST_PATH),
        "rejected_sample": rejected[:10]
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("COMPLETE CASE 0 BACKGROUND EXTRACTION FINISHED")
    print("=" * 60)
    print(f"Total backgrounds saved : {len(manifest_rows):,}")
    print(f"Total frames rejected   : {len(rejected):,}")
    print(f"Time taken              : {elapsed_time}s")
    print(f"Saved to directory      : {BACKGROUNDS_DIR}")
    print(f"Saved manifest to       : {MANIFEST_PATH}")
    print(f"Saved report to         : {REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract all background filter paper images from CLASE_0.zip")
    parser.add_argument("--count", type=int, default=3000, help="Number of backgrounds to extract (3000 for all)")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality for saved backgrounds")
    parser.add_argument("--workers", type=int, default=12, help="Number of parallel worker threads")
    args = parser.parse_args()

    extract_and_clean_backgrounds(target_count=args.count, quality=args.quality, num_workers=args.workers)
