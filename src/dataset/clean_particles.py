"""
Dataset Cleaning & Harmonization Pipeline for Particle Crops
Processes Dataset 1 (26511253) and Dataset 2 (Microplastics and algae),
removes system junk, validates image integrity, standardizes classes,
and outputs clean, verified images into a dedicated cleaned directory.
"""

import os
import sys
import glob
import json
import csv
import time
from pathlib import Path
from PIL import Image

# Output directories
CLEANED_ROOT = os.path.join("data", "cleaned")
PARTICLES_DIR = os.path.join(CLEANED_ROOT, "particles")
MANIFEST_PATH = os.path.join(CLEANED_ROOT, "particles_manifest.csv")
REPORT_PATH = os.path.join(CLEANED_ROOT, "cleaning_report.json")

# Valid image extensions
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Files and patterns to strictly ignore
IGNORED_FILENAMES = {".ds_store", "desktop.ini", "thumbs.db"}


def load_exclusion_list(to_remove_path: str) -> set:
    """Load filenames marked for removal in to_remove.txt."""
    excluded = set()
    if os.path.exists(to_remove_path):
        with open(to_remove_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    excluded.add(os.path.basename(line).lower())
    return excluded


def process_image(src_path: str, min_size: int = 5):
    """
    Validates, loads, and converts an image to RGB.
    Returns (PIL.Image, width, height) or None if invalid.
    """
    try:
        with Image.open(src_path) as img:
            img.load()  # Force decode to catch corrupted image data
            
            # Check dimensions
            w, h = img.size
            if w < min_size or h < min_size:
                return None, f"Image too small ({w}x{h} < {min_size}px)"
            
            # Convert to standard RGB
            if img.mode == "RGBA":
                # Composite onto clean white background
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3])
                rgb_img = bg
            elif img.mode != "RGB":
                rgb_img = img.convert("RGB")
            else:
                rgb_img = img.copy()

            return rgb_img, (w, h)
    except Exception as e:
        return None, str(e)


def run_cleaning():
    start_time = time.time()
    os.makedirs(PARTICLES_DIR, exist_ok=True)

    # 1. Load explicit exclusions from to_remove.txt in Dataset 1
    to_remove_path = os.path.join("data", "raw", "26511253", "MICRO", "MICRO", "to_remove.txt")
    excluded_basenames = load_exclusion_list(to_remove_path)
    print(f"[Init] Loaded {len(excluded_basenames)} excluded filenames from to_remove.txt")

    # 2. Define source folders and category mapping
    # Format: (source_dataset_name, folder_path, target_class)
    job_mappings = [
        # Dataset 1: 26511253 / MICRO / MICRO
        ("26511253_micro", os.path.join("data", "raw", "26511253", "MICRO", "MICRO", "line"), "fibre"),
        ("26511253_micro", os.path.join("data", "raw", "26511253", "MICRO", "MICRO", "hard"), "fragment"),
        ("26511253_micro", os.path.join("data", "raw", "26511253", "MICRO", "MICRO", "pellet"), "pellet"),
        ("26511253_micro", os.path.join("data", "raw", "26511253", "MICRO", "MICRO", "foam"), "foam_film"),
        ("26511253_micro", os.path.join("data", "raw", "26511253", "MICRO", "MICRO", "noise"), "noise"),

        # Dataset 1: 26511253 / VALIDATION / VALIDATION / MICRO
        ("26511253_val", os.path.join("data", "raw", "26511253", "VALIDATION", "VALIDATION", "MICRO", "line"), "fibre"),
        ("26511253_val", os.path.join("data", "raw", "26511253", "VALIDATION", "VALIDATION", "MICRO", "hard"), "fragment"),
        ("26511253_val", os.path.join("data", "raw", "26511253", "VALIDATION", "VALIDATION", "MICRO", "pellet"), "pellet"),
        ("26511253_val", os.path.join("data", "raw", "26511253", "VALIDATION", "VALIDATION", "MICRO", "foam"), "foam_film"),
        ("26511253_val", os.path.join("data", "raw", "26511253", "VALIDATION", "VALIDATION", "MICRO", "film"), "foam_film"),

        # Dataset 2: Microplastics and algae
        ("algae_dataset", os.path.join("data", "raw", "Microplastic images and Automatic classification by deep learning", "Microplastic images and Automatic classification by deep learning", "Microplastics and algae", "filament"), "fibre"),
        ("algae_dataset", os.path.join("data", "raw", "Microplastic images and Automatic classification by deep learning", "Microplastic images and Automatic classification by deep learning", "Microplastics and algae", "fragment"), "fragment"),
        ("algae_dataset", os.path.join("data", "raw", "Microplastic images and Automatic classification by deep learning", "Microplastic images and Automatic classification by deep learning", "Microplastics and algae", "pellet"), "pellet"),
        ("algae_dataset", os.path.join("data", "raw", "Microplastic images and Automatic classification by deep learning", "Microplastic images and Automatic classification by deep learning", "Microplastics and algae", "algae I"), "algae"),
    ]

    manifest_rows = []
    class_counts = {}
    source_counts = {}
    rejections = []
    class_indices = {}

    print("\nStarting validation and cleaning across all particle sources...")

    for dataset_tag, folder_path, target_class in job_mappings:
        if not os.path.exists(folder_path):
            print(f"[Warning] Folder not found: {folder_path}")
            continue

        target_dir = os.path.join(PARTICLES_DIR, target_class)
        os.makedirs(target_dir, exist_ok=True)

        class_counts.setdefault(target_class, 0)
        source_counts.setdefault(dataset_tag, 0)
        class_indices.setdefault(target_class, 0)

        entries = sorted(os.listdir(folder_path))
        for entry in entries:
            src_file = os.path.join(folder_path, entry)
            if not os.path.isfile(src_file):
                continue

            fname_lower = entry.lower()
            base, ext = os.path.splitext(fname_lower)

            # Skip system junk
            if entry.lower() in IGNORED_FILENAMES or entry.startswith("."):
                rejections.append({"file": src_file, "reason": "Ignored OS/system file"})
                continue

            # Skip non-image extensions
            if ext not in VALID_EXTENSIONS:
                rejections.append({"file": src_file, "reason": f"Unsupported extension: {ext}"})
                continue

            # Check explicit exclusions
            if fname_lower in excluded_basenames:
                rejections.append({"file": src_file, "reason": "Explicitly listed in to_remove.txt"})
                continue

            # Process and validate image
            rgb_img, result = process_image(src_file)
            if rgb_img is None:
                rejections.append({"file": src_file, "reason": f"Validation failed: {result}"})
                continue

            w, h = result
            aspect_ratio = round(w / h, 4) if h > 0 else 1.0

            # Save clean standardized image
            class_indices[target_class] += 1
            idx = class_indices[target_class]
            dst_filename = f"{target_class}_{dataset_tag}_{idx:05d}.jpg"
            dst_path = os.path.join(target_dir, dst_filename)

            rgb_img.save(dst_path, "JPEG", quality=95)

            # Record in manifest
            manifest_rows.append({
                "particle_id": dst_filename,
                "target_class": target_class,
                "source_dataset": dataset_tag,
                "source_relpath": os.path.relpath(src_file),
                "width": w,
                "height": h,
                "aspect_ratio": aspect_ratio
            })

            class_counts[target_class] += 1
            source_counts[dataset_tag] += 1

    # 3. Write Manifest CSV
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["particle_id", "target_class", "source_dataset", "source_relpath", "width", "height", "aspect_ratio"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    elapsed_time = round(time.time() - start_time, 2)

    # 4. Write Summary Report JSON
    total_valid = len(manifest_rows)
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_particles_cleaned": total_valid,
        "total_rejected": len(rejections),
        "execution_time_seconds": elapsed_time,
        "class_breakdown": class_counts,
        "source_dataset_breakdown": source_counts,
        "rejection_sample": rejections[:20],
        "output_directory": os.path.abspath(PARTICLES_DIR),
        "manifest_path": os.path.abspath(MANIFEST_PATH)
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print user-facing report
    print("\n" + "=" * 60)
    print("CLEANING & STANDARDIZATION COMPLETE")
    print("=" * 60)
    print(f"Total valid particles saved: {total_valid:,}")
    print(f"Total files skipped/rejected: {len(rejections):,}")
    print(f"Execution time: {elapsed_time}s\n")
    print("Cleaned Class Breakdown:")
    for cls_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {cls_name:<12}: {count:,} particles")
    print(f"\nSaved master manifest to: {MANIFEST_PATH}")
    print(f"Saved audit report to   : {REPORT_PATH}")
    print(f"All images written to   : {PARTICLES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    run_cleaning()
