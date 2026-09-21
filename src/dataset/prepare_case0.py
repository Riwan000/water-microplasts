"""
Case 0 (CLASE_0) Background Partitioning & Negative Sample Generator
Splits the 500 cleaned CLASE_0 filter paper images into:
1. Canvas Pool (400 frames) - Used by the Scene Compositor to paste plastic particles onto.
2. Negative Controls Pool (100 frames) - Preserved with ZERO plastic particles.
   Generates 640x640 negative training patches with empty YOLO annotation files
   to train the model to ignore paper grain, printed grid lines, and mineral silt.
"""

import os
import shutil
import json
import random
from PIL import Image

CLEANED_DIR = os.path.join("data", "cleaned")
RAW_BG_DIR = os.path.join(CLEANED_DIR, "backgrounds")
CANVAS_DIR = os.path.join(CLEANED_DIR, "canvas_backgrounds")
NEG_POOL_DIR = os.path.join(CLEANED_DIR, "negative_backgrounds")
NEG_TILES_IMG_DIR = os.path.join(CLEANED_DIR, "negatives", "images")
NEG_TILES_LBL_DIR = os.path.join(CLEANED_DIR, "negatives", "labels")
REPORT_PATH = os.path.join(CLEANED_DIR, "case0_prep_report.json")

PATCH_SIZE = 640
NUM_NEGATIVES = 500
SEED = 42


def prepare_case0(num_negatives: int = NUM_NEGATIVES, patch_size: int = PATCH_SIZE):
    random.seed(SEED)
    
    if not os.path.exists(RAW_BG_DIR):
        raise FileNotFoundError(f"Backgrounds directory not found: {RAW_BG_DIR}")

    bg_files = sorted([f for f in os.listdir(RAW_BG_DIR) if f.lower().endswith((".jpg", ".jpeg"))])
    total_bg = len(bg_files)
    print(f"[Init] Found {total_bg} cleaned backgrounds in {RAW_BG_DIR}")

    # Deterministic shuffle to split into canvas and negatives
    shuffled = list(bg_files)
    random.shuffle(shuffled)

    neg_files = sorted(shuffled[:num_negatives])
    canvas_files = sorted(shuffled[num_negatives:])

    print(f"[Split] Allocating {len(canvas_files)} frames for Particle Compositing Canvas")
    print(f"[Split] Allocating {len(neg_files)} frames for Pure Negative Controls")

    # Create directories
    os.makedirs(CANVAS_DIR, exist_ok=True)
    os.makedirs(NEG_POOL_DIR, exist_ok=True)
    os.makedirs(NEG_TILES_IMG_DIR, exist_ok=True)
    os.makedirs(NEG_TILES_LBL_DIR, exist_ok=True)

    # 1. Populate Canvas Backgrounds
    for f in canvas_files:
        src = os.path.join(RAW_BG_DIR, f)
        dst = os.path.join(CANVAS_DIR, f)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    # 2. Populate Negative Backgrounds & generate 640x640 negative patches
    neg_patch_count = 0
    for idx, f in enumerate(neg_files, 1):
        src = os.path.join(RAW_BG_DIR, f)
        dst = os.path.join(NEG_POOL_DIR, f)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

        # Crop two distinct 640x640 patches from each full 2040x1528 frame
        # Crop 1: left quadrant; Crop 2: right quadrant
        with Image.open(src) as img:
            w, h = img.size
            
            # Patch A (left/center)
            x_a = (w - 2 * patch_size) // 3
            y_a = (h - patch_size) // 2
            crop_a = img.crop((x_a, y_a, x_a + patch_size, y_a + patch_size))
            
            neg_patch_count += 1
            name_a = f"neg_bg_{neg_patch_count:05d}.jpg"
            crop_a.save(os.path.join(NEG_TILES_IMG_DIR, name_a), "JPEG", quality=95)
            # Create corresponding empty YOLO label file
            with open(os.path.join(NEG_TILES_LBL_DIR, f"neg_bg_{neg_patch_count:05d}.txt"), "w") as lf:
                pass  # Empty file instructs YOLO: no objects in this image

            # Patch B (right/center)
            x_b = w - patch_size - x_a
            y_b = (h - patch_size) // 2
            crop_b = img.crop((x_b, y_b, x_b + patch_size, y_b + patch_size))
            
            neg_patch_count += 1
            name_b = f"neg_bg_{neg_patch_count:05d}.jpg"
            crop_b.save(os.path.join(NEG_TILES_IMG_DIR, name_b), "JPEG", quality=95)
            with open(os.path.join(NEG_TILES_LBL_DIR, f"neg_bg_{neg_patch_count:05d}.txt"), "w") as lf:
                pass

    report = {
        "total_source_backgrounds": total_bg,
        "canvas_backgrounds_count": len(canvas_files),
        "canvas_directory": os.path.abspath(CANVAS_DIR),
        "negative_backgrounds_pool": len(neg_files),
        "negative_pool_directory": os.path.abspath(NEG_POOL_DIR),
        "generated_negative_patches": neg_patch_count,
        "patch_resolution": f"{patch_size}x{patch_size}",
        "negative_images_dir": os.path.abspath(NEG_TILES_IMG_DIR),
        "negative_labels_dir": os.path.abspath(NEG_TILES_LBL_DIR),
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("CASE 0 PARTITIONING & NEGATIVE PATCH GENERATION COMPLETE")
    print("=" * 60)
    print(f"Canvas Backgrounds for Compositing : {len(canvas_files)} frames")
    print(f"Negative Control Full Frames       : {len(neg_files)} frames")
    print(f"Negative 640x640 Training Patches  : {neg_patch_count} patches (with empty YOLO labels)")
    print(f"Report written to                  : {REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    prepare_case0()
