"""
Step B: Scene Compositor for Microplastic YOLO Segmentation Dataset
Generates 20,000 synthetic training scenes by:
  1. Cropping 640x640 patches from gridded filter paper canvases (CLASE_0)
  2. Pasting 5-15 cleaned particle stickers per scene using polygon masks
  3. Applying IBELL USB microscope optical simulation (50% of scenes)
  4. Saving images + YOLO segmentation labels in standard dataset structure
"""

import os
import cv2
import json
import math
import random
import shutil
import time
import yaml
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from concurrent.futures import ProcessPoolExecutor, as_completed

# ─── Paths ────────────────────────────────────────────────────────────────────
CANVAS_DIR      = os.path.join("data", "cleaned", "canvas_backgrounds")
PARTICLES_DIR   = os.path.join("data", "cleaned", "particles")
MASKS_DIR       = os.path.join("data", "cleaned", "particle_masks")
NEGATIVES_DIR   = os.path.join("data", "cleaned", "negatives")
OUTPUT_DIR      = os.path.join("data", "yolo_dataset")
REPORT_PATH     = os.path.join(OUTPUT_DIR, "compositor_report.json")

# ─── Parameters ───────────────────────────────────────────────────────────────
N_SCENES            = 20_000
TRAIN_SPLIT         = 0.80
PATCH_SIZE          = 640
MIN_PARTICLES       = 5
MAX_PARTICLES       = 15
SCALE_MIN           = 0.2
SCALE_MAX           = 1.8
OPTICS_PROB         = 0.50
MAX_OVERLAP_IOU     = 0.30
JPEG_QUALITY        = 92
SEED                = 42

# The 26511253 source dataset's raw photos carry a blue/cyan lighting cast
# (red channel crushed near-zero in the worst crops) that the neutral-toned
# canvas backgrounds don't share. Blending each sticker toward its own
# grayscale luminance softens that mismatch without the noise amplification
# a per-channel white-balance gain would cause on a near-zero red channel.
# The blend strength scales with how cast-affected each crop actually is
# (STICKER_CAST_NORM = the (G/B - R) gap, in 0-255 units, treated as "fully
# cast"), so a naturally-colored crop (e.g. algae) keeps its color while a
# heavily cast one gets pushed close to neutral gray.
STICKER_GRAY_BLEND_MIN = 0.0
STICKER_GRAY_BLEND_MAX = 0.9
STICKER_CAST_NORM      = 150.0

# Class IDs (must match extract_masks.py)
CLASS_IDS = {
    "fibre":    0,
    "fragment": 1,
    "pellet":   2,
    "foam_film":3,
    "algae":    4,
}


# ─── Helper: IoU between two axis-aligned bounding boxes ──────────────────────
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (area1 + area2 - inter + 1e-6)


# ─── Function 1: build_alpha_sticker ──────────────────────────────────────────
def soften_color_cast(img: Image.Image) -> Image.Image:
    """Blend an RGB image toward its own grayscale luminance, scaled to how
    strong its color cast is (see STICKER_GRAY_BLEND_MIN/MAX/STICKER_CAST_NORM)."""
    arr = np.asarray(img, dtype=np.float32)
    r, g, b = arr[..., 0].mean(), arr[..., 1].mean(), arr[..., 2].mean()
    cast_strength = np.clip((max(g, b) - r) / STICKER_CAST_NORM, 0.0, 1.0)
    blend = STICKER_GRAY_BLEND_MIN + cast_strength * (STICKER_GRAY_BLEND_MAX - STICKER_GRAY_BLEND_MIN)

    gray = img.convert("L").convert("RGB")
    return Image.blend(img, gray, float(blend))


def build_alpha_sticker(img_path: str, mask_txt_path: str):
    """
    Load a particle crop and its YOLO polygon mask.
    Returns (RGBA PIL image with transparent background, polygon points in the
    crop's own pixel space) so the true outline can be carried through the
    scale/rotate/paste transform in paste_particle() instead of being discarded.
    """
    img = Image.open(img_path).convert("RGB")
    img = soften_color_cast(img)
    w, h = img.size

    # Read polygon from YOLO .txt
    with open(mask_txt_path) as f:
        line = f.read().strip()
    if not line:
        return None

    parts = line.split()
    if len(parts) < 7:  # class_id + at least 3 points (6 coords)
        return None

    coords = list(map(float, parts[1:]))
    pts = []
    for i in range(0, len(coords)-1, 2):
        px = coords[i] * w
        py = coords[i+1] * h
        pts.append((px, py))

    if len(pts) < 3:
        return None

    # Create alpha mask from polygon
    alpha = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(alpha)
    draw.polygon([(int(x), int(y)) for x, y in pts], fill=255)

    # Small morphological dilation to avoid hairline white borders
    alpha_np = np.array(alpha)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    alpha_np = cv2.dilate(alpha_np, kernel, iterations=1)
    alpha = Image.fromarray(alpha_np)

    rgba = img.convert("RGBA")
    rgba.putalpha(alpha)
    return rgba, pts


# ─── Function 2: paste_particle ───────────────────────────────────────────────
def paste_particle(canvas_rgba, sticker_rgba, sticker_pts, class_id, placed_boxes, patch_size):
    """
    Rotate, scale, and paste one sticker onto the canvas.
    The sticker's true polygon outline (sticker_pts, in the original crop's
    pixel space) is carried through the same scale + rotation affine transform
    as the pixels, so the emitted label is the particle's real silhouette
    rather than its bounding box.
    Returns (updated_canvas, yolo_label_str, new_box) or (canvas, None) if placement fails.
    """
    sw, sh = sticker_rgba.size

    # Random scale: clamp so sticker fits within patch
    max_scale = min(SCALE_MAX, (patch_size * 0.8) / max(sw, sh, 1))
    scale = random.uniform(SCALE_MIN, max_scale)
    new_w = max(4, int(sw * scale))
    new_h = max(4, int(sh * scale))
    sticker = sticker_rgba.resize((new_w, new_h), Image.LANCZOS)

    # Scale factor actually applied (differs slightly from `scale` due to int rounding)
    eff_scale_x = new_w / sw
    eff_scale_y = new_h / sh
    scaled_pts = [(x * eff_scale_x, y * eff_scale_y) for x, y in sticker_pts]

    # Random rotation via explicit affine matrix (expand-to-fit, like PIL's
    # rotate(expand=True)) so the polygon points can be transformed with the
    # exact same matrix as the pixels instead of being re-derived afterward.
    angle = random.uniform(0, 360)
    center = (new_w / 2.0, new_h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    rw = max(1, int(new_h * sin + new_w * cos))
    rh = max(1, int(new_h * cos + new_w * sin))
    M[0, 2] += (rw / 2.0) - center[0]
    M[1, 2] += (rh / 2.0) - center[1]

    sticker_np = np.array(sticker)
    rotated_np = cv2.warpAffine(
        sticker_np, M, (rw, rh),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
    )
    sticker = Image.fromarray(rotated_np, mode="RGBA")

    rotated_pts = [
        (M[0, 0] * x + M[0, 1] * y + M[0, 2], M[1, 0] * x + M[1, 1] * y + M[1, 2])
        for x, y in scaled_pts
    ]

    # Random position within patch with margin
    margin = 4
    if rw >= patch_size - margin or rh >= patch_size - margin:
        return canvas_rgba, None
    max_x = patch_size - rw - margin
    max_y = patch_size - rh - margin
    px = random.randint(margin, max(margin + 1, max_x))
    py = random.randint(margin, max(margin + 1, max_y))

    # Overlap check against already placed particles (axis-aligned approximation)
    new_box = (px, py, px + rw, py + rh)
    for existing_box in placed_boxes:
        if compute_iou(new_box, existing_box) > MAX_OVERLAP_IOU:
            return canvas_rgba, None  # Too much overlap, skip

    # Alpha composite paste
    canvas_rgba.paste(sticker, (px, py), sticker)

    # Build the true YOLO segmentation polygon label (particle's real outline,
    # transformed and translated into canvas space, clamped to [0, 1])
    final_pts = [
        (min(1.0, max(0.0, (px + x) / patch_size)), min(1.0, max(0.0, (py + y) / patch_size)))
        for x, y in rotated_pts
    ]
    if len(final_pts) < 3:
        return canvas_rgba, None
    coord_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in final_pts)
    label_str = f"{class_id} {coord_str}"

    return canvas_rgba, label_str, new_box


# ─── Function 3: apply_ibell_optics ───────────────────────────────────────────
def apply_ibell_optics(img_np: np.ndarray) -> np.ndarray:
    """
    Apply 4 stacked IBELL USB microscope optical simulation effects.
    """
    h, w = img_np.shape[:2]

    # 1. LED Ring Hotspot & Vignette
    cx = w // 2 + random.randint(-15, 15)
    cy = h // 2 + random.randint(-15, 15)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    sigma = max(w, h) * random.uniform(0.4, 0.6)
    vignette = np.exp(-dist**2 / (2 * sigma**2))
    vignette = (0.55 + 0.45 * vignette).astype(np.float32)
    img_float = img_np.astype(np.float32)
    for c in range(3):
        img_float[:, :, c] *= vignette
    img_float = np.clip(img_float, 0, 255)

    # 2. Chromatic Aberration
    shift = random.randint(1, 3)
    r = img_float[:, :, 2]  # Red
    b = img_float[:, :, 0]  # Blue
    r_shifted = np.roll(np.roll(r, shift, axis=1), shift, axis=0)
    b_shifted = np.roll(np.roll(b, -shift, axis=1), -shift, axis=0)
    img_float[:, :, 2] = r_shifted
    img_float[:, :, 0] = b_shifted

    # 3. Defocus Blur
    sigma_blur = random.uniform(0.8, 1.8)
    ksize = max(3, int(2 * math.ceil(2 * sigma_blur) + 1))
    if ksize % 2 == 0:
        ksize += 1
    img_float = cv2.GaussianBlur(img_float, (ksize, ksize), sigma_blur)

    # 4. Sensor Noise (Gaussian + slight Poisson approximation)
    noise_std = random.uniform(2.0, 8.0)
    noise = np.random.normal(0, noise_std, img_float.shape).astype(np.float32)
    img_float = np.clip(img_float + noise, 0, 255).astype(np.uint8)

    return img_float


# ─── Helper: per-class inverse-frequency weights for sticker_pool ─────────────
def compute_class_balanced_weights(sticker_pool):
    """
    Weight each sticker by 1 / (size of its class) so every class carries
    equal total sampling probability, regardless of how many source images
    it has (e.g. fragment=3822 vs foam_film=212 vs algae=300).
    """
    class_counts = {}
    for _, _, class_id in sticker_pool:
        class_counts[class_id] = class_counts.get(class_id, 0) + 1
    return [1.0 / class_counts[class_id] for _, _, class_id in sticker_pool]


# ─── Function 4: generate_scene ───────────────────────────────────────────────
def generate_scene(scene_idx, canvas_files, sticker_pool, sticker_weights, rng_seed):
    """
    Generate one synthetic scene.
    Returns (scene_np, label_lines_list) or None on failure.
    """
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    # Pick a random canvas and crop a 640x640 patch
    canvas_path = random.choice(canvas_files)
    canvas_img = Image.open(canvas_path).convert("RGBA")
    cw, ch = canvas_img.size

    max_x = max(0, cw - PATCH_SIZE)
    max_y = max(0, ch - PATCH_SIZE)
    ox = random.randint(0, max_x) if max_x > 0 else 0
    oy = random.randint(0, max_y) if max_y > 0 else 0
    canvas_patch = canvas_img.crop((ox, oy, ox + PATCH_SIZE, oy + PATCH_SIZE)).convert("RGBA")

    # Pick and paste 5-15 particles, class-balanced via inverse-frequency weights
    n_particles = random.randint(MIN_PARTICLES, MAX_PARTICLES)
    candidates = random.choices(sticker_pool, weights=sticker_weights, k=n_particles)

    label_lines = []
    placed_boxes = []

    for img_path, mask_path, class_id in candidates:
        sticker_data = build_alpha_sticker(img_path, mask_path)
        if sticker_data is None:
            continue
        sticker_rgba, sticker_pts = sticker_data
        result = paste_particle(canvas_patch, sticker_rgba, sticker_pts, class_id, placed_boxes, PATCH_SIZE)
        if len(result) == 3:
            canvas_patch, label_str, box = result
            if label_str:
                label_lines.append(label_str)
                placed_boxes.append(box)
        else:
            canvas_patch, _ = result

    # Convert to BGR numpy for optional optics + saving
    scene_rgb = canvas_patch.convert("RGB")
    scene_np = cv2.cvtColor(np.array(scene_rgb), cv2.COLOR_RGB2BGR)

    # Apply IBELL optics 50% of the time
    if random.random() < OPTICS_PROB:
        scene_np = apply_ibell_optics(scene_np)

    return scene_np, label_lines


# ─── Function 5: run_compositor ───────────────────────────────────────────────
def run_compositor():
    random.seed(SEED)
    start_time = time.time()

    # Set up output dirs
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

    # Load canvas files
    canvas_files = sorted([
        os.path.join(CANVAS_DIR, f)
        for f in os.listdir(CANVAS_DIR)
        if f.lower().endswith((".jpg", ".jpeg"))
    ])
    print(f"[Init] Loaded {len(canvas_files)} canvas backgrounds.")

    # Build sticker pool (all plastic + algae, no noise)
    sticker_pool = []
    for cls, class_id in CLASS_IDS.items():
        p_folder = os.path.join(PARTICLES_DIR, cls)
        m_folder = os.path.join(MASKS_DIR, cls)
        if not os.path.exists(p_folder):
            continue
        for fname in sorted(os.listdir(p_folder)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            stem = os.path.splitext(fname)[0]
            mask_path = os.path.join(m_folder, stem + ".txt")
            if os.path.exists(mask_path):
                sticker_pool.append((os.path.join(p_folder, fname), mask_path, class_id))

    print(f"[Init] Sticker pool: {len(sticker_pool)} particle stickers.")
    sticker_weights = compute_class_balanced_weights(sticker_pool)

    # Determine train/val split indices
    indices = list(range(N_SCENES))
    random.shuffle(indices)
    n_train = int(N_SCENES * TRAIN_SPLIT)
    train_set = set(indices[:n_train])

    succeeded = 0
    failed = 0

    print(f"[Compositor] Generating {N_SCENES:,} synthetic scenes ({n_train:,} train / {N_SCENES - n_train:,} val)...")

    for i in range(N_SCENES):
        split = "train" if i in train_set else "val"
        scene_name = f"scene_{i:06d}"
        img_out = os.path.join(OUTPUT_DIR, "images", split, scene_name + ".jpg")
        lbl_out = os.path.join(OUTPUT_DIR, "labels", split, scene_name + ".txt")

        try:
            result = generate_scene(i, canvas_files, sticker_pool, sticker_weights, rng_seed=SEED + i)
            if result is None:
                failed += 1
                continue
            scene_np, label_lines = result

            # Save image
            cv2.imwrite(img_out, scene_np, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

            # Save labels
            with open(lbl_out, "w") as f:
                f.write("\n".join(label_lines) + "\n" if label_lines else "")

            succeeded += 1
        except Exception as e:
            failed += 1

        if (i + 1) % 2000 == 0 or (i + 1) == N_SCENES:
            elapsed = round(time.time() - start_time, 1)
            print(f"  [{i+1:>6}/{N_SCENES}] succeeded={succeeded} failed={failed} time={elapsed}s")

    # Copy negative patches into train (800) and val (200) splits
    neg_img_dir = os.path.join(NEGATIVES_DIR, "images")
    neg_lbl_dir = os.path.join(NEGATIVES_DIR, "labels")
    if os.path.exists(neg_img_dir):
        neg_files = sorted(os.listdir(neg_img_dir))
        random.shuffle(neg_files)
        n_neg_train = int(len(neg_files) * TRAIN_SPLIT)
        for j, nf in enumerate(neg_files):
            split = "train" if j < n_neg_train else "val"
            shutil.copy2(os.path.join(neg_img_dir, nf),
                         os.path.join(OUTPUT_DIR, "images", split, "neg_" + nf))
            stem = os.path.splitext(nf)[0]
            lbl_src = os.path.join(neg_lbl_dir, stem + ".txt")
            lbl_dst = os.path.join(OUTPUT_DIR, "labels", split, "neg_" + stem + ".txt")
            if os.path.exists(lbl_src):
                shutil.copy2(lbl_src, lbl_dst)
            else:
                open(lbl_dst, "w").close()
        print(f"[Negatives] Copied {len(neg_files)} negative patches into train/val splits.")

    # Write dataset.yaml
    dataset_yaml = {
        "path": os.path.abspath(OUTPUT_DIR),
        "train": "images/train",
        "val":   "images/val",
        "nc":    5,
        "names": ["fibre", "fragment", "pellet", "foam_film", "algae"]
    }
    yaml_path = os.path.join(OUTPUT_DIR, "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)

    elapsed_total = round(time.time() - start_time, 2)

    # Count final files
    n_train_imgs = len(os.listdir(os.path.join(OUTPUT_DIR, "images", "train")))
    n_val_imgs   = len(os.listdir(os.path.join(OUTPUT_DIR, "images", "val")))

    # Write report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_scenes_requested": N_SCENES,
        "n_scenes_succeeded": succeeded,
        "n_scenes_failed": failed,
        "train_images_total": n_train_imgs,
        "val_images_total": n_val_imgs,
        "execution_time_seconds": elapsed_total,
        "patch_size": PATCH_SIZE,
        "particles_per_scene": f"{MIN_PARTICLES}–{MAX_PARTICLES}",
        "optics_applied_prob": OPTICS_PROB,
        "dataset_yaml": os.path.abspath(yaml_path),
        "output_dir": os.path.abspath(OUTPUT_DIR)
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("SCENE COMPOSITOR COMPLETE")
    print("=" * 60)
    print(f"Synthetic scenes generated : {succeeded:,}")
    print(f"Train images (incl. neg)   : {n_train_imgs:,}")
    print(f"Val images   (incl. neg)   : {n_val_imgs:,}")
    print(f"Time taken                 : {elapsed_total}s")
    print(f"Dataset YAML               : {yaml_path}")
    print(f"Output directory           : {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)


if __name__ == "__main__":
    run_compositor()
