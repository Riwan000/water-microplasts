"""
Step A: Automated Polygon Mask Extractor for Microplastic Particle Crops
Traces pixel-accurate polygon outlines for all cleaned particle images
and saves them in standard YOLO segmentation .txt format.

Segmentation strategy, per particle:
  1. If the particle's original raw crop has a matching TSV bounding-box
     annotation, run GrabCut inside that box on the *original raw source
     photo* (real background context) and also compute an Otsu candidate
     on the crop itself; evaluate both and keep the better one.
  2. Otherwise (no bounding-box coverage, e.g. algae/noise), fall back to
     plain Otsu thresholding on the crop, exactly as before.
This never modifies anything under data/raw/ -- only reads from it.

Class ID Mapping:
  0 = fibre
  1 = fragment
  2 = pellet
  3 = foam_film
  4 = algae
  noise -> empty .txt files (negative distractors, no mask needed)
"""

import csv
import os
import json
import random
import time
from functools import lru_cache

import cv2
import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────────────
CLEANED_ROOT          = os.path.join("data", "cleaned")
PARTICLES_DIR         = os.path.join(CLEANED_ROOT, "particles")
PARTICLES_MANIFEST    = os.path.join(CLEANED_ROOT, "particles_manifest.csv")
MASKS_DIR             = os.path.join(CLEANED_ROOT, "particle_masks")
DEBUG_DIR             = os.path.join(CLEANED_ROOT, "mask_debug")
REPORT_PATH           = os.path.join(CLEANED_ROOT, "mask_extraction_report.json")

DATASET_ROOTS = [
    os.path.join("data", "raw", "26511253", "MICRO", "MICRO"),
    os.path.join("data", "raw", "26511253", "VALIDATION", "VALIDATION", "MICRO"),
]

# ─── Class Config ─────────────────────────────────────────────────────────────
CLASS_IDS = {
    "fibre":     0,
    "fragment":  1,
    "pellet":    2,
    "foam_film": 3,
    "algae":     4,
    "noise":     -1,   # No mask, empty label only
}

# TSV "ID" column folder prefixes that clean_particles.py actually maps to a
# target class (e.g. "reference" crops exist in the raw data but were never
# cleaned into data/cleaned/particles, so they carry no bbox we'd ever use).
KNOWN_BBOX_FOLDERS = {"line", "hard", "pellet", "foam", "film", "noise"}

ALGAE_MAX_DIM        = 640   # Resize algae to this before processing
MIN_CONTOUR_AREA     = 8     # Discard contours smaller than this (pure noise)
SUSPICIOUS_AREA_FRAC = 0.85  # Contour covering more than this fraction of the crop is
                              # almost certainly the background (wrong Otsu/GrabCut
                              # polarity), not the particle -> rejected, not accepted.

# A second, tighter suspicious rule for compact-shaped classes only (not
# "fibre" -- a correctly-traced winding/looping fibre naturally encloses a
# large filled area and touches most crop edges, so this signal would
# misfire on it). For fragment/pellet/foam_film/algae, a contour that's both
# fairly large AND touches most of the crop's edges is almost always a
# GrabCut/Otsu failure that grabbed surrounding background rather than a
# particle that genuinely fills its own tight crop (verified against
# foam_film_26511253_micro_00171, area_frac=0.83, border_touch_sides=3 --
# under the 0.85 cutoff above but visibly a background-inclusive mask).
SUSPICIOUS_AREA_FRAC_COMPACT   = 0.6
SUSPICIOUS_BORDER_TOUCH_COMPACT = 3
NON_COMPACT_CLASSES            = {"fibre"}

# A third suspicious signal, independent of shape: a mask that fuses the
# particle with an adjacent chunk of background (a "tail") rather than
# swallowing the whole crop won't trip the area/border rules above, but its
# pixels split into two distinct, substantial color clusters -- one per
# material -- instead of one. Verified against foam_film_26511253_micro_00062
# (dist=274, minority=0.43, area_frac=0.68 -- a real leak) vs.
# fibre_26511253_micro_00103 (dist=251, area_frac=0.57 -- a correctly-traced
# but looping fibre whose enclosed loop just happens to read as bimodal,
# hence LEAK_AREA_FRAC_MIN plus excluding "fibre" via NON_COMPACT_CLASSES).
LEAK_COLOR_DIST_MIN     = 200.0
LEAK_MINORITY_FRAC_MIN  = 0.15
LEAK_AREA_FRAC_MIN      = 0.3

DEBUG_PER_CLASS      = 30    # Total debug overlays saved per class
DEBUG_SUSPECT_TARGET = 15    # Of which, up to this many are drawn from suspicious cases
SEED                 = 42

GRABCUT_PAD_FRAC     = 0.25  # Context padding around the box, as a fraction of box size
GRABCUT_PAD_MIN      = 8     # ...and a floor in pixels, for tiny boxes
GRABCUT_ITERS        = 5
GRABCUT_MAX_DIM      = 400   # Downscale large padded windows before GrabCut (huge
                              # speedup on big fibre/fragment boxes); mask is scaled
                              # back up afterward, before cropping to the tight box.
RAW_IMAGE_CACHE_SIZE = 6     # Full source photos are large; cache a handful across
                              # the dozens of particles typically cropped from each one.


# ─── Bounding-box index (TSV annotations -> raw source photo + box) ───────────
def _normalize_tsv_id(raw_id: str) -> str:
    """TSV 'ID' entries use colon-formatted timestamps, but the actual crop
    files on disk have colons replaced with underscores (Windows-safe names).
    """
    return raw_id.strip().replace(":", "_")


def _parse_tsv_rows(tsv_path: str, dataset_root: str, raw_img_path: str) -> dict:
    entries = {}
    with open(tsv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            raw_id = _normalize_tsv_id(row.get("ID", ""))
            folder = raw_id.split("/", 1)[0] if "/" in raw_id else ""
            if folder not in KNOWN_BBOX_FOLDERS:
                continue
            try:
                bbox = (int(row["minr"]), int(row["minc"]), int(row["maxr"]), int(row["maxc"]))
            except (KeyError, ValueError):
                continue
            relpath = os.path.normpath(os.path.join(dataset_root, raw_id.replace("/", os.sep)))
            entries.setdefault(relpath, {"raw_img_path": raw_img_path, "bbox": bbox})
    return entries


def build_bbox_index(dataset_roots: list) -> dict:
    """Scan every annotation/*.tsv under each dataset root and return
    {normalized_raw_crop_relpath: {"raw_img_path", "bbox"}}.
    A TSV whose corresponding raw_img source photo is missing is skipped
    entirely; its particles simply fall back to the Otsu-only path.
    """
    bbox_index = {}
    for root in dataset_roots:
        ann_dir = os.path.join(root, "annotation")
        raw_img_dir = os.path.join(root, "raw_img")
        if not os.path.isdir(ann_dir):
            continue
        for tsv_name in sorted(os.listdir(ann_dir)):
            if not tsv_name.lower().endswith(".tsv"):
                continue
            raw_img_path = os.path.join(raw_img_dir, tsv_name[:-4])
            if not os.path.isfile(raw_img_path):
                continue
            tsv_path = os.path.join(ann_dir, tsv_name)
            bbox_index.update(_parse_tsv_rows(tsv_path, root, raw_img_path))
    return bbox_index


def load_particle_manifest(manifest_path: str) -> dict:
    """Map each cleaned particle image's own path to its original raw crop's
    relpath, so it can be looked up in the bbox index built above.
    """
    manifest_map = {}
    if not os.path.isfile(manifest_path):
        return manifest_map
    with open(manifest_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            particle_path = os.path.normpath(
                os.path.join(PARTICLES_DIR, row["target_class"], row["particle_id"])
            )
            manifest_map[particle_path] = os.path.normpath(row["source_relpath"])
    return manifest_map


@lru_cache(maxsize=RAW_IMAGE_CACHE_SIZE)
def _load_raw_image(path: str):
    return cv2.imread(path)


# ─── Function: apply_class_preprocessing ──────────────────────────────────────
def apply_class_preprocessing(gray: np.ndarray, cls: str, orig_w: int, orig_h: int) -> np.ndarray:
    """Class-specific pre-processing shared by both the crop-only Otsu path
    and the bbox-guided Otsu candidate.
    """
    if cls in ("algae", "foam_film"):
        return cv2.GaussianBlur(gray, (5, 5), 0)

    if cls == "fibre":
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)

    if cls == "pellet":
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        return cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)

    if cls == "fragment" and min(orig_w, orig_h) >= 20:
        return cv2.GaussianBlur(gray, (3, 3), 0)

    return gray


# ─── Function: load_and_preprocess ────────────────────────────────────────────
def load_and_preprocess(img_path: str, cls: str):
    """
    Load particle crop, convert to grayscale, and apply class-specific
    pre-processing to get the best binary mask from Otsu thresholding.
    Returns (gray_preprocessed, original_bgr, original_size, scale_factor)
    """
    bgr = cv2.imread(img_path)
    if bgr is None:
        return None, None, None, 1.0

    orig_h, orig_w = bgr.shape[:2]
    original_size  = (orig_w, orig_h)
    scale_factor   = 1.0

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    if cls == "algae":
        max_dim = max(orig_w, orig_h)
        if max_dim > ALGAE_MAX_DIM:
            scale_factor = ALGAE_MAX_DIM / max_dim
            new_w = int(orig_w * scale_factor)
            new_h = int(orig_h * scale_factor)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
            bgr  = cv2.resize(bgr,  (new_w, new_h), interpolation=cv2.INTER_AREA)

    gray = apply_class_preprocessing(gray, cls, orig_w, orig_h)
    return gray, bgr, original_size, scale_factor


# ─── Function: auto_polarity_binary ───────────────────────────────────────────
def auto_polarity_binary(gray: np.ndarray) -> np.ndarray:
    """
    Otsu thresholding with automatic foreground polarity detection.

    A hardcoded THRESH_BINARY_INV assumes the particle is always darker than
    its background. That holds for some source datasets (e.g. algae_dataset
    crops) but not others (e.g. 26511253 'hard'/'line' crops, where the
    plastic shard is often brighter than the gray background) -- picking the
    wrong polarity silently selects the background as "foreground" and traces
    almost the entire crop instead of the particle.

    Heuristic: a correctly cropped particle sits with a background margin, so
    its mask should NOT dominate a thin ring around the image border. Try
    both polarities and keep whichever leaves less foreground on that ring.
    """
    h, w = gray.shape
    _, dark_is_fg  = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, light_is_fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    border = np.zeros((h, w), dtype=bool)
    b = max(1, min(h, w) // 12)
    border[:b, :] = border[-b:, :] = border[:, :b] = border[:, -b:] = True

    dark_border_frac  = (dark_is_fg[border] > 0).mean()
    light_border_frac = (light_is_fg[border] > 0).mean()

    return dark_is_fg if dark_border_frac <= light_border_frac else light_is_fg


# ─── GrabCut candidate (bounding-box-guided) ──────────────────────────────────
def grabcut_mask_for_bbox(raw_bgr: np.ndarray, bbox: tuple):
    """
    Run GrabCut on a context-padded window of the ORIGINAL raw source photo,
    seeded from the annotated bounding box, and return the resulting binary
    mask cropped back down to the tight box (i.e. the particle crop's own
    pixel dimensions). Returns None if the box can't be segmented.
    """
    minr, minc, maxr, maxc = bbox
    img_h, img_w = raw_bgr.shape[:2]
    box_h, box_w = maxr - minr, maxc - minc

    pad_r = max(GRABCUT_PAD_MIN, int(GRABCUT_PAD_FRAC * box_h))
    pad_c = max(GRABCUT_PAD_MIN, int(GRABCUT_PAD_FRAC * box_w))
    pr0, pc0 = max(0, minr - pad_r), max(0, minc - pad_c)
    pr1, pc1 = min(img_h, maxr + pad_r), min(img_w, maxc + pad_c)

    window = raw_bgr[pr0:pr1, pc0:pc1]
    if window.size == 0:
        return None

    # GrabCut's GMM fitting cost scales with pixel count; large fibre/fragment
    # boxes can be thousands of pixels wide, so cap the working resolution and
    # scale the resulting mask back up afterward.
    win_h, win_w = window.shape[:2]
    scale = min(1.0, GRABCUT_MAX_DIM / max(win_h, win_w))
    small = window if scale >= 1.0 else cv2.resize(
        window, (max(1, int(win_w * scale)), max(1, int(win_h * scale))), interpolation=cv2.INTER_AREA
    )
    small_h, small_w = small.shape[:2]

    gc_mask = np.full((small_h, small_w), cv2.GC_PR_BGD, dtype=np.uint8)
    fr0, fc0 = int((minr - pr0) * scale), int((minc - pc0) * scale)
    fr1 = max(fr0 + 1, min(small_h, int((maxr - pr0) * scale)))
    fc1 = max(fc0 + 1, min(small_w, int((maxc - pc0) * scale)))
    gc_mask[fr0:fr1, fc0:fc1] = cv2.GC_PR_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small, gc_mask, None, bgd_model, fgd_model,
                    GRABCUT_ITERS, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None

    fg_small = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    fg = fg_small if scale >= 1.0 else cv2.resize(fg_small, (win_w, win_h), interpolation=cv2.INTER_NEAREST)
    return fg[minr - pr0:maxr - pr0, minc - pc0:maxc - pc0]


# ─── Candidate scoring / selection / refinement ───────────────────────────────
def largest_external_contour(binary: np.ndarray):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_CONTOUR_AREA:
        return None
    return largest


def _bimodal_leak_score(masked_pixels_bgr: np.ndarray):
    """Split a mask's pixels into two brightness clusters via Otsu and
    return (color_distance, minority_fraction) between their mean colors.
    A real particle/background leak shows up as two well-separated,
    substantial clusters; shading or anti-aliasing within one material
    doesn't. See LEAK_* constants above.
    """
    if len(masked_pixels_bgr) < 20:
        return 0.0, 0.0
    gray = cv2.cvtColor(masked_pixels_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY).reshape(-1)
    if gray.min() == gray.max():
        return 0.0, 0.0
    thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    lo = masked_pixels_bgr[gray <= thresh]
    hi = masked_pixels_bgr[gray > thresh]
    if len(lo) == 0 or len(hi) == 0:
        return 0.0, 0.0
    dist = float(np.linalg.norm(lo.mean(axis=0).astype(np.float32) - hi.mean(axis=0).astype(np.float32)))
    minority_frac = min(len(lo), len(hi)) / len(gray)
    return dist, minority_frac


def _mask_diagnostics(binary: np.ndarray):
    contour = largest_external_contour(binary)
    if contour is None:
        return None
    h, w = binary.shape
    area_frac = cv2.contourArea(contour) / float(w * h)
    return {"area_frac": area_frac, "suspicious": area_frac > SUSPICIOUS_AREA_FRAC}


def choose_candidate_mask(otsu_binary, grabcut_binary):
    """Evaluate the Otsu and GrabCut candidates and pick the better one.
    GrabCut wins when it isn't suspicious (it has real background context,
    so it's usually the tighter/more accurate boundary); otherwise fall back
    to whichever candidate is non-suspicious, or the smaller-area one if both
    are flagged (less likely to be "whole crop mistaken for foreground").
    """
    grab_diag = _mask_diagnostics(grabcut_binary) if grabcut_binary is not None else None
    otsu_diag = _mask_diagnostics(otsu_binary) if otsu_binary is not None else None

    if grab_diag and not grab_diag["suspicious"]:
        return grabcut_binary, "grabcut"
    if otsu_diag and not otsu_diag["suspicious"]:
        return otsu_binary, "otsu"
    if grab_diag and otsu_diag:
        if grab_diag["area_frac"] <= otsu_diag["area_frac"]:
            return grabcut_binary, "grabcut"
        return otsu_binary, "otsu"
    if grab_diag:
        return grabcut_binary, "grabcut"
    if otsu_diag:
        return otsu_binary, "otsu"
    return None, "none"


def refine_binary_mask(binary: np.ndarray) -> np.ndarray:
    """Close small gaps, open small speckles, then keep only the largest
    external contour (filled) to drop leftover fragments/holes.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    contour = largest_external_contour(cleaned)
    if contour is None:
        return cleaned

    filled = np.zeros_like(cleaned)
    cv2.drawContours(filled, [contour], -1, color=255, thickness=cv2.FILLED)
    return filled


# ─── Function: binary_to_polygon / extract_polygon ────────────────────────────
def binary_to_polygon(binary: np.ndarray, orig_w: int, orig_h: int, scale_factor: float = 1.0,
                       cls: str = None, bgr: np.ndarray = None):
    """
    Find the largest external contour in an already-binarized mask, simplify,
    normalize to 0.0-1.0, and return (polygon, diagnostics) or (None, None).
    """
    contour = largest_external_contour(binary)
    if contour is None:
        return None, None

    work_h, work_w = binary.shape
    area_frac = cv2.contourArea(contour) / float(work_w * work_h)
    x, y, cw, ch = cv2.boundingRect(contour)
    border_touch_sides = sum([
        x <= 1, y <= 1, (x + cw) >= work_w - 1, (y + ch) >= work_h - 1,
    ])
    suspicious = area_frac > SUSPICIOUS_AREA_FRAC
    if cls not in NON_COMPACT_CLASSES:
        suspicious = suspicious or (
            area_frac > SUSPICIOUS_AREA_FRAC_COMPACT
            and border_touch_sides >= SUSPICIOUS_BORDER_TOUCH_COMPACT
        )
        if not suspicious and bgr is not None and area_frac > LEAK_AREA_FRAC_MIN:
            dist, minority_frac = _bimodal_leak_score(bgr[binary > 0])
            suspicious = dist > LEAK_COLOR_DIST_MIN and minority_frac > LEAK_MINORITY_FRAC_MIN
    diagnostics = {
        "area_frac": round(area_frac, 4),
        "border_touch_sides": border_touch_sides,
        "suspicious": bool(suspicious),
    }

    perimeter = cv2.arcLength(contour, True)
    epsilon = float(np.clip(0.01 * perimeter, 0.75, 2.5))
    simplified = cv2.approxPolyDP(contour, epsilon, True)
    if len(simplified) < 3:
        simplified = contour

    pts = simplified.reshape(-1, 2).astype(float)
    if scale_factor != 1.0 and scale_factor > 0:
        pts = pts / scale_factor
    pts[:, 0] = np.clip(pts[:, 0] / orig_w, 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1] / orig_h, 0.0, 1.0)

    return pts.flatten().tolist(), diagnostics


def extract_polygon(gray: np.ndarray, cls: str, original_size: tuple, scale_factor: float, bgr: np.ndarray = None):
    """Otsu-only path (used when no bbox annotation is available)."""
    orig_w, orig_h = original_size
    binary = auto_polarity_binary(gray)
    return binary_to_polygon(binary, orig_w, orig_h, scale_factor, cls=cls, bgr=bgr)


# ─── Bounding-box-guided segmentation entry point ─────────────────────────────
def process_bbox_guided(img_path: str, cls: str, bbox_entry: dict):
    """
    Segment a particle using its TSV bounding box: GrabCut on the padded
    region of the original raw source photo, versus an Otsu candidate on the
    crop itself, then pick, refine, and polygon-ize the winner.
    Returns (polygon, diagnostics, method); polygon is None if this path
    can't be used, so the caller can fall back to the plain Otsu path.
    """
    orig_bgr = cv2.imread(img_path)
    if orig_bgr is None:
        return None, None, "read_error"

    orig_h, orig_w = orig_bgr.shape[:2]
    minr, minc, maxr, maxc = bbox_entry["bbox"]

    # Guard against a stale/mismatched TSV<->particle lookup (e.g. a rename)
    # silently producing a garbage mask -- fail safe to Otsu-only instead.
    if abs((maxr - minr) - orig_h) > 2 or abs((maxc - minc) - orig_w) > 2:
        return None, None, "bbox_mismatch"

    raw_bgr = _load_raw_image(bbox_entry["raw_img_path"])
    if raw_bgr is None:
        return None, None, "raw_img_missing"

    grab_bin = grabcut_mask_for_bbox(raw_bgr, (minr, minc, maxr, maxc))

    gray = apply_class_preprocessing(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY), cls, orig_w, orig_h)
    otsu_bin = auto_polarity_binary(gray)

    chosen_bin, method = choose_candidate_mask(otsu_bin, grab_bin)
    if chosen_bin is None:
        return None, None, "none"

    refined = refine_binary_mask(chosen_bin)
    polygon, diagnostics = binary_to_polygon(refined, orig_w, orig_h, scale_factor=1.0, cls=cls, bgr=orig_bgr)
    if diagnostics and diagnostics["suspicious"]:
        # The winning candidate still looks background-inclusive after
        # refinement -- reject rather than accept it, so the caller falls
        # back to the plain crop-only Otsu path instead of saving a mask
        # that pastes a chunk of raw-photo background as if it were the
        # particle (see SUSPICIOUS_AREA_FRAC_COMPACT above).
        return None, diagnostics, f"{method}_rejected"
    return polygon, diagnostics, method


def resolve_polygon_for_image(img_path: str, cls: str, manifest_map: dict, bbox_index: dict):
    """Try the bbox-guided path first; fall back to plain Otsu on the crop."""
    bbox_key = manifest_map.get(os.path.normpath(img_path))
    bbox_entry = bbox_index.get(bbox_key) if bbox_key else None

    if bbox_entry is not None:
        polygon, diagnostics, method = process_bbox_guided(img_path, cls, bbox_entry)
        if polygon is not None:
            return polygon, diagnostics, method

    gray, bgr, original_size, scale_factor = load_and_preprocess(img_path, cls)
    if gray is None:
        return None, None, "read_error"
    polygon, diagnostics = extract_polygon(gray, cls, original_size, scale_factor, bgr=bgr)
    method = "otsu_fallback" if bbox_entry is not None else "otsu_no_bbox"
    if diagnostics and diagnostics["suspicious"]:
        # Last-resort path is also background-inclusive -- no further
        # fallback exists, so record it as failed rather than silently
        # saving a bad mask.
        return None, diagnostics, f"{method}_rejected"
    return polygon, diagnostics, method


# ─── Function: save_yolo_label ────────────────────────────────────────────────
def save_yolo_label(polygon, class_id: int, output_path: str):
    """
    Write YOLO segmentation .txt file:
      class_id  x1 y1  x2 y2  x3 y3  ...
    If polygon is None, writes an empty file (failed / negative sample).
    """
    with open(output_path, "w") as f:
        if polygon and len(polygon) >= 6:
            coords = " ".join(f"{v:.6f}" for v in polygon)
            f.write(f"{class_id} {coords}\n")
        # else: empty file → YOLO treats as background-only image


# ─── Function: save_debug_overlay ─────────────────────────────────────────────
def save_debug_overlay(img_path: str, polygon, cls: str, output_path: str,
                        suspicious: bool = False, method: str = ""):
    """
    Draw the extracted polygon contour over the original image and save as JPEG.
    """
    bgr = cv2.imread(img_path)
    if bgr is None or polygon is None or len(polygon) < 6:
        return

    orig_h, orig_w = bgr.shape[:2]
    pts = np.array(polygon).reshape(-1, 2)
    pts[:, 0] *= orig_w
    pts[:, 1] *= orig_h
    pts = pts.astype(np.int32)

    # Draw filled semi-transparent overlay
    overlay = bgr.copy()
    fill_color = (0, 0, 255) if suspicious else (0, 200, 100)
    cv2.fillPoly(overlay, [pts], color=fill_color)
    bgr = cv2.addWeighted(overlay, 0.3, bgr, 0.7, 0)

    # Draw contour border
    line_color = (0, 0, 255) if suspicious else (0, 255, 80)
    cv2.polylines(bgr, [pts], isClosed=True, color=line_color, thickness=2)

    # Label
    tag = f"{cls}/{method}" if method else cls
    label = f"{tag} SUSPECT" if suspicious else tag
    cv2.putText(bgr, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, line_color, 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, bgr)


# ─── Main ─────────────────────────────────────────────────────────────────────
def run_extraction():
    random.seed(SEED)
    start_time = time.time()

    os.makedirs(MASKS_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)

    print("[Init] Indexing TSV bounding-box annotations...")
    bbox_index = build_bbox_index(DATASET_ROOTS)
    manifest_map = load_particle_manifest(PARTICLES_MANIFEST)
    print(f"[Init] Indexed {len(bbox_index):,} bounding boxes across {len(DATASET_ROOTS)} dataset roots, "
          f"{len(manifest_map):,} particles in manifest.")

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_attempted": 0,
        "total_succeeded": 0,
        "total_failed": 0,
        "total_suspicious": 0,
        "method_breakdown": {},
        "class_breakdown": {},
        "failures": []
    }

    for cls, class_id in CLASS_IDS.items():
        src_folder  = os.path.join(PARTICLES_DIR, cls)
        mask_folder = os.path.join(MASKS_DIR, cls)
        os.makedirs(mask_folder, exist_ok=True)

        if not os.path.exists(src_folder):
            print(f"[Skip] Source folder not found: {src_folder}")
            continue

        img_files = sorted([f for f in os.listdir(src_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        total = len(img_files)

        attempted = 0
        succeeded = 0
        failed    = 0
        vertex_counts = []
        method_counts = {}
        records = []  # (fname, img_path, polygon, suspicious) for post-hoc debug sampling

        print(f"\n[{cls.upper()}] Processing {total:,} images (class_id={class_id})...")

        for fname in img_files:
            img_path   = os.path.join(src_folder, fname)
            label_stem = os.path.splitext(fname)[0]
            label_path = os.path.join(mask_folder, label_stem + ".txt")

            attempted += 1
            report["total_attempted"] += 1

            # Noise class → empty label, skip mask extraction
            if class_id == -1:
                save_yolo_label(None, 0, label_path)
                succeeded += 1
                report["total_succeeded"] += 1
                continue

            polygon, diagnostics, method = resolve_polygon_for_image(img_path, cls, manifest_map, bbox_index)
            method_counts[method] = method_counts.get(method, 0) + 1
            report["method_breakdown"][method] = report["method_breakdown"].get(method, 0) + 1

            save_yolo_label(polygon, class_id, label_path)

            if polygon and len(polygon) >= 6:
                succeeded += 1
                report["total_succeeded"] += 1
                vertex_counts.append(len(polygon) // 2)
                suspicious = diagnostics["suspicious"]
                if suspicious:
                    report["total_suspicious"] += 1
                records.append((fname, img_path, polygon, suspicious, method))
            else:
                failed += 1
                report["total_failed"] += 1
                report["failures"].append({"file": img_path, "reason": f"No valid contour found ({method})"})

        # ── Debug sample selection: bias toward suspicious cases so a human
        #    can actually spot polarity/threshold failures during review. ──
        suspicious_records = [r for r in records if r[3]]
        clean_records      = [r for r in records if not r[3]]
        random.shuffle(suspicious_records)
        random.shuffle(clean_records)

        n_suspect = min(DEBUG_SUSPECT_TARGET, len(suspicious_records))
        n_clean   = min(DEBUG_PER_CLASS - n_suspect, len(clean_records))
        debug_set = suspicious_records[:n_suspect] + clean_records[:n_clean]

        for fname, img_path, polygon, suspicious, method in debug_set:
            label_stem = os.path.splitext(fname)[0]
            tag = "_SUSPECT" if suspicious else ""
            debug_path = os.path.join(DEBUG_DIR, f"{cls}_{label_stem}{tag}_debug.jpg")
            save_debug_overlay(img_path, polygon, cls, debug_path, suspicious=suspicious, method=method)

        avg_vertices = round(sum(vertex_counts) / len(vertex_counts), 1) if vertex_counts else 0
        suspicious_count = len(suspicious_records)
        suspicious_rate = round(100 * suspicious_count / attempted, 1) if attempted else 0
        report["class_breakdown"][cls] = {
            "attempted": attempted,
            "succeeded": succeeded,
            "failed":    failed,
            "suspicious": suspicious_count,
            "suspicious_rate_pct": suspicious_rate,
            "avg_polygon_vertices": avg_vertices,
            "method_breakdown": method_counts,
        }

        success_rate = round(100 * succeeded / attempted, 1) if attempted else 0
        print(f"  Done: {succeeded}/{attempted} succeeded ({success_rate}%) | "
              f"suspicious: {suspicious_count} ({suspicious_rate}%) | avg vertices: {avg_vertices} | "
              f"methods: {method_counts}")

    # Save report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 60)
    print("POLYGON MASK EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Total attempted  : {report['total_attempted']:,}")
    print(f"Total succeeded  : {report['total_succeeded']:,}")
    print(f"Total failed     : {report['total_failed']:,}")
    print(f"Total suspicious : {report['total_suspicious']:,}")
    print(f"Method breakdown : {report['method_breakdown']}")
    print(f"Success rate     : {round(100 * report['total_succeeded'] / max(report['total_attempted'], 1), 1)}%")
    print(f"Time taken       : {elapsed}s")
    print(f"Masks saved to   : {os.path.abspath(MASKS_DIR)}")
    print(f"Debug images     : {os.path.abspath(DEBUG_DIR)}")
    print(f"Report saved to  : {os.path.abspath(REPORT_PATH)}")
    print("=" * 60)


if __name__ == "__main__":
    run_extraction()
