"""
Step 3: Morphological Sizing Engine
Computes minimum-area bounding box length/width in µm from a segmentation
polygon, and applies a Laplacian variance sharpness filter to discard
out-of-focus particles before sizing.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ParticleDimensions:
    length_um: float
    width_um: float
    aspect_ratio: float
    surface_area_um2: float


def is_in_focus(crop_bgr: np.ndarray, sharpness_threshold: float = 100.0) -> bool:
    """Laplacian variance sharpness gate; rejects blurry/out-of-focus crops.

    Converts the crop to greyscale, applies the Laplacian operator, and
    measures variance. A high variance means sharp edges are present (in focus);
    a low variance means the crop is blurry and should be discarded.

    Args:
        crop_bgr: BGR image crop of a single particle region (NumPy array).
        sharpness_threshold: Laplacian variance below this value is considered
            out-of-focus. Default 100.0 works well for IBELL USB microscope
            images — tune upward for higher-resolution sensors.

    Returns:
        True if the crop is sharp enough to size reliably, False otherwise.
    """
    grey = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(grey, cv2.CV_64F).var()
    return float(variance) >= sharpness_threshold


def measure_particle(
    polygon_px: list[tuple[float, float]],
    px_to_um: float,
) -> ParticleDimensions:
    """Compute morphological dimensions from a segmentation polygon.

    Fits a minimum-area bounding rectangle (cv2.minAreaRect) to the polygon
    points and converts pixel lengths to micrometres using the calibration
    factor. The longer side is always reported as ``length_um`` so that
    aspect_ratio >= 1.0 for elongated fibres.

    Args:
        polygon_px: List of (x, y) pixel coordinate pairs forming the
            segmentation contour (from YOLO mask output).
        px_to_um: Calibration factor — micrometres per pixel — for the
            current microscope magnification (sidebar slider default 0.5).

    Returns:
        ParticleDimensions with length_um, width_um, aspect_ratio, and
        surface_area_um2 (approximated as length × width of the min-area rect).

    Raises:
        ValueError: If fewer than 3 points are provided (can't fit a rect).
    """
    if len(polygon_px) < 3:
        raise ValueError(
            f"Need at least 3 polygon points to fit a bounding rect, "
            f"got {len(polygon_px)}."
        )

    # cv2.minAreaRect expects a contour of shape (N, 1, 2) in float32
    contour = np.array(polygon_px, dtype=np.float32).reshape(-1, 1, 2)
    _, (w_px, h_px), _ = cv2.minAreaRect(contour)

    # Ensure length >= width (longer axis first)
    long_px, short_px = (h_px, w_px) if h_px >= w_px else (w_px, h_px)

    length_um = long_px * px_to_um
    width_um = short_px * px_to_um
    aspect_ratio = (length_um / width_um) if width_um > 0 else float("inf")
    surface_area_um2 = length_um * width_um

    return ParticleDimensions(
        length_um=round(length_um, 3),
        width_um=round(width_um, 3),
        aspect_ratio=round(aspect_ratio, 3),
        surface_area_um2=round(surface_area_um2, 3),
    )
