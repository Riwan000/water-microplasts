"""
Step 3: Morphological Sizing Engine
Computes minimum-area bounding box length/width in µm from a segmentation
polygon, and applies a Laplacian variance sharpness filter to discard
out-of-focus particles before sizing.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParticleDimensions:
    length_um: float
    width_um: float
    aspect_ratio: float
    surface_area_um2: float


def is_in_focus(crop_bgr: np.ndarray, sharpness_threshold: float = 100.0) -> bool:
    """Laplacian variance sharpness gate; rejects blurry/out-of-focus crops."""
    raise NotImplementedError("Wire up cv2.Laplacian variance check here (Step 3).")


def measure_particle(polygon_px: list[tuple[float, float]], px_to_um: float) -> ParticleDimensions:
    """Min-area rect (cv2.minAreaRect) on the polygon, converted to µm."""
    raise NotImplementedError("Wire up cv2.minAreaRect sizing here (Step 3).")
