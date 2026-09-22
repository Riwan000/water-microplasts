"""
Step 3: Two-Tier Output Logic
Tier 1 — aggregates plastic detections into a Total Microplastics Count and
Particles/L. Tier 2 — per-particle morphotype, dimensions, aspect ratio.
"""

from dataclasses import dataclass

PLASTIC_CLASSES = {"fibre", "fragment", "pellet", "foam_film"}
EXCLUDED_CLASSES = {"algae"}


@dataclass(frozen=True)
class Tier1Summary:
    total_plastic_count: int
    particles_per_litre: float


def compute_tier1(detections: list[dict], filtered_volume_litres: float) -> Tier1Summary:
    """Aggregate Tier 1 concentration: C = N / V_filtered.

    Only counts detections whose ``class_name`` is in PLASTIC_CLASSES.
    Detections in EXCLUDED_CLASSES (algae, silt) are silently ignored —
    they should have already been routed to the quarantine bucket by the
    quarantine gate, but this provides a second-line filter.

    Args:
        detections: List of detection dicts. Each dict must have at minimum:
            ``{"class_name": str, "confidence": float, ...}``
        filtered_volume_litres: Volume of water filtered through the membrane
            in litres, used to compute concentration (particles/L).

    Returns:
        Tier1Summary with total_plastic_count and particles_per_litre.

    Raises:
        ValueError: If filtered_volume_litres is zero or negative.
    """
    if filtered_volume_litres <= 0:
        raise ValueError(
            f"filtered_volume_litres must be positive, got {filtered_volume_litres}."
        )

    plastic_detections = [
        d for d in detections
        if d.get("class_name") in PLASTIC_CLASSES
    ]
    total_count = len(plastic_detections)
    concentration = total_count / filtered_volume_litres

    return Tier1Summary(
        total_plastic_count=total_count,
        particles_per_litre=round(concentration, 4),
    )


def compute_tier2(detections: list[dict]) -> list[dict]:
    """Build per-particle morphotype breakdown for Tier 2 output.

    For each detection in PLASTIC_CLASSES, returns a flat dict suitable for
    CSV export and the Streamlit particle inspector. Non-plastic classes
    (algae, unclassified) are excluded from Tier 2.

    Args:
        detections: List of detection dicts. Expected keys per detection:
            - ``class_name`` (str): morphotype label
            - ``confidence`` (float): model confidence 0–1
            - ``length_um`` (float): particle length in µm (from sizing engine)
            - ``width_um`` (float): particle width in µm
            - ``aspect_ratio`` (float): length / width
            - ``surface_area_um2`` (float): approximate surface area in µm²
            - ``bbox`` (list[float]): [x1, y1, x2, y2] pixel bounding box
            - ``particle_id`` (int | str): unique identifier within the sample

    Returns:
        List of dicts — one per confirmed plastic particle — with all
        morphological fields normalised and ready for CSV / PDF export.
    """
    tier2_rows = []
    for det in detections:
        if det.get("class_name") not in PLASTIC_CLASSES:
            continue
        tier2_rows.append(
            {
                "particle_id": det.get("particle_id", ""),
                "class_name": det.get("class_name", "unknown"),
                "confidence": round(float(det.get("confidence", 0.0)), 4),
                "length_um": det.get("length_um", 0.0),
                "width_um": det.get("width_um", 0.0),
                "aspect_ratio": det.get("aspect_ratio", 0.0),
                "surface_area_um2": det.get("surface_area_um2", 0.0),
                "bbox_x1": det.get("bbox", [None, None, None, None])[0],
                "bbox_y1": det.get("bbox", [None, None, None, None])[1],
                "bbox_x2": det.get("bbox", [None, None, None, None])[2],
                "bbox_y2": det.get("bbox", [None, None, None, None])[3],
            }
        )
    return tier2_rows

