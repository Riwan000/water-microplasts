"""
Step 4: Pandas-based CSV export for particle-level audit trails.

Converts the Tier 2 detection list into a UTF-8 CSV byte string ready
to be returned from the FastAPI export endpoint or saved to disk under
data/sessions/{sample_id}.csv.
"""

import io

import pandas as pd

# Column order for the exported CSV
_COLUMNS = [
    "particle_id",
    "class_name",
    "confidence",
    "length_um",
    "width_um",
    "aspect_ratio",
    "surface_area_um2",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
]


def build_csv_report(sample_id: str, tier2: list[dict]) -> bytes:
    """Build a CSV audit trail from Tier 2 particle detections.

    Args:
        sample_id: Unique identifier for the sample (used as a header comment).
        tier2: List of per-particle dicts as returned by ``compute_tier2()``.
            Each dict should contain the keys listed in ``_COLUMNS``.

    Returns:
        UTF-8 encoded CSV bytes with a one-line comment header, ready for
        ``FileResponse`` or ``st.download_button``.
    """
    buf = io.StringIO()

    # Prepend a comment line so the file is self-identifying
    buf.write(f"# Microplastic Detector — Sample: {sample_id}\n")

    if not tier2:
        # Return a valid but empty CSV so downstream code doesn't break
        buf.write(",".join(_COLUMNS) + "\n")
        return buf.getvalue().encode("utf-8")

    df = pd.DataFrame(tier2)

    # Reorder / fill missing columns gracefully
    for col in _COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[_COLUMNS]

    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

