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
    """C = N / V_filtered over plastic-only detections (algae/silt excluded)."""
    raise NotImplementedError("Aggregate Tier 1 concentration here (Step 3).")


def compute_tier2(detections: list[dict]) -> list[dict]:
    """Per-particle morphotype + dimensions breakdown."""
    raise NotImplementedError("Build Tier 2 morphotype breakdown here (Step 3).")
