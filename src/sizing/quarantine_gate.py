"""
Step 3: Quarantine Gate
Routes low-confidence detections to an "Unclassified Debris" review bucket
instead of the plastic count, so weak/anomalous predictions can't inflate
Tier 1/2 results.
"""

CONFIDENCE_THRESHOLD = 0.45


def passes_quarantine_gate(confidence: float) -> bool:
    """True if a detection is confident enough to count toward Tier 1/2."""
    return confidence >= CONFIDENCE_THRESHOLD
