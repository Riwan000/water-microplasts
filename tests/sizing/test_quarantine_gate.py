from src.sizing.quarantine_gate import CONFIDENCE_THRESHOLD, passes_quarantine_gate


def test_passes_when_confidence_meets_threshold():
    assert passes_quarantine_gate(CONFIDENCE_THRESHOLD) is True


def test_fails_when_confidence_below_threshold():
    assert passes_quarantine_gate(CONFIDENCE_THRESHOLD - 0.01) is False
