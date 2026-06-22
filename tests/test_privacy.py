import numpy as np

from backend.attention import Track
from backend.vision import anonymize_and_annotate


def test_anonymizer_does_not_mutate_or_return_raw_frame():
    frame = np.random.default_rng(7).integers(0, 255, (300, 400, 3), dtype=np.uint8)
    original = frame.copy()
    track = Track(1, (140, 80, 80, 100), last_seen=1.0)
    safe = anonymize_and_annotate(frame, [track])
    assert safe is not frame
    assert np.array_equal(frame, original)
    assert not np.array_equal(safe[50:210, 120:240], original[50:210, 120:240])
    assert len(np.unique(safe[70:180, 140:220].reshape(-1, 3), axis=0)) < len(
        np.unique(original[70:180, 140:220].reshape(-1, 3), axis=0)
    )

