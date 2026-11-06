"""Unit test for RealTimeCapture."""

import numpy as np
from lsec_sdk.constants import TOTAL_KEYPOINTS_COUNT
from lsec_sdk.realtime_capture import RealTimeCapture


def test_realtime_capture_initialization():
    capture = RealTimeCapture()
    assert capture.holistic is not None

    # Test extracting from mock empty results
    class MockResults:
        pose_landmarks = None
        face_landmarks = None
        left_hand_landmarks = None
        right_hand_landmarks = None

    arr = capture.extract_landmarks_array(MockResults())
    assert arr.shape == (TOTAL_KEYPOINTS_COUNT, 3)
    assert arr.dtype == np.float32
    assert np.all(arr == 0.0)

    capture.close()

