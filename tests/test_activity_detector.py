import pytest
import numpy as np
from lsec_sdk.activity_detector import SignActivityDetector, ActivityState


def test_activity_detector_idle():
    detector = SignActivityDetector()
    # Dummy resting frame (wrists down at y=0.90, zero motion)
    kp = np.zeros((137, 3), dtype=np.float32)
    kp[4, 1] = 0.90  # RWrist down
    kp[7, 1] = 0.90  # LWrist down
    kp[4, 2] = 0.8
    kp[7, 2] = 0.8

    for _ in range(20):
        state, payload = detector.process_frame(kp)
        assert state == ActivityState.IDLE
        assert payload is None


def test_activity_detector_valid_gesture_cycle():
    detector = SignActivityDetector(
        pre_buffer_frames=5,
        min_gesture_frames=15,
        max_gesture_frames=50,
        end_cooldown_frames=4,
    )

    # 1. Idle frames
    idle_kp = np.zeros((137, 3), dtype=np.float32)
    idle_kp[4, 1] = 0.90
    idle_kp[7, 1] = 0.90
    idle_kp[4, 2] = 0.8
    idle_kp[7, 2] = 0.8

    for _ in range(5):
        detector.process_frame(idle_kp)

    # 2. Start gesture: wrists elevated, dynamic motion in hands
    activated = False
    for i in range(1, 21):
        active_kp = idle_kp.copy()
        active_kp[4, 1] = 0.45  # Elevated into chest/throat signing space
        active_kp[116:137, 0] = np.sin(i * 0.5) * 0.1  # Dynamic hand motion
        state, payload = detector.process_frame(active_kp)
        if state == ActivityState.ACTIVE:
            activated = True

    assert activated

    # 3. Rest frames (lowered back to rest)
    final_payload = None
    for _ in range(6):
        state, payload = detector.process_frame(idle_kp)
        if state == ActivityState.COMPLETED:
            final_payload = payload
            break

    assert final_payload is not None
    assert final_payload["frames_count"] >= 15
    assert final_payload["duration_sec"] > 0
    assert final_payload["keypoints_sequence"].shape[0] == final_payload["frames_count"]
    assert final_payload["keypoints_sequence"].shape[1:] == (137, 3)


def test_activity_detector_short_noise_rejected():
    detector = SignActivityDetector(min_gesture_frames=20, end_cooldown_frames=3)

    idle_kp = np.zeros((137, 3), dtype=np.float32)
    idle_kp[4, 1] = 0.90
    idle_kp[4, 2] = 0.8

    # Brief 3-frame scratch or twitch
    for i in range(3):
        active_kp = idle_kp.copy()
        active_kp[4, 1] = 0.40
        active_kp[116:137, 0] = i * 0.05
        detector.process_frame(active_kp)

    # Then immediately idle again
    completed = False
    for _ in range(5):
        state, payload = detector.process_frame(idle_kp)
        if state == ActivityState.COMPLETED:
            completed = True

    # Too short -> rejected as noise, returned to IDLE
    assert not completed
    assert detector.state == ActivityState.IDLE
