"""Unit tests for FeatureProvider."""

import json
from pathlib import Path
import numpy as np
import pytest

from lsec_sdk.constants import (
    FACE_SLICE,
    HAND_LEFT_SLICE,
    HAND_RIGHT_SLICE,
    POSE_SLICE,
    TOTAL_KEYPOINTS_COUNT,
)
from lsec_sdk.feature_provider import FeatureProvider, SequenceResult


@pytest.fixture
def sample_sequence_dir():
    return Path("openpose_output/json/-fZc293MpJk_0-1-rgb_front")


def test_constants():
    assert TOTAL_KEYPOINTS_COUNT == 137
    assert (POSE_SLICE.stop - POSE_SLICE.start) == 25
    assert (FACE_SLICE.stop - FACE_SLICE.start) == 70
    assert (HAND_LEFT_SLICE.stop - HAND_LEFT_SLICE.start) == 21
    assert (HAND_RIGHT_SLICE.stop - HAND_RIGHT_SLICE.start) == 21


def test_extract_frame_index():
    fp = FeatureProvider()
    assert fp._extract_frame_index("-fZc293MpJk_0-1-rgb_front_000000000000_keypoints.json") == 0
    assert fp._extract_frame_index("-fZc293MpJk_0-1-rgb_front_000000000016_keypoints.json") == 16
    assert fp._extract_frame_index("frame_99_keypoints.json") == 99


def test_load_frame_shape(sample_sequence_dir):
    json_files = sorted(list(sample_sequence_dir.glob("*.json")))
    assert len(json_files) == 17

    fp = FeatureProvider()
    frame_arr = fp.load_frame_json(json_files[0])

    assert frame_arr.shape == (137, 3)
    assert frame_arr.dtype == np.float32

    # Verify pose, face, hands have non-zero points
    pose = frame_arr[POSE_SLICE]
    face = frame_arr[FACE_SLICE]
    hand_left = frame_arr[HAND_LEFT_SLICE]
    hand_right = frame_arr[HAND_RIGHT_SLICE]

    assert pose.shape == (25, 3)
    assert face.shape == (70, 3)
    assert hand_left.shape == (21, 3)
    assert hand_right.shape == (21, 3)

    assert np.any(pose[:, 2] > 0)
    assert np.any(face[:, 2] > 0)
    assert np.any(hand_right[:, 2] > 0)


def test_confidence_filter():
    fp = FeatureProvider(confidence_threshold=0.2, fill_strategy="zero")
    sample_kp = np.array([
        [100.0, 200.0, 0.05],  # Low conf -> should become 0, 0, 0
        [300.0, 400.0, 0.85],  # High conf -> should remain unchanged
    ], dtype=np.float32)

    filtered = fp.apply_confidence_filter(sample_kp)
    assert np.allclose(filtered[0], [0.0, 0.0, 0.0])
    assert np.allclose(filtered[1], [300.0, 400.0, 0.85])


def test_normalization_resolution():
    fp = FeatureProvider(normalization="resolution", resolution=(1280, 720))
    sample_kp = np.array([
        [640.0, 360.0, 0.9],
        [1280.0, 720.0, 1.0],
    ], dtype=np.float32)

    normalized = fp.normalize_keypoints(sample_kp)
    assert np.isclose(normalized[0, 0], 0.5)
    assert np.isclose(normalized[0, 1], 0.5)
    assert np.isclose(normalized[0, 2], 0.9)  # Confidence unchanged
    assert np.isclose(normalized[1, 0], 1.0)
    assert np.isclose(normalized[1, 1], 1.0)


def test_process_sequence_and_save(sample_sequence_dir, tmp_path):
    fp = FeatureProvider(
        confidence_threshold=0.1,
        normalization="resolution",
        resolution=(1280, 720),
    )

    result = fp.process_sequence(sample_sequence_dir)

    assert isinstance(result, SequenceResult)
    assert result.sequence_id == "-fZc293MpJk_0-1-rgb_front"
    assert result.num_frames == 17
    assert result.tensor.shape == (17, 137, 3)
    assert result.frame_indices == list(range(17))

    # Check metrics
    assert "mean_conf_pose" in result.metrics
    assert "mean_conf_face" in result.metrics
    assert "mean_conf_overall" in result.metrics
    assert result.metrics["mean_conf_overall"] > 0

    # Save .npy
    out_dir = tmp_path / "sequences"
    npy_file = fp.save_sequence(result, out_dir)
    assert npy_file.exists()

    # Load and verify .npy
    loaded_arr = np.load(npy_file)
    assert loaded_arr.shape == (17, 137, 3)
    assert np.allclose(loaded_arr, result.tensor)

    # Save metadata CSV
    csv_file = tmp_path / "metadata.csv"
    fp.save_metadata_csv([result], csv_file)
    assert csv_file.exists()

