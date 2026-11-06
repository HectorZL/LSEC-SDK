"""FeatureProvider module for How2Sign OpenPose keypoints processing."""

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .constants import (
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    DEFAULT_FPS,
    FACE_KEYPOINTS_COUNT,
    FACE_SLICE,
    HAND_LEFT_KEYPOINTS_COUNT,
    HAND_LEFT_SLICE,
    HAND_RIGHT_KEYPOINTS_COUNT,
    HAND_RIGHT_SLICE,
    POSE_KEYPOINTS_COUNT,
    POSE_SLICE,
    TOTAL_KEYPOINTS_COUNT,
)

logger = logging.getLogger(__name__)


@dataclass
class SequenceResult:
    """Encapsulates processed sequence tensor and related metadata."""
    sequence_id: str
    tensor: np.ndarray  # Shape: (T, 137, 3) where channels are (x, y, confidence)
    num_frames: int
    frame_indices: List[int]
    fps: float
    width: int
    height: int
    normalization: str
    confidence_threshold: float
    metrics: Dict[str, float] = field(default_factory=dict)
    npy_path: Optional[str] = None
    label: Optional[str] = None

    def to_metadata_dict(self) -> Dict[str, Union[str, int, float]]:
        """Converts sequence metadata and quality metrics to a dictionary for CSV logging."""
        record: Dict[str, Union[str, int, float]] = {
            "sequence_id": self.sequence_id,
            "label": self.label or "",
            "num_frames": self.num_frames,
            "fps": self.fps,
            "resolution_w": self.width,
            "resolution_h": self.height,
            "total_keypoints": TOTAL_KEYPOINTS_COUNT,
            "tensor_shape": f"{self.tensor.shape}",
            "normalization": self.normalization,
            "confidence_threshold": self.confidence_threshold,
            "npy_path": self.npy_path or "",
        }
        record.update(self.metrics)
        return record


class FeatureProvider:
    """Extracts, filters, normalizes and structures OpenPose keypoint sequences from How2Sign."""

    def __init__(
        self,
        confidence_threshold: float = 0.1,
        normalization: str = "resolution",
        resolution: Tuple[int, int] = (DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT),
        fps: float = DEFAULT_FPS,
        fill_strategy: str = "zero",
    ):
        """
        Initializes FeatureProvider.

        Args:
            confidence_threshold: Threshold below which points are considered invalid.
            normalization: 'resolution' (scaled by W, H to [0, 1]), 'center' (normalized by neck/shoulders), or 'none'.
            resolution: Tuple (width, height) of original video frames.
            fps: Frame rate of the sequence.
            fill_strategy: 'zero' to set coordinates and confidence of low-confidence points to 0.0.
        """
        self.confidence_threshold = confidence_threshold
        self.normalization = normalization.lower()
        self.width, self.height = resolution
        self.fps = fps
        self.fill_strategy = fill_strategy

        if self.normalization not in ("resolution", "center", "none"):
            raise ValueError(f"Unsupported normalization: {normalization}. Use 'resolution', 'center', or 'none'.")

    def _extract_frame_index(self, filename: str) -> Optional[int]:
        """Extracts the numeric frame index from an OpenPose JSON filename."""
        # e.g. -fZc293MpJk_0-1-rgb_front_000000000015_keypoints.json
        match = re.search(r"_(\d+)_keypoints\.json$", filename)
        if match:
            return int(match.group(1))
        # Fallback for generic patterns
        match = re.search(r"(\d+)", filename)
        return int(match.group(1)) if match else None

    def _select_main_person(self, people: List[dict]) -> Optional[dict]:
        """Selects the primary signer from the people array based on confidence and upper-body presence."""
        if not people:
            return None
        if len(people) == 1:
            return people[0]

        best_person = None
        best_score = -1.0

        for p in people:
            pose = p.get("pose_keypoints_2d", [])
            face = p.get("face_keypoints_2d", [])
            # Sum confidences of pose and face
            conf_sum = sum(pose[2::3]) + sum(face[2::3])
            if conf_sum > best_score:
                best_score = conf_sum
                best_person = p

        return best_person

    def load_frame_json(self, json_path: Union[str, Path]) -> np.ndarray:
        """
        Reads an OpenPose JSON file and returns structured keypoints array of shape (137, 3).

        Keypoints ordering:
        - [0:25]   : Body / Pose (25 points)
        - [25:95]  : Face (70 points)
        - [95:116] : Left hand (21 points)
        - [116:137]: Right hand (21 points)
        """
        json_path = Path(json_path)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        people = data.get("people", [])
        person = self._select_main_person(people)

        frame_data = np.zeros((TOTAL_KEYPOINTS_COUNT, 3), dtype=np.float32)
        if person is None:
            return frame_data

        def unpack(key: str, expected_count: int) -> np.ndarray:
            raw = person.get(key, [])
            arr = np.array(raw, dtype=np.float32)
            if len(arr) == expected_count * 3:
                return arr.reshape((expected_count, 3))
            elif len(arr) == 0:
                return np.zeros((expected_count, 3), dtype=np.float32)
            else:
                out = np.zeros((expected_count, 3), dtype=np.float32)
                valid_len = min(len(arr) // 3, expected_count)
                out[:valid_len] = arr[:valid_len * 3].reshape((valid_len, 3))
                return out

        frame_data[POSE_SLICE] = unpack("pose_keypoints_2d", POSE_KEYPOINTS_COUNT)
        frame_data[FACE_SLICE] = unpack("face_keypoints_2d", FACE_KEYPOINTS_COUNT)
        frame_data[HAND_LEFT_SLICE] = unpack("hand_left_keypoints_2d", HAND_LEFT_KEYPOINTS_COUNT)
        frame_data[HAND_RIGHT_SLICE] = unpack("hand_right_keypoints_2d", HAND_RIGHT_KEYPOINTS_COUNT)

        return frame_data

    def apply_confidence_filter(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Sets coordinates with confidence below threshold to 0.

        Args:
            keypoints: Array of shape (..., 3) where [..., 0]=x, [..., 1]=y, [..., 2]=conf
        Returns:
            Filtered keypoints array.
        """
        filtered = keypoints.copy()
        conf = filtered[..., 2]
        low_conf = conf < self.confidence_threshold

        if self.fill_strategy == "zero":
            filtered[..., 0][low_conf] = 0.0
            filtered[..., 1][low_conf] = 0.0
            filtered[..., 2][low_conf] = 0.0

        return filtered

    def normalize_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Normalizes keypoint coordinates based on selected strategy.

        Args:
            keypoints: Array of shape (..., 3)
        Returns:
            Normalized keypoints array.
        """
        normalized = keypoints.copy()

        if self.normalization == "none":
            return normalized

        if self.normalization == "resolution":
            # Scale x by width and y by height
            normalized[..., 0] = normalized[..., 0] / float(self.width)
            normalized[..., 1] = normalized[..., 1] / float(self.height)
            return normalized

        if self.normalization == "center":
            # Signer-centric normalization:
            # Pose index 1 is Neck, index 2 is RShoulder, index 5 is LShoulder
            neck = keypoints[..., 1:2, :2]  # Shape (..., 1, 2)
            r_sh = keypoints[..., 2:3, :2]
            l_sh = keypoints[..., 5:6, :2]

            # Calculate shoulder width for scaling
            shoulder_dist = np.linalg.norm(r_sh - l_sh, axis=-1, keepdims=True)  # (..., 1, 1)
            scale = np.where(shoulder_dist > 1e-4, shoulder_dist, float(self.width) * 0.2)

            # Center relative to neck and scale
            normalized[..., :2] = (normalized[..., :2] - neck) / scale
            return normalized

        return normalized

    def compute_metrics(self, raw_tensor: np.ndarray, filtered_tensor: np.ndarray) -> Dict[str, float]:
        """Calculates statistics on keypoint quality and confidences."""
        pose_conf = raw_tensor[:, POSE_SLICE, 2]
        face_conf = raw_tensor[:, FACE_SLICE, 2]
        lh_conf = raw_tensor[:, HAND_LEFT_SLICE, 2]
        rh_conf = raw_tensor[:, HAND_RIGHT_SLICE, 2]
        all_conf = raw_tensor[:, :, 2]

        total_points = all_conf.size
        low_conf_count = int(np.sum(all_conf < self.confidence_threshold))

        return {
            "mean_conf_pose": round(float(np.mean(pose_conf)), 4),
            "mean_conf_face": round(float(np.mean(face_conf)), 4),
            "mean_conf_hand_left": round(float(np.mean(lh_conf)), 4),
            "mean_conf_hand_right": round(float(np.mean(rh_conf)), 4),
            "mean_conf_overall": round(float(np.mean(all_conf)), 4),
            "low_conf_ratio": round(float(low_conf_count / total_points), 4),
        }

    def process_sequence(
        self,
        sequence_dir: Union[str, Path],
        sequence_id: Optional[str] = None,
    ) -> SequenceResult:
        """
        Reads, sorts, filters, and normalizes all JSON files in a sequence directory.

        Args:
            sequence_dir: Path to directory containing frame JSON files.
            sequence_id: Optional custom ID; defaults to folder name.
        Returns:
            SequenceResult object containing tensor of shape (T, 137, 3) and metadata.
        """
        seq_path = Path(sequence_dir)
        if not seq_path.is_dir():
            raise FileNotFoundError(f"Sequence directory not found: {seq_path}")

        if sequence_id is None:
            sequence_id = seq_path.name

        json_files = list(seq_path.glob("*.json"))
        if not json_files:
            raise ValueError(f"No JSON files found in {seq_path}")

        # Parse frame index and sort temporally
        indexed_files = []
        for f in json_files:
            idx = self._extract_frame_index(f.name)
            if idx is not None:
                indexed_files.append((idx, f))
            else:
                # If no integer found, fallback to name sort with index 0
                indexed_files.append((0, f))

        indexed_files.sort(key=lambda item: item[0])

        frames = []
        frame_indices = []
        for idx, file_path in indexed_files:
            frame_indices.append(idx)
            frame_arr = self.load_frame_json(file_path)
            frames.append(frame_arr)

        raw_tensor = np.stack(frames, axis=0)  # Shape (T, 137, 3)

        # Apply confidence filtering
        filtered_tensor = self.apply_confidence_filter(raw_tensor)

        # Apply coordinate normalization
        normalized_tensor = self.normalize_keypoints(filtered_tensor)

        # Calculate metrics
        metrics = self.compute_metrics(raw_tensor, filtered_tensor)

        return SequenceResult(
            sequence_id=sequence_id,
            tensor=normalized_tensor.astype(np.float32),
            num_frames=len(frames),
            frame_indices=frame_indices,
            fps=self.fps,
            width=self.width,
            height=self.height,
            normalization=self.normalization,
            confidence_threshold=self.confidence_threshold,
            metrics=metrics,
        )

    def process_video(
        self,
        video_path: Union[str, Path],
        sequence_id: Optional[str] = None,
        label: Optional[str] = None,
    ) -> SequenceResult:
        """
        Extracts keypoint features from an MP4/AVI/MOV video file using MediaPipe Holistic.

        Args:
            video_path: Path to video file.
            sequence_id: Identifier for the sequence.
            label: Linguistic label or gloss (e.g. 'COMO TE SIENTES').
        Returns:
            SequenceResult containing tensor of shape (T, 137, 3) and metrics.
        """
        import cv2
        from .realtime_capture import RealTimeCapture

        v_path = Path(video_path)
        if not v_path.exists():
            raise FileNotFoundError(f"Video file not found: {v_path}")

        if sequence_id is None:
            clean_name = re.sub(r"[^\w\-]", "_", v_path.stem)
            sequence_id = clean_name

        cap = cv2.VideoCapture(str(v_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {v_path}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.width
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.height
        v_fps = float(cap.get(cv2.CAP_PROP_FPS)) or self.fps

        capture = RealTimeCapture()
        frames_list = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = capture.holistic.process(rgb)
                kp = capture.extract_landmarks_array(results)
                frames_list.append(kp)
        finally:
            cap.release()
            capture.close()

        if not frames_list:
            raise ValueError(f"No frames could be extracted from video: {v_path}")

        raw_tensor = np.stack(frames_list, axis=0)  # Shape (T, 137, 3)

        if self.normalization == "center":
            normalized_tensor = self.normalize_keypoints(raw_tensor)
        elif self.normalization == "none":
            normalized_tensor = raw_tensor.copy()
            normalized_tensor[..., 0] *= float(w)
            normalized_tensor[..., 1] *= float(h)
        else:  # 'resolution' - MediaPipe is already normalized in [0, 1]
            normalized_tensor = raw_tensor.copy()

        filtered_tensor = self.apply_confidence_filter(normalized_tensor)
        metrics = self.compute_metrics(raw_tensor, filtered_tensor)

        return SequenceResult(
            sequence_id=sequence_id,
            tensor=filtered_tensor.astype(np.float32),
            num_frames=len(frames_list),
            frame_indices=list(range(len(frames_list))),
            fps=round(v_fps, 2),
            width=w,
            height=h,
            normalization=self.normalization,
            confidence_threshold=self.confidence_threshold,
            metrics=metrics,
            label=label,
        )

    def save_sequence(self, result: SequenceResult, output_dir: Union[str, Path]) -> Path:
        """
        Saves sequence tensor as a .npy file.

        Args:
            result: Processed SequenceResult.
            output_dir: Directory where .npy files are stored.
        Returns:
            Path to saved .npy file.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        npy_file = out_path / f"{result.sequence_id}.npy"
        np.save(npy_file, result.tensor)
        result.npy_path = str(npy_file.resolve())
        return npy_file

    def save_metadata_csv(
        self,
        results: List[SequenceResult],
        csv_path: Union[str, Path],
        append: bool = False,
    ) -> Path:
        """
        Exports metadata records of processed sequences to a CSV file.

        Args:
            results: List of SequenceResult objects.
            csv_path: Destination path for CSV.
            append: If True, appends to existing CSV without duplicating headers.
        Returns:
            Path to written CSV file.
        """
        csv_file = Path(csv_path)
        csv_file.parent.mkdir(parents=True, exist_ok=True)

        rows = [r.to_metadata_dict() for r in results]
        df_new = pd.DataFrame(rows)

        if append and csv_file.exists():
            df_existing = pd.read_csv(csv_file)
            # Remove any duplicate sequence_id before appending
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=["sequence_id"], keep="last")
            df_combined.to_csv(csv_file, index=False)
        else:
            df_new.to_csv(csv_file, index=False)

        return csv_file

