"""
DWPose / RTMW Whole-Body Pose Extractor Module (MMPose / ONNX Runtime).
Provides high-fidelity, whole-body keypoint extraction designed for complex
sign language gestures with severe hand rotations and side-view profile angles.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .constants import (
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    FACE_SLICE,
    HAND_LEFT_SLICE,
    HAND_RIGHT_SLICE,
    POSE_SLICE,
    TOTAL_KEYPOINTS_COUNT,
)

logger = logging.getLogger(__name__)


class DWPoseExtractor:
    """
    Whole-body keypoint extractor using DWPose / RTMW architecture.
    Extracts 133 whole-body keypoints and maps them directly to the LSEC SDK
    (T, 137, 3) OpenPose-compatible format.
    
    Can run via ONNX Runtime with CUDA execution provider (GPU) or CPU fallback.
    """

    def __init__(
        self,
        det_model_path: Optional[Union[str, Path]] = None,
        pose_model_path: Optional[Union[str, Path]] = None,
        device: str = "cuda",
    ):
        self.device = device
        self.det_model_path = Path(det_model_path) if det_model_path else None
        self.pose_model_path = Path(pose_model_path) if pose_model_path else None
        self.session_det = None
        self.session_pose = None
        self._is_initialized = False

    def init_sessions(self):
        """Initializes ONNX Runtime inference sessions if models are provided."""
        try:
            import onnxruntime as ort

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
            
            if self.pose_model_path and self.pose_model_path.exists():
                self.session_pose = ort.InferenceSession(str(self.pose_model_path), providers=providers)
                self._is_initialized = True
                logger.info(f"DWPose session initialized on {providers[0]}")
        except Exception as e:
            logger.warning(f"Could not initialize DWPose ONNX session: {e}")
            self._is_initialized = False

    def map_dwpose_to_lsec_137(
        self,
        dwpose_keypoints: np.ndarray,
        dwpose_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Maps DWPose 133-keypoint format into LSEC SDK 137-keypoint format (137, 3).
        
        DWPose layout (133):
          - 0..17: Body (18 keypoints)
          - 18..23: Feet (6 keypoints)
          - 24..91: Face (68 keypoints)
          - 92..112: Left Hand (21 keypoints)
          - 113..133: Right Hand (21 keypoints)
        """
        arr = np.zeros((TOTAL_KEYPOINTS_COUNT, 3), dtype=np.float32)

        # 1. Body & Pose mapping (18 body joints -> OpenPose 25 layout)
        # 0: Nose, 1: Neck, 2: RShoulder, 3: RElbow, 4: RWrist, 5: LShoulder, 6: LElbow, 7: LWrist...
        body_count = min(18, len(dwpose_keypoints))
        for i in range(body_count):
            score = float(dwpose_scores[i]) if dwpose_scores is not None else 1.0
            arr[i] = [dwpose_keypoints[i, 0], dwpose_keypoints[i, 1], score]

        # 2. Face (68 keypoints -> 70 points in LSEC)
        if len(dwpose_keypoints) >= 92:
            face_pts = dwpose_keypoints[24:92]
            face_sc = dwpose_scores[24:92] if dwpose_scores is not None else np.ones(len(face_pts))
            for i in range(min(68, len(face_pts))):
                arr[FACE_SLICE.start + i] = [face_pts[i, 0], face_pts[i, 1], float(face_sc[i])]

        # 3. Left Hand (21 keypoints: 92..112)
        if len(dwpose_keypoints) >= 113:
            lh_pts = dwpose_keypoints[92:113]
            lh_sc = dwpose_scores[92:113] if dwpose_scores is not None else np.ones(len(lh_pts))
            for i in range(min(21, len(lh_pts))):
                arr[HAND_LEFT_SLICE.start + i] = [lh_pts[i, 0], lh_pts[i, 1], float(lh_sc[i])]

        # 4. Right Hand (21 keypoints: 113..133)
        if len(dwpose_keypoints) >= 134:
            rh_pts = dwpose_keypoints[113:134]
            rh_sc = dwpose_scores[113:134] if dwpose_scores is not None else np.ones(len(rh_pts))
            for i in range(min(21, len(rh_pts))):
                arr[HAND_RIGHT_SLICE.start + i] = [rh_pts[i, 0], rh_pts[i, 1], float(rh_sc[i])]

        return arr
