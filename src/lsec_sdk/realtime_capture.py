"""Real-time sign language keypoint extraction using MediaPipe Holistic and OpenCV."""

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union

import cv2
import mediapipe as mp
import numpy as np

from .constants import (
    FACE_SLICE,
    HAND_LEFT_SLICE,
    HAND_RIGHT_SLICE,
    POSE_SLICE,
    TOTAL_KEYPOINTS_COUNT,
)

logger = logging.getLogger(__name__)


class RealTimeCapture:
    """Captures video stream from webcam and tracks body, face, and hands in real time."""

    def __init__(
        self,
        camera_index: int = 0,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.camera_index = camera_index
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1,
            smooth_landmarks=True,
        )

        # Drawing styles
        self.pose_conn_spec = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
        self.pose_pt_spec = self.mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=3, circle_radius=3)
        self.hand_l_conn = self.mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
        self.hand_l_pt = self.mp_drawing.DrawingSpec(color=(200, 0, 200), thickness=2, circle_radius=2)
        self.hand_r_conn = self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
        self.hand_r_pt = self.mp_drawing.DrawingSpec(color=(0, 200, 200), thickness=2, circle_radius=2)
        self.face_spec = self.mp_drawing.DrawingSpec(color=(255, 200, 0), thickness=1, circle_radius=1)

    def extract_landmarks_array(self, results) -> np.ndarray:
        """
        Extracts detected landmarks into a structured (137, 3) tensor format
        compatible with the LSEC SDK standards (x, y, confidence).
        """
        arr = np.zeros((TOTAL_KEYPOINTS_COUNT, 3), dtype=np.float32)

        # 1. Pose: Map MediaPipe 33 pose landmarks to OpenPose 25 approximation
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            # MediaPipe indices:
            # 0: nose, 11: left_shoulder, 12: right_shoulder, 13: left_elbow, 14: right_elbow,
            # 15: left_wrist, 16: right_wrist, 23: left_hip, 24: right_hip
            # Neck approximated as midpoint of shoulders
            neck_x = (lm[11].x + lm[12].x) / 2.0
            neck_y = (lm[11].y + lm[12].y) / 2.0
            neck_c = min(lm[11].visibility, lm[12].visibility)

            mid_hip_x = (lm[23].x + lm[24].x) / 2.0
            mid_hip_y = (lm[23].y + lm[24].y) / 2.0
            mid_hip_c = min(lm[23].visibility, lm[24].visibility)

            pose_map = [
                (0, lm[0]),               # Nose
                (1, (neck_x, neck_y, neck_c)), # Neck
                (2, lm[12]),              # RShoulder
                (3, lm[14]),              # RElbow
                (4, lm[16]),              # RWrist
                (5, lm[11]),              # LShoulder
                (6, lm[13]),              # LElbow
                (7, lm[15]),              # LWrist
                (8, (mid_hip_x, mid_hip_y, mid_hip_c)), # MidHip
                (9, lm[24]),              # RHip
                (10, lm[26]),             # RKnee
                (11, lm[28]),             # RAnkle
                (12, lm[23]),             # LHip
                (13, lm[25]),             # LKnee
                (14, lm[27]),             # LAnkle
                (15, lm[5]),              # REye
                (16, lm[2]),              # LEye
                (17, lm[8]),              # REar
                (18, lm[7]),              # LEar
            ]
            for target_idx, val in pose_map:
                if isinstance(val, tuple):
                    arr[target_idx] = [val[0], val[1], val[2]]
                else:
                    arr[target_idx] = [val.x, val.y, val.visibility]

        # 2. Face: Sample 70 landmarks if face mesh is detected
        if results.face_landmarks:
            flm = results.face_landmarks.landmark
            # Sample 70 landmarks evenly across the 468 mesh points
            step = max(1, len(flm) // 70)
            for i in range(70):
                src_i = min(i * step, len(flm) - 1)
                pt = flm[src_i]
                arr[FACE_SLICE.start + i] = [pt.x, pt.y, 1.0]

        # 3. Left Hand (21 landmarks)
        if results.left_hand_landmarks:
            for i, pt in enumerate(results.left_hand_landmarks.landmark):
                if i < 21:
                    arr[HAND_LEFT_SLICE.start + i] = [pt.x, pt.y, 1.0]

        # 4. Right Hand (21 landmarks)
        if results.right_hand_landmarks:
            for i, pt in enumerate(results.right_hand_landmarks.landmark):
                if i < 21:
                    arr[HAND_RIGHT_SLICE.start + i] = [pt.x, pt.y, 1.0]

        return arr

    def render_hud(
        self,
        frame: np.ndarray,
        fps: float,
        is_recording: bool,
        recorded_frames: int,
        results,
    ) -> np.ndarray:
        """Overlays diagnostic information and recording status on the frame."""
        h, w, _ = frame.shape
        overlay = frame.copy()

        # Top banner background
        cv2.rectangle(overlay, (0, 0), (w, 60), (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

        # FPS
        cv2.putText(
            frame,
            f"FPS: {fps:4.1f}",
            (15, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Tracked parts indicators
        face_detected = results.face_landmarks is not None
        rh_detected = results.right_hand_landmarks is not None
        lh_detected = results.left_hand_landmarks is not None

        c_face = (0, 255, 0) if face_detected else (80, 80, 80)
        c_rh = (0, 255, 255) if rh_detected else (80, 80, 80)
        c_lh = (255, 0, 255) if lh_detected else (80, 80, 80)

        cv2.putText(frame, "Rostro", (150, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c_face, 2, cv2.LINE_AA)
        cv2.putText(frame, "Mano D.", (240, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c_rh, 2, cv2.LINE_AA)
        cv2.putText(frame, "Mano I.", (340, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c_lh, 2, cv2.LINE_AA)

        # Recording Status
        if is_recording:
            # Pulsing red circle
            pulse = int((time.time() * 4) % 2)
            rec_color = (0, 0, 255) if pulse else (0, 0, 180)
            cv2.circle(frame, (w - 180, 32), 10, rec_color, -1, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"REC [{recorded_frames:03d}]",
                (w - 160, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                frame,
                "[ESPACIO]: Grabar  |  [Q]: Salir",
                (w - 360, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

        return frame

    def draw_skeleton(self, frame: np.ndarray, results) -> np.ndarray:
        """Draws skeletal lines and landmarks directly on the frame using MediaPipe."""
        # Pose
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.pose_pt_spec,
                connection_drawing_spec=self.pose_conn_spec,
            )

        # Face Mesh Contours
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.face_landmarks,
                self.mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.face_spec,
            )

        # Left Hand
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.hand_l_pt,
                connection_drawing_spec=self.hand_l_conn,
            )

        # Right Hand
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.hand_r_pt,
                connection_drawing_spec=self.hand_r_conn,
            )

        return frame

    def close(self):
        """Releases MediaPipe resources."""
        self.holistic.close()

