"""Visualizer module for LSEC SDK - Renders skeletons, overlays, and comparisons."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from .constants import (
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    FACE_SLICE,
    HAND_LEFT_SLICE,
    HAND_RIGHT_SLICE,
    POSE_SLICE,
)

# Bone connections for BODY_25 (indices 0 to 24)
BODY_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1),   # Nose - Neck
    (1, 2),   # Neck - RShoulder
    (2, 3),   # RShoulder - RElbow
    (3, 4),   # RElbow - RWrist
    (1, 5),   # Neck - LShoulder
    (5, 6),   # LShoulder - LElbow
    (6, 7),   # LElbow - LWrist
    (1, 8),   # Neck - MidHip
    (8, 9),   # MidHip - RHip
    (9, 10),  # RHip - RKnee
    (10, 11), # RKnee - RAnkle
    (8, 12),  # MidHip - LHip
    (12, 13), # LHip - LKnee
    (13, 14), # LKnee - LAnkle
]

# Anatomical Face connections (relative to FACE_SLICE.start = 25)
FACE_CONNECTIONS: List[Tuple[int, int]] = [
    # Jawline (0..16)
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
    (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),
    # Right eyebrow (17..21)
    (17, 18), (18, 19), (19, 20), (20, 21),
    # Left eyebrow (22..26)
    (22, 23), (23, 24), (24, 25), (25, 26),
    # Nose bridge (27..30)
    (27, 28), (28, 29), (29, 30),
    # Nose bottom (31..35)
    (31, 32), (32, 33), (33, 34), (34, 35),
    # Right eye loop (36..41)
    (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36),
    # Left eye loop (42..47)
    (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42),
    # Outer lips loop (48..59)
    (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54),
    (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 48),
    # Inner lips loop (60..67)
    (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67), (67, 60),
]

# Hand finger bones (relative to wrist 0)
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
]


class Visualizer:
    """Renders 2D keypoints and skeletal structures onto frames or canvases."""

    def __init__(
        self,
        confidence_threshold: float = 0.1,
        color_pose: Tuple[int, int, int] = (0, 255, 0),        # Green (BGR)
        color_face: Tuple[int, int, int] = (255, 200, 0),      # Cyan/Blue
        color_hand_left: Tuple[int, int, int] = (255, 0, 255), # Magenta
        color_hand_right: Tuple[int, int, int] = (0, 255, 255),# Yellow
    ):
        self.conf_thresh = confidence_threshold
        self.color_pose = color_pose
        self.color_face = color_face
        self.color_hand_left = color_hand_left
        self.color_hand_right = color_hand_right

    def draw_skeleton(
        self,
        canvas: np.ndarray,
        keypoints: np.ndarray,
        width: int,
        height: int,
        normalized: bool = True,
    ) -> np.ndarray:
        """
        Draws pose bones, face points, and hand finger lines on the canvas.

        Args:
            canvas: BGR image array (H, W, 3).
            keypoints: Array of shape (137, 3) where columns are (x, y, conf).
            width: Frame width for denormalization.
            height: Frame height for denormalization.
            normalized: True if coordinates are in [0, 1].
        """
        img = canvas.copy()
        scale_x = width if normalized else 1.0
        scale_y = height if normalized else 1.0

        pts = []
        for i in range(137):
            x, y, c = keypoints[i]
            if c >= self.conf_thresh and (x > 0 or y > 0):
                px = int(np.clip(x * scale_x, 0, width - 1))
                py = int(np.clip(y * scale_y, 0, height - 1))
                pts.append((px, py, c))
            else:
                pts.append(None)

        # 1. Draw Body Bones
        for i1, i2 in BODY_CONNECTIONS:
            p1, p2 = pts[i1], pts[i2]
            if p1 is not None and p2 is not None:
                cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), self.color_pose, 2, cv2.LINE_AA)

        # 2. Draw Facial Contours (Face: 25 to 94)
        face_offset = FACE_SLICE.start
        for i1, i2 in FACE_CONNECTIONS:
            p1, p2 = pts[face_offset + i1], pts[face_offset + i2]
            if p1 is not None and p2 is not None:
                cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), self.color_face, 1, cv2.LINE_AA)

        # 3. Draw Hand Bones (Left Hand: 95 to 115)
        lh_offset = HAND_LEFT_SLICE.start
        for i1, i2 in HAND_CONNECTIONS:
            p1, p2 = pts[lh_offset + i1], pts[lh_offset + i2]
            if p1 is not None and p2 is not None:
                cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), self.color_hand_left, 2, cv2.LINE_AA)

        # 4. Draw Hand Bones (Right Hand: 116 to 136)
        rh_offset = HAND_RIGHT_SLICE.start
        for i1, i2 in HAND_CONNECTIONS:
            p1, p2 = pts[rh_offset + i1], pts[rh_offset + i2]
            if p1 is not None and p2 is not None:
                cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), self.color_hand_right, 2, cv2.LINE_AA)

        # 5. Draw Joint Circles
        for i, p in enumerate(pts):
            if p is None:
                continue
            x, y, _ = p
            if i < POSE_SLICE.stop:
                cv2.circle(img, (x, y), 4, (0, 200, 0), -1, cv2.LINE_AA)
            elif i < FACE_SLICE.stop:
                cv2.circle(img, (x, y), 2, self.color_face, -1, cv2.LINE_AA)
            elif i < HAND_LEFT_SLICE.stop:
                cv2.circle(img, (x, y), 3, self.color_hand_left, -1, cv2.LINE_AA)
            else:
                cv2.circle(img, (x, y), 3, self.color_hand_right, -1, cv2.LINE_AA)

        return img

    def create_side_by_side(
        self,
        raw_frame: np.ndarray,
        keypoints: np.ndarray,
        frame_idx: int,
        total_frames: int,
        scale: float = 0.5,
    ) -> np.ndarray:
        """
        Creates a side-by-side comparison:
        [Panel 1: Original Video] | [Panel 2: Superimposed Skeleton with HUD]
        """
        h, w, _ = raw_frame.shape

        # Superimposed frame
        overlay = self.draw_skeleton(raw_frame, keypoints, w, h, normalized=True)

        # Add HUD annotations to overlay
        info_text = f"Frame: {frame_idx + 1:02d}/{total_frames:02d}"
        cv2.putText(overlay, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        # Legend
        cv2.putText(overlay, "Pose", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color_pose, 2, cv2.LINE_AA)
        cv2.putText(overlay, "Face", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color_face, 2, cv2.LINE_AA)
        cv2.putText(overlay, "Hand R", (180, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color_hand_right, 2, cv2.LINE_AA)
        cv2.putText(overlay, "Hand L", (280, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color_hand_left, 2, cv2.LINE_AA)

        # Add label to raw frame
        raw_labeled = raw_frame.copy()
        cv2.putText(raw_labeled, "Original Video", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

        # Concatenate horizontally
        combined = np.hstack([raw_labeled, overlay])

        if scale != 1.0:
            new_w = int(combined.shape[1] * scale)
            new_h = int(combined.shape[0] * scale)
            combined = cv2.resize(combined, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return combined

    def export_gif(
        self,
        npy_path: Union[str, Path],
        video_path: Union[str, Path],
        output_gif_path: Union[str, Path],
        fps: int = 10,
        scale: float = 0.5,
    ) -> Path:
        """Generates an animated GIF comparing original video with keypoint detection."""
        tensor = np.load(npy_path)
        cap = cv2.VideoCapture(str(video_path))

        frames_pil = []
        num_frames = tensor.shape[0]

        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            sbs = self.create_side_by_side(frame, tensor[i], i, num_frames, scale=scale)
            # Convert BGR (OpenCV) to RGB (PIL)
            rgb = cv2.cvtColor(sbs, cv2.COLOR_BGR2RGB)
            frames_pil.append(Image.fromarray(rgb))

        cap.release()

        out_path = Path(output_gif_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if frames_pil:
            duration_ms = int(1000 / fps)
            frames_pil[0].save(
                out_path,
                save_all=True,
                append_images=frames_pil[1:],
                optimize=True,
                duration=duration_ms,
                loop=0,
            )

        return out_path

