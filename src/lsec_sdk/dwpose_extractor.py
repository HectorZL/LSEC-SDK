"""
DWPose / RTMW GPU-Accelerated Whole-Body Pose Estimator (OpenMMLab / ONNX Runtime).
Processes webcam or video frames using an NVIDIA GPU via CUDAExecutionProvider,
delivering robust tracking for hands in side-view / profile angles.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

# Ensure PyTorch CUDA & cuDNN DLLs are visible to ONNX Runtime
try:
    import torch
    torch_lib = Path(torch.__file__).parent / "lib"
    if torch_lib.exists():
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(torch_lib))
        os.environ["PATH"] = str(torch_lib) + ";" + os.environ.get("PATH", "")
except Exception:
    pass

import onnxruntime as ort

from .constants import (
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    FACE_SLICE,
    HAND_LEFT_SLICE,
    HAND_RIGHT_SLICE,
    POSE_SLICE,
    TOTAL_KEYPOINTS_COUNT,
)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.45) -> List[int]:
    """Pure NumPy Non-Maximum Suppression for bounding boxes [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


class DWPoseDetector:
    """
    GPU-accelerated DWPose Whole-Body Detector using YOLOX + DWPose.
    Delivers robust 133-point whole-body tracking even during profile/side hand angles.
    """

    def __init__(
        self,
        det_model: Union[str, Path] = "models/dwpose/yolox_l.onnx",
        pose_model: Union[str, Path] = "models/dwpose/dw-ll_ucoco_384.onnx",
        providers: Optional[List[str]] = None,
    ):
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.det_sess = ort.InferenceSession(str(det_model), providers=providers)
        self.pose_sess = ort.InferenceSession(str(pose_model), providers=providers)
        self.provider_name = self.pose_sess.get_providers()[0]

        # Model input dimensions
        self.det_h, self.det_w = 640, 640
        self.pose_h, self.pose_w = 384, 288

    def _preprocess_yolox(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        """Letterbox resizing for YOLOX detector."""
        h, w = img.shape[:2]
        r = min(self.det_h / h, self.det_w / w)
        rw, rh = int(w * r), int(h * r)
        resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
        padded = np.full((self.det_h, self.det_w, 3), 114, dtype=np.uint8)
        padded[:rh, :rw] = resized
        # Transpose to (1, 3, 640, 640) float32
        inp = padded.transpose(2, 0, 1)[None].astype(np.float32)
        return inp, r

    def _postprocess_yolox(self, output: np.ndarray, r: float, conf_thresh: float = 0.40) -> Optional[np.ndarray]:
        """Decodes YOLOX detections and returns largest person bounding box [x1, y1, x2, y2]."""
        # output shape: (1, 8400, 85)
        preds = output[0]
        # YOLOX anchor grids
        grids = []
        strides = [8, 16, 32]
        for stride in strides:
            h_s, w_s = self.det_h // stride, self.det_w // stride
            xv, yv = np.meshgrid(np.arange(w_s), np.arange(h_s))
            grid = np.stack((xv, yv), 2).reshape((-1, 2))
            grids.append(grid)
        grids = np.concatenate(grids, axis=0)

        # Decode box centers & dimensions
        box_xy = (preds[:, :2] + grids) * np.repeat(strides, [80*80, 40*40, 20*20])[:, None]
        box_wh = np.exp(preds[:, 2:4]) * np.repeat(strides, [80*80, 40*40, 20*20])[:, None]

        scores = preds[:, 4] * preds[:, 5]  # person class is index 5 (first class)
        mask = scores > conf_thresh
        if not np.any(mask):
            return None

        boxes = np.zeros((np.sum(mask), 4), dtype=np.float32)
        valid_xy = box_xy[mask]
        valid_wh = box_wh[mask]
        valid_sc = scores[mask]

        boxes[:, 0] = (valid_xy[:, 0] - valid_wh[:, 0] / 2) / r
        boxes[:, 1] = (valid_xy[:, 1] - valid_wh[:, 1] / 2) / r
        boxes[:, 2] = (valid_xy[:, 0] + valid_wh[:, 0] / 2) / r
        boxes[:, 3] = (valid_xy[:, 1] + valid_wh[:, 1] / 2) / r

        keep = nms(boxes, valid_sc)
        if not keep:
            return None

        # Return the largest person bbox
        areas = (boxes[keep, 2] - boxes[keep, 0]) * (boxes[keep, 3] - boxes[keep, 1])
        best_idx = keep[np.argmax(areas)]
        return boxes[best_idx]

    def _preprocess_pose(self, img: np.ndarray, bbox: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Crops person bounding box with aspect ratio adjustment and normalization for DWPose."""
        img_h, img_w = img.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = max(10, x2 - x1), max(10, y2 - y1)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        # Maintain 384x288 aspect ratio (4:3) with 15% margin
        aspect_ratio = self.pose_w / self.pose_h  # 288 / 384 = 0.75
        if bw > bh * aspect_ratio:
            bh = bw / aspect_ratio
        else:
            bw = bh * aspect_ratio
        bw *= 1.25
        bh *= 1.25

        scale = np.array([bw, bh], dtype=np.float32)
        center = np.array([cx, cy], dtype=np.float32)

        # Affine crop
        src_pts = np.float32([
            [cx - bw / 2, cy - bh / 2],
            [cx + bw / 2, cy - bh / 2],
            [cx + bw / 2, cy + bh / 2],
        ])
        dst_pts = np.float32([
            [0, 0],
            [self.pose_w, 0],
            [self.pose_w, self.pose_h],
        ])
        trans = cv2.getAffineTransform(src_pts, dst_pts)
        crop = cv2.warpAffine(img, trans, (self.pose_w, self.pose_h), flags=cv2.INTER_LINEAR)

        # ImageNet normalization
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        norm_crop = (crop.astype(np.float32) - mean) / std
        inp = norm_crop.transpose(2, 0, 1)[None].astype(np.float32)

        return inp, center, scale

    def _decode_simcc(
        self,
        simcc_x: np.ndarray,
        simcc_y: np.ndarray,
        center: np.ndarray,
        scale: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Decodes SimCC spatial outputs into pixel coordinates and confidence scores."""
        # simcc_x: (1, 133, 576), simcc_y: (1, 133, 768)
        x_locs = np.argmax(simcc_x[0], axis=1).astype(np.float32) / 2.0
        y_locs = np.argmax(simcc_y[0], axis=1).astype(np.float32) / 2.0

        x_max = np.max(simcc_x[0], axis=1)
        y_max = np.max(simcc_y[0], axis=1)
        scores = np.minimum(x_max, y_max)

        # Map from 288x384 back to original image space
        kpts = np.zeros((133, 2), dtype=np.float32)
        kpts[:, 0] = (x_locs / self.pose_w) * scale[0] + (center[0] - scale[0] / 2.0)
        kpts[:, 1] = (y_locs / self.pose_h) * scale[1] + (center[1] - scale[1] / 2.0)

        return kpts, scores

    def process(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Runs full DWPose GPU pipeline on an input frame (BGR).
        Returns (keypoints_133x2, scores_133) in pixel coordinates, or None if no person detected.
        """
        inp_det, r = self._preprocess_yolox(frame)
        det_out = self.det_sess.run(None, {"images": inp_det})
        bbox = self._postprocess_yolox(det_out[0], r)

        if bbox is None:
            # Fallback to center-biased frame crop if YOLOX was too strict
            h, w = frame.shape[:2]
            bbox = np.array([w * 0.1, h * 0.05, w * 0.9, h * 0.95], dtype=np.float32)

        inp_pose, center, scale = self._preprocess_pose(frame, bbox)
        pose_out = self.pose_sess.run(None, {"input": inp_pose})

        simcc_x = pose_out[0]
        simcc_y = pose_out[1]

        kpts, scores = self._decode_simcc(simcc_x, simcc_y, center, scale)
        return kpts, scores

    def extract_lsec_tensor(self, frame: np.ndarray) -> np.ndarray:
        """
        Extracts keypoints and normalizes them into LSEC SDK standard (137, 3) tensor format [x, y, confidence].
        Normalized to [0, 1] range.
        """
        h, w = frame.shape[:2]
        res = self.process(frame)
        tensor = np.zeros((TOTAL_KEYPOINTS_COUNT, 3), dtype=np.float32)

        if res is None:
            return tensor

        kpts, scores = res

        # 1. Pose: DWPose (18 body joints) -> OpenPose 25 layout
        # 0: Nose, 1: Neck, 2: RShoulder, 3: RElbow, 4: RWrist, 5: LShoulder, 6: LElbow, 7: LWrist...
        for i in range(min(18, len(kpts))):
            tensor[i] = [kpts[i, 0] / w, kpts[i, 1] / h, float(scores[i])]

        # 2. Face: DWPose 68 landmarks (24..92) -> LSEC SDK 70 face points
        if len(kpts) >= 92:
            face_kpts = kpts[24:92]
            face_sc = scores[24:92]
            for i in range(min(68, len(face_kpts))):
                tensor[FACE_SLICE.start + i] = [face_kpts[i, 0] / w, face_kpts[i, 1] / h, float(face_sc[i])]

        # 3. Left Hand: DWPose 21 landmarks (92..113)
        if len(kpts) >= 113:
            lh_kpts = kpts[92:113]
            lh_sc = scores[92:113]
            for i in range(21):
                tensor[HAND_LEFT_SLICE.start + i] = [lh_kpts[i, 0] / w, lh_kpts[i, 1] / h, float(lh_sc[i])]

        # 4. Right Hand: DWPose 21 landmarks (113..134)
        if len(kpts) >= 134:
            rh_kpts = kpts[113:134]
            rh_sc = scores[113:134]
            for i in range(21):
                tensor[HAND_RIGHT_SLICE.start + i] = [rh_kpts[i, 0] / w, rh_kpts[i, 1] / h, float(rh_sc[i])]

        return tensor

    def draw_skeleton(self, frame: np.ndarray, kpts: np.ndarray, scores: np.ndarray, thresh: float = 0.25) -> np.ndarray:
        """Visualizes the 133 DWPose whole-body skeleton with distinct colors for body, hands, and face."""
        # 1. Pose connections
        body_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4),   # Right arm
            (1, 5), (5, 6), (6, 7),           # Left arm
            (1, 8), (8, 9), (9, 10),          # Right leg
            (1, 11), (11, 12), (12, 13),      # Left leg
            (0, 14), (14, 16), (0, 15), (15, 17) # Face outline
        ]
        for p1, p2 in body_edges:
            if p1 < len(scores) and p2 < len(scores):
                if scores[p1] > thresh and scores[p2] > thresh:
                    pt1 = (int(kpts[p1, 0]), int(kpts[p1, 1]))
                    pt2 = (int(kpts[p2, 0]), int(kpts[p2, 1]))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # 2. Hand connections
        hand_edges = [
            (0,1), (1,2), (2,3), (3,4),
            (0,5), (5,6), (6,7), (7,8),
            (5,9), (9,10), (10,11), (11,12),
            (9,13), (13,14), (14,15), (15,16),
            (13,17), (17,18), (18,19), (19,20),
            (0,17)
        ]
        # Left hand (92..113) - Magenta
        for e1, e2 in hand_edges:
            p1, p2 = 92 + e1, 92 + e2
            if p1 < len(scores) and p2 < len(scores):
                if scores[p1] > thresh and scores[p2] > thresh:
                    pt1 = (int(kpts[p1, 0]), int(kpts[p1, 1]))
                    pt2 = (int(kpts[p2, 0]), int(kpts[p2, 1]))
                    cv2.line(frame, pt1, pt2, (255, 0, 255), 2)

        # Right hand (113..134) - Cyan
        for e1, e2 in hand_edges:
            p1, p2 = 113 + e1, 113 + e2
            if p1 < len(scores) and p2 < len(scores):
                if scores[p1] > thresh and scores[p2] > thresh:
                    pt1 = (int(kpts[p1, 0]), int(kpts[p1, 1]))
                    pt2 = (int(kpts[p2, 0]), int(kpts[p2, 1]))
                    cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

        # Draw hand joints
        for i in range(92, min(134, len(kpts))):
            if scores[i] > thresh:
                color = (255, 100, 255) if i < 113 else (100, 255, 255)
                cv2.circle(frame, (int(kpts[i, 0]), int(kpts[i, 1])), 3, color, -1)

        return frame
