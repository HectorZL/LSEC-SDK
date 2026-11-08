"""
DWPose / RTMW GPU-Accelerated Whole-Body Pose Estimator (OpenMMLab / ONNX Runtime).
Processes webcam or video frames using an NVIDIA GPU via CUDAExecutionProvider,
delivering robust tracking for hands in side-view / profile angles.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
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

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

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
        enable_facemesh: bool = True,
    ):
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.det_sess = ort.InferenceSession(str(det_model), providers=providers)
        self.pose_sess = ort.InferenceSession(str(pose_model), providers=providers)
        self.provider_name = self.pose_sess.get_providers()[0]

        # Model input dimensions
        self.det_h, self.det_w = 640, 640
        self.pose_h, self.pose_w = 384, 288

        # High-Fidelity Facial Tracking via MediaPipe FaceMesh
        self.enable_facemesh = enable_facemesh and HAS_MEDIAPIPE
        self.face_mesh = None
        self.last_face_mesh_landmarks = None
        self.face_mesh_cache = None
        self.face_mesh_cache_ttl = 0
        if self.enable_facemesh:
            try:
                self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=3,
                    refine_landmarks=True,
                    min_detection_confidence=0.30,
                    min_tracking_confidence=0.30,
                )
            except Exception as e:
                print(f"[DWPoseDetector] MediaPipe FaceMesh no disponible: {e}")
                self.enable_facemesh = False

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

        # Update FaceMesh landmarks on RGB frame with spatial anchor validation
        # Ensures FaceMesh attaches ONLY to the user's head, rejecting ghost faces on chairs, logos or background
        if self.enable_facemesh and self.face_mesh is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                fm_out = self.face_mesh.process(rgb)
                valid_face = None

                if fm_out and fm_out.multi_face_landmarks:
                    h, w = frame.shape[:2]
                    ref_pt = None

                    # Use DWPose nose (0) or shoulder midpoint (5, 6) as anatomical anchor
                    if len(scores) > 0 and scores[0] > 0.15:
                        ref_pt = np.array([kpts[0, 0], kpts[0, 1]])
                    elif len(scores) > 6 and (scores[5] > 0.15 and scores[6] > 0.15):
                        ref_pt = np.array([(kpts[5, 0] + kpts[6, 0]) / 2.0, min(kpts[5, 1], kpts[6, 1]) - 80])

                    max_allowed_dist = 180.0
                    if len(scores) > 6 and scores[5] > 0.15 and scores[6] > 0.15:
                        shoulder_dist = float(np.linalg.norm(kpts[5] - kpts[6]))
                        max_allowed_dist = max(110.0, shoulder_dist * 0.90)

                    if ref_pt is not None:
                        best_face = None
                        min_dist = float('inf')

                        for candidate_face in fm_out.multi_face_landmarks:
                            lms = candidate_face.landmark
                            c_x = (lms[1].x + lms[4].x + lms[168].x) / 3.0 * w
                            c_y = (lms[1].y + lms[4].y + lms[168].y) / 3.0 * h
                            dist = float(np.linalg.norm(ref_pt - np.array([c_x, c_y])))

                            if dist < min_dist:
                                min_dist = dist
                                best_face = candidate_face

                        # Only accept face if it aligns with the user's physical head (not a chair or poster)
                        if min_dist <= max_allowed_dist:
                            valid_face = best_face
                    else:
                        valid_face = fm_out.multi_face_landmarks[0]

                if valid_face is not None:
                    self.face_mesh_cache = valid_face
                    self.face_mesh_cache_ttl = 8
                    self.last_face_mesh_landmarks = valid_face
                elif self.face_mesh_cache_ttl > 0:
                    self.face_mesh_cache_ttl -= 1
                    self.last_face_mesh_landmarks = self.face_mesh_cache
                else:
                    self.face_mesh_cache = None
                    self.last_face_mesh_landmarks = None
            except Exception:
                self.last_face_mesh_landmarks = None
        else:
            self.last_face_mesh_landmarks = None

        return kpts, scores

    def extract_lsec_tensor(
        self,
        frame: np.ndarray,
        precomputed_res: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Extracts keypoints and normalizes them into LSEC SDK standard (137, 3) tensor format [x, y, confidence].
        Properly maps COCO-WholeBody (133) joints to OpenPose 25 layout and handles 1-handed or 2-handed gestures.
        Accepts precomputed DWPose results to eliminate redundant GPU inference passes.
        """
        h, w = frame.shape[:2]
        res = precomputed_res if precomputed_res is not None else self.process(frame)
        tensor = np.zeros((TOTAL_KEYPOINTS_COUNT, 3), dtype=np.float32)

        if res is None:
            return tensor

        kpts, scores = res

        # 1. Pose: Map COCO 17 body keypoints to OpenPose 25 layout
        # COCO: 0: Nose, 1: LEye, 2: REye, 3: LEar, 4: REar,
        #       5: LShoulder, 6: RShoulder, 7: LElbow, 8: RElbow,
        #       9: LWrist, 10: RWrist, 11: LHip, 12: RHip,
        #       13: LKnee, 14: RKnee, 15: LAnkle, 16: RAnkle

        # OpenPose 0: Nose
        if len(kpts) > 0 and scores[0] > 0.1:
            tensor[0] = [kpts[0, 0] / w, kpts[0, 1] / h, float(scores[0])]

        # OpenPose 1: Neck (midpoint of LShoulder 5 and RShoulder 6)
        if len(kpts) > 6 and scores[5] > 0.1 and scores[6] > 0.1:
            neck_x = (kpts[5, 0] + kpts[6, 0]) / 2.0 / w
            neck_y = (kpts[5, 1] + kpts[6, 1]) / 2.0 / h
            neck_c = min(float(scores[5]), float(scores[6]))
            tensor[1] = [neck_x, neck_y, neck_c]

        # OpenPose 2: RShoulder (COCO 6)
        if len(kpts) > 6:
            tensor[2] = [kpts[6, 0] / w, kpts[6, 1] / h, float(scores[6])]
        # OpenPose 3: RElbow (COCO 8)
        if len(kpts) > 8:
            tensor[3] = [kpts[8, 0] / w, kpts[8, 1] / h, float(scores[8])]
        # OpenPose 4: RWrist (COCO 10)
        if len(kpts) > 10:
            tensor[4] = [kpts[10, 0] / w, kpts[10, 1] / h, float(scores[10])]

        # OpenPose 5: LShoulder (COCO 5)
        if len(kpts) > 5:
            tensor[5] = [kpts[5, 0] / w, kpts[5, 1] / h, float(scores[5])]
        # OpenPose 6: LElbow (COCO 7)
        if len(kpts) > 7:
            tensor[6] = [kpts[7, 0] / w, kpts[7, 1] / h, float(scores[7])]
        # OpenPose 7: LWrist (COCO 9)
        if len(kpts) > 9:
            tensor[7] = [kpts[9, 0] / w, kpts[9, 1] / h, float(scores[9])]

        # OpenPose 8: MidHip (midpoint of LHip 11 and RHip 12)
        if len(kpts) > 12 and scores[11] > 0.1 and scores[12] > 0.1:
            mhip_x = (kpts[11, 0] + kpts[12, 0]) / 2.0 / w
            mhip_y = (kpts[11, 1] + kpts[12, 1]) / 2.0 / h
            mhip_c = min(float(scores[11]), float(scores[12]))
            tensor[8] = [mhip_x, mhip_y, mhip_c]

        # OpenPose 9: RHip (COCO 12), 10: RKnee (COCO 14), 11: RAnkle (COCO 16)
        if len(kpts) > 16:
            tensor[9] = [kpts[12, 0] / w, kpts[12, 1] / h, float(scores[12])]
            tensor[10] = [kpts[14, 0] / w, kpts[14, 1] / h, float(scores[14])]
            tensor[11] = [kpts[16, 0] / w, kpts[16, 1] / h, float(scores[16])]

        # OpenPose 12: LHip (COCO 11), 13: LKnee (COCO 13), 14: LAnkle (COCO 15)
        if len(kpts) > 15:
            tensor[12] = [kpts[11, 0] / w, kpts[11, 1] / h, float(scores[11])]
            tensor[13] = [kpts[13, 0] / w, kpts[13, 1] / h, float(scores[13])]
            tensor[14] = [kpts[15, 0] / w, kpts[15, 1] / h, float(scores[15])]

        # OpenPose 15: REye (COCO 2), 16: LEye (COCO 1), 17: REar (COCO 4), 18: LEar (COCO 3)
        if len(kpts) > 4:
            tensor[15] = [kpts[2, 0] / w, kpts[2, 1] / h, float(scores[2])]
            tensor[16] = [kpts[1, 0] / w, kpts[1, 1] / h, float(scores[1])]
            tensor[17] = [kpts[4, 0] / w, kpts[4, 1] / h, float(scores[4])]
            tensor[18] = [kpts[3, 0] / w, kpts[3, 1] / h, float(scores[3])]

        # 2. Face: COCO WholeBody 68 landmarks (23..91) -> LSEC SDK 70 points
        if len(kpts) >= 91:
            for i in range(min(68, len(kpts) - 23)):
                tensor[FACE_SLICE.start + i] = [kpts[23 + i, 0] / w, kpts[23 + i, 1] / h, float(scores[23 + i])]

        # 3. Left Hand: COCO WholeBody 21 landmarks (91..112) -> LSEC SDK 95..116
        if len(kpts) >= 112:
            for i in range(21):
                tensor[HAND_LEFT_SLICE.start + i] = [kpts[91 + i, 0] / w, kpts[91 + i, 1] / h, float(scores[91 + i])]

        # 4. Right Hand: COCO WholeBody 21 landmarks (112..133) -> LSEC SDK 116..137
        if len(kpts) >= 133:
            for i in range(21):
                tensor[HAND_RIGHT_SLICE.start + i] = [kpts[112 + i, 0] / w, kpts[112 + i, 1] / h, float(scores[112 + i])]

        return tensor

    def draw_skeleton(
        self,
        frame: np.ndarray,
        kpts: np.ndarray,
        scores: np.ndarray,
        thresh: float = 0.25,
        face_landmarks: Optional[Any] = None,
    ) -> np.ndarray:
        """Visualizes the 133 DWPose whole-body skeleton with true anatomical topology and high-fidelity face tracking."""
        # 1. Body pose connections (COCO 17 layout - NO lines cutting across eyes/nose)
        body_edges = [
            (5, 6),                # Shoulders
            (5, 7), (7, 9),        # Left arm: Shoulder -> Elbow -> Wrist
            (6, 8), (8, 10),       # Right arm: Shoulder -> Elbow -> Wrist
            (5, 11), (6, 12),      # Torso: Shoulders to Hips
            (11, 12),              # Hips
            (11, 13), (13, 15),    # Left leg
            (12, 14), (14, 16),    # Right leg
        ]
        for p1, p2 in body_edges:
            if p1 < len(scores) and p2 < len(scores):
                if scores[p1] > thresh and scores[p2] > thresh:
                    pt1 = (int(kpts[p1, 0]), int(kpts[p1, 1]))
                    pt2 = (int(kpts[p2, 0]), int(kpts[p2, 1]))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw anatomical neck line from shoulder midpoint to throat base (never touches face or chin)
        if len(scores) > 6 and scores[5] > thresh and scores[6] > thresh:
            neck_pt = (int((kpts[5, 0] + kpts[6, 0]) / 2.0), int((kpts[5, 1] + kpts[6, 1]) / 2.0))
            if len(scores) > 31 and scores[31] > thresh:
                # Stop 25% below chin to ensure clean anatomical gap between neck and jawline
                chin_y = kpts[31, 1]
                throat_y = int(neck_pt[1] - 0.75 * (neck_pt[1] - chin_y))
                throat_x = int(neck_pt[0] + 0.75 * (kpts[31, 0] - neck_pt[0]))
                cv2.line(frame, neck_pt, (throat_x, throat_y), (0, 255, 0), 2, cv2.LINE_AA)
            elif scores[0] > thresh:
                throat_y = int(neck_pt[1] - 0.35 * (neck_pt[1] - kpts[0, 1]))
                cv2.line(frame, neck_pt, (neck_pt[0], throat_y), (0, 255, 0), 2, cv2.LINE_AA)

        # Draw body joints (shoulders, elbows, wrists, hips, knees, ankles)
        for i in range(5, min(17, len(kpts))):
            if scores[i] > thresh:
                cv2.circle(frame, (int(kpts[i, 0]), int(kpts[i, 1])), 3, (0, 220, 0), -1, cv2.LINE_AA)

        # 2. Face: High-Fidelity Facial Visualization (Hybrid FaceMesh or DWPose 68 anatomical contours)
        fm = face_landmarks if face_landmarks is not None else self.last_face_mesh_landmarks
        if fm is not None and HAS_MEDIAPIPE:
            try:
                mp_drawing = mp.solutions.drawing_utils
                mp_face_mesh = mp.solutions.face_mesh
                
                # Cohesive aesthetic: elegant cyan contours and soft mint irises (no red/green clown eyes)
                contour_spec = mp_drawing.DrawingSpec(color=(0, 230, 255), thickness=1, circle_radius=1)
                iris_spec = mp_drawing.DrawingSpec(color=(0, 255, 180), thickness=1, circle_radius=1)

                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=fm,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=contour_spec,
                )
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=fm,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=iris_spec,
                )
            except Exception:
                fm = None

        if fm is None and len(kpts) >= 91:
            # Fallback to DWPose 68 clean facial contours
            FACE_EDGES = [
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
            face_offset = 23
            for (p1, p2) in FACE_EDGES:
                idx1 = face_offset + p1
                idx2 = face_offset + p2
                if idx1 < len(scores) and idx2 < len(scores):
                    if scores[idx1] > thresh and scores[idx2] > thresh:
                        pt1 = (int(kpts[idx1, 0]), int(kpts[idx1, 1]))
                        pt2 = (int(kpts[idx2, 0]), int(kpts[idx2, 1]))
                        cv2.line(frame, pt1, pt2, (0, 220, 255), 1, cv2.LINE_AA)

            for i in range(min(68, max(0, len(kpts) - face_offset))):
                idx = face_offset + i
                if scores[idx] > thresh:
                    cv2.circle(frame, (int(kpts[idx, 0]), int(kpts[idx, 1])), 1, (255, 200, 0), -1, cv2.LINE_AA)

        # 3. Hand connections (Left: Magenta, Right: Cyan)
        hand_edges = [
            (0,1), (1,2), (2,3), (3,4),
            (0,5), (5,6), (6,7), (7,8),
            (5,9), (9,10), (10,11), (11,12),
            (9,13), (13,14), (14,15), (15,16),
            (13,17), (17,18), (18,19), (19,20),
            (0,17)
        ]
        # Left hand (91..112) - Magenta
        for e1, e2 in hand_edges:
            p1, p2 = 91 + e1, 91 + e2
            if p1 < len(scores) and p2 < len(scores):
                if scores[p1] > thresh and scores[p2] > thresh:
                    pt1 = (int(kpts[p1, 0]), int(kpts[p1, 1]))
                    pt2 = (int(kpts[p2, 0]), int(kpts[p2, 1]))
                    cv2.line(frame, pt1, pt2, (255, 0, 255), 2, cv2.LINE_AA)

        # Right hand (112..133) - Cyan
        for e1, e2 in hand_edges:
            p1, p2 = 112 + e1, 112 + e2
            if p1 < len(scores) and p2 < len(scores):
                if scores[p1] > thresh and scores[p2] > thresh:
                    pt1 = (int(kpts[p1, 0]), int(kpts[p1, 1]))
                    pt2 = (int(kpts[p2, 0]), int(kpts[p2, 1]))
                    cv2.line(frame, pt1, pt2, (0, 255, 255), 2, cv2.LINE_AA)

        # Draw hand joints
        for i in range(91, min(133, len(kpts))):
            if scores[i] > thresh:
                color = (255, 100, 255) if i < 112 else (100, 255, 255)
                cv2.circle(frame, (int(kpts[i, 0]), int(kpts[i, 1])), 3, color, -1, cv2.LINE_AA)

        return frame
