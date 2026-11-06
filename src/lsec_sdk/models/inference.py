"""Real-time sliding window inference engine for live sign recognition with multi-scale support and anti-flicker debouncing."""

from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple, Union
import numpy as np
import torch

from .dataset import LabelEncoder, resample_sequence
from .lstm_classifier import SignGRUClassifier, SignLSTMClassifier


def clean_ascii(text: str) -> str:
    """Sanitizes text removing accents, punctuation, and non-ASCII chars for robust comparison."""
    if text is None:
        return ""
    res = str(text)
    replacements = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
        "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
        "ñ": "n", "Ñ": "N",
        "¿": "", "¡": "", "?": "", "!": "", ",": "",
    }
    for k, v in replacements.items():
        res = res.replace(k, v)
    return res.encode("ascii", "ignore").decode("ascii")


class LiveInferenceEngine:
    """Processes streaming keypoint frames with multi-scale temporal evaluation and stability filtering."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        window_size: int = 32,
        min_prediction_confidence: float = 0.72,
        device: Optional[str] = None,
    ):
        self.window_size = window_size
        self.min_confidence = min_prediction_confidence

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.label_encoder = LabelEncoder()
        self.label_encoder.label_to_idx = ckpt["label_to_idx"]
        self.label_encoder.idx_to_label = {int(k): v for k, v in ckpt["idx_to_label"].items()}

        num_classes = len(self.label_encoder.label_to_idx)
        config = ckpt.get("config", {})

        # Reconstruct model
        model_name = ckpt.get("model_class", "SignLSTMClassifier")
        if model_name == "SignGRUClassifier":
            self.model = SignGRUClassifier(
                input_dim=config.get("input_dim", 411),
                hidden_dim=config.get("hidden_dim", 96),
                num_classes=num_classes,
            )
        else:
            self.model = SignLSTMClassifier(
                input_dim=config.get("input_dim", 411),
                hidden_dim=config.get("hidden_dim", 128),
                num_classes=num_classes,
                use_attention=config.get("use_attention", True),
            )

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Multi-scale buffer: stores up to 90 frames (~3.6s) to capture both short and long phrases
        self.buffer: Deque[np.ndarray] = deque(maxlen=90)
        self.prediction_history: Deque[str] = deque(maxlen=10)
        self.last_stable_prediction: Optional[str] = None
        self.consecutive_count: int = 0

        # Debouncing and emission tracking
        self.frame_counter: int = 0
        self.last_emitted_sign: Optional[str] = None
        self.last_emitted_frame: int = -999
        self.cooldown_frames: int = 25  # ~1.0 second before emitting a different sign

        # Fast lookup cache for clean target label to index
        self._target_index_cache: Dict[str, Optional[int]] = {}

    def reset(self):
        """Clears the temporal frame buffer and prediction state."""
        self.buffer.clear()
        self.prediction_history.clear()
        self.last_stable_prediction = None
        self.consecutive_count = 0
        self.last_emitted_sign = None
        self.last_emitted_frame = -999

    def _resolve_target_index(self, target_label: str) -> Optional[int]:
        """Resolves target sign name to class index with accent and token tolerance."""
        target_clean = clean_ascii(target_label).strip().upper()
        if target_clean in self._target_index_cache:
            return self._target_index_cache[target_clean]

        # 1. First priority: Exact cleaned match (avoids confusing 'DONDE' with 'DONDE ESTUDIO')
        for idx, lbl in self.label_encoder.idx_to_label.items():
            lbl_clean = clean_ascii(lbl).strip().upper()
            if lbl_clean == target_clean:
                self._target_index_cache[target_clean] = idx
                return idx

        # 2. Second priority: Substring match
        for idx, lbl in self.label_encoder.idx_to_label.items():
            lbl_clean = clean_ascii(lbl).strip().upper()
            if target_clean in lbl_clean or lbl_clean in target_clean:
                self._target_index_cache[target_clean] = idx
                return idx

        # 3. Third priority: Key token overlap
        t_words = [w for w in target_clean.split() if len(w) > 3]
        for idx, lbl in self.label_encoder.idx_to_label.items():
            l_words = [w for w in clean_ascii(lbl).strip().upper().split() if len(w) > 3]
            if any(tw in l_words for tw in t_words) or any(lw in t_words for lw in l_words):
                self._target_index_cache[target_clean] = idx
                return idx

        self._target_index_cache[target_clean] = None
        return None

    def get_target_confidence(self, target_label: str) -> float:
        """
        Evaluates the current multi-scale buffer specifically for the target sign probability.
        Searches across short (32), medium (56, 80), and full temporal scales with kinematic gating.
        """
        if len(self.buffer) < 16:
            return 0.0

        target_idx = self._resolve_target_index(target_label)
        if target_idx is None:
            return 0.0

        seq_arr = np.array(list(self.buffer), dtype=np.float32)

        # Kinematic Signing Space & Motion Gate:
        # Check wrist elevation (Pose keypoints: 4 RWrist, 7 LWrist)
        n_check = min(12, len(seq_arr))
        rw_y = seq_arr[-n_check:, 4, 1]
        lw_y = seq_arr[-n_check:, 7, 1]
        rw_c = seq_arr[-n_check:, 4, 2]
        lw_c = seq_arr[-n_check:, 7, 2]

        # In sign language, signs are articulated in the signing space (above lap/desk, Y < 0.82)
        r_elevated = bool(np.min(rw_y) < 0.82 and np.mean(rw_c) > 0.15)
        l_elevated = bool(np.min(lw_y) < 0.82 and np.mean(lw_c) > 0.15)

        # Check motion velocity in hands over the recent window
        rh_diff = np.diff(seq_arr[-n_check:, 116:137, :2], axis=0) if len(seq_arr) >= n_check else np.zeros((1, 21, 2))
        lh_diff = np.diff(seq_arr[-n_check:, 95:116, :2], axis=0) if len(seq_arr) >= n_check else np.zeros((1, 21, 2))
        rh_motion = float(np.mean(np.abs(rh_diff))) if len(rh_diff) > 0 else 0.0
        lh_motion = float(np.mean(np.abs(lh_diff))) if len(lh_diff) > 0 else 0.0
        active_motion = max(rh_motion, lh_motion)

        # If both wrists are in resting position (lap/table) or motionless:
        if not (r_elevated or l_elevated) or active_motion < 0.0030:
            return 0.0

        max_prob = 0.0
        # Evaluate multiple scales: short (32), medium (56, 80), and full buffer
        scales = [32]
        if len(seq_arr) >= 56:
            scales.append(56)
        if len(seq_arr) >= 80:
            scales.append(min(len(seq_arr), 90))

        for scale in set(scales):
            if scale > len(seq_arr) or scale < 16:
                continue
            sub = seq_arr[-scale:]
            resampled = resample_sequence(sub, self.window_size)
            inp = torch.from_numpy(resampled).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits, _ = self.model(inp)
                probs = torch.softmax(logits, dim=-1)
                p = float(probs[0, target_idx].item())
                if p > max_prob:
                    max_prob = p

        return max_prob

    def push_frame(self, frame_keypoints: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Pushes a new frame (137, 3) into the sliding window.
        Returns (predicted_label, probability) with temporal debouncing and anti-flicker cooldown.
        Applies kinematic signing space elevation and motion filtering to eliminate idle false positives.
        """
        self.frame_counter += 1
        self.buffer.append(frame_keypoints)

        if len(self.buffer) < max(16, self.window_size // 2):
            return None, 0.0

        seq_arr = np.array(list(self.buffer), dtype=np.float32)

        # 1. Kinematic Filter: Check signing space elevation and kinetic motion
        # (Supports both 1-handed and 2-handed Ecuadorian signs)
        n_check = min(12, len(seq_arr))
        rw_y = seq_arr[-n_check:, 4, 1]
        lw_y = seq_arr[-n_check:, 7, 1]
        rw_c = seq_arr[-n_check:, 4, 2]
        lw_c = seq_arr[-n_check:, 7, 2]

        r_elevated = bool(np.min(rw_y) < 0.82 and np.mean(rw_c) > 0.15)
        l_elevated = bool(np.min(lw_y) < 0.82 and np.mean(lw_c) > 0.15)

        rh_diff = np.diff(seq_arr[-n_check:, 116:137, :2], axis=0) if len(seq_arr) >= n_check else np.zeros((1, 21, 2))
        lh_diff = np.diff(seq_arr[-n_check:, 95:116, :2], axis=0) if len(seq_arr) >= n_check else np.zeros((1, 21, 2))
        rh_motion = float(np.mean(np.abs(rh_diff))) if len(rh_diff) > 0 else 0.0
        lh_motion = float(np.mean(np.abs(lh_diff))) if len(lh_diff) > 0 else 0.0
        active_motion = max(rh_motion, lh_motion)

        if not (r_elevated or l_elevated) or active_motion < 0.0030:
            # Hands are resting down or motionless: clear prediction state and emit silence
            self.last_emitted_sign = None
            self.last_stable_prediction = None
            self.prediction_history.clear()
            return None, 0.0

        # Multi-scale evaluation: evaluate last window_size frames and extended windows
        scales = [self.window_size]
        if len(seq_arr) >= 56:
            scales.append(56)
        if len(seq_arr) >= 80:
            scales.append(min(len(seq_arr), 90))

        best_label = None
        best_weighted_conf = 0.0
        raw_conf = 0.0

        for sc in scales:
            sub = seq_arr[-sc:]
            resampled = resample_sequence(sub, self.window_size)
            inp_tensor = torch.from_numpy(resampled).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits, _ = self.model(inp_tensor)
                probs = torch.softmax(logits, dim=-1)
                conf, pred_idx = probs.max(dim=-1)
                c_val = float(conf.item())

                # Weight longer temporal scales (56, 80+) higher because they capture
                # the complete holistic gesture rather than transitional intermediate arm positions
                scale_w = 1.15 if sc >= 56 else 1.0
                weighted = c_val * scale_w
                if weighted > best_weighted_conf:
                    best_weighted_conf = weighted
                    raw_conf = c_val
                    best_label = self.label_encoder.inverse_transform(int(pred_idx.item()))

        # Debouncing filter: require consistent predictions before emitting
        if best_label is not None and raw_conf >= self.min_confidence and best_label != "REPOSO":
            self.prediction_history.append(best_label)
        else:
            self.prediction_history.append("REPOSO")

        # Temporal debouncing window: candidate must be dominant in at least 7 of last 10 predictions
        # This prevents premature firing during arm transition and ensures the user actually executed the gesture
        recent = list(self.prediction_history)[-10:]
        non_reposo = [lbl for lbl in recent if lbl != "REPOSO"]

        if non_reposo:
            candidate = max(set(non_reposo), key=non_reposo.count)
            count = non_reposo.count(candidate)

            # Strict stability condition: at least 7 consistent frames out of 10
            if count >= 7:
                time_since_last = self.frame_counter - self.last_emitted_frame
                # Allow re-affirming same sign or emitting new sign after cooldown period
                if candidate == self.last_emitted_sign or time_since_last >= self.cooldown_frames:
                    self.last_emitted_sign = candidate
                    self.last_emitted_frame = self.frame_counter
                    self.last_stable_prediction = candidate
                    return candidate, min(1.0, raw_conf)

        return None, raw_conf
