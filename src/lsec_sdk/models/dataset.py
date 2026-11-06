"""Dataset and preprocessing for sign language sequence tensors."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class LabelEncoder:
    """Encodes categorical string labels into integers and vice versa."""

    def __init__(self, labels: Optional[List[str]] = None):
        self.label_to_idx: Dict[str, int] = {}
        self.idx_to_label: Dict[int, str] = {}
        if labels:
            self.fit(labels)

    def fit(self, labels: List[str]) -> "LabelEncoder":
        unique_labels = sorted(list(set(labels)))
        self.label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
        self.idx_to_label = {i: lbl for i, lbl in enumerate(unique_labels)}
        return self

    def transform(self, label: str) -> int:
        return self.label_to_idx[label]

    def inverse_transform(self, idx: int) -> str:
        return self.idx_to_label.get(idx, f"UNKNOWN_{idx}")

    @property
    def num_classes(self) -> int:
        return len(self.label_to_idx)


def resample_sequence(sequence: np.ndarray, target_length: int) -> np.ndarray:
    """
    Interpolates a sequence of shape (T, ...) to exactly target_length frames
    using linear interpolation along the time dimension.
    """
    orig_len = sequence.shape[0]
    if orig_len == target_length:
        return sequence

    orig_indices = np.linspace(0, 1, orig_len)
    target_indices = np.linspace(0, 1, target_length)

    # Flatten trailing dimensions for interpolation
    trailing_shape = sequence.shape[1:]
    flattened = sequence.reshape(orig_len, -1)
    resampled_flat = np.empty((target_length, flattened.shape[1]), dtype=sequence.dtype)

    for col in range(flattened.shape[1]):
        resampled_flat[:, col] = np.interp(target_indices, orig_indices, flattened[:, col])

    return resampled_flat.reshape((target_length, *trailing_shape))


class SignSequenceDataset(Dataset):
    """
    PyTorch Dataset for sign language keypoint sequences.
    Loads .npy sequence tensors and prepares them for training or evaluation.
    """

    def __init__(
        self,
        samples: List[Tuple[Union[str, Path], str]],
        label_encoder: Optional[LabelEncoder] = None,
        target_frames: int = 32,
        augment: bool = False,
        replicate_factor: int = 1,
    ):
        """
        Args:
            samples: List of tuples (npy_file_path, string_label).
            label_encoder: Fitted LabelEncoder or None to fit automatically.
            target_frames: Fixed sequence length after temporal interpolation.
            augment: Whether to apply data augmentation (jitter, scaling).
            replicate_factor: Number of augmented copies per sample per epoch.
        """
        self.base_samples = [(Path(p), lbl) for p, lbl in samples]
        self.target_frames = target_frames
        self.augment = augment
        self.replicate_factor = max(1, replicate_factor)

        labels = [lbl for _, lbl in self.base_samples]
        if label_encoder is None:
            self.label_encoder = LabelEncoder(labels)
        else:
            self.label_encoder = label_encoder

    @classmethod
    def from_metadata_csv(
        cls,
        csv_path: Union[str, Path],
        target_frames: int = 32,
        augment: bool = False,
        replicate_factor: int = 1,
    ) -> "SignSequenceDataset":
        """Loads labeled samples from a metadata CSV file."""
        df = pd.read_csv(csv_path)
        if "label" not in df.columns:
            raise ValueError("CSV must contain a 'label' column.")
        labeled_df = df[df["label"].notna() & (df["label"].str.strip() != "")]
        samples = [(row["npy_path"], row["label"]) for _, row in labeled_df.iterrows()]
        return cls(samples, target_frames=target_frames, augment=augment, replicate_factor=replicate_factor)

    def __len__(self) -> int:
        return len(self.base_samples) * self.replicate_factor

    def _apply_augmentation(self, seq: np.ndarray) -> np.ndarray:
        """Applies temporal slicing, spatial scaling, translation and jitter augmentation."""
        augmented = seq.copy()
        n_frames = len(augmented)

        # 1. Random temporal sub-slice if sequence is long enough (> target_frames)
        if n_frames > self.target_frames + 6:
            # Crop between 85% and 100% of length
            crop_len = int(n_frames * np.random.uniform(0.85, 1.0))
            max_start = max(1, n_frames - crop_len)
            start_idx = np.random.randint(0, max_start)
            augmented = augmented[start_idx : start_idx + crop_len]

        # 2. Random scaling [0.93, 1.07]
        scale = np.random.uniform(0.93, 1.07)
        augmented[:, :, :2] *= scale

        # 3. Small random shift (translation) [-0.02, 0.02]
        shift_x = np.random.uniform(-0.02, 0.02)
        shift_y = np.random.uniform(-0.02, 0.02)
        nonzero_mask = augmented[:, :, 2] > 0
        augmented[:, :, 0][nonzero_mask] += shift_x
        augmented[:, :, 1][nonzero_mask] += shift_y

        # 4. Gaussian jitter
        noise = np.random.normal(0.0, 0.003, size=augmented[:, :, :2].shape).astype(np.float32)
        augmented[:, :, 0][nonzero_mask] += noise[:, :, 0][nonzero_mask]
        augmented[:, :, 1][nonzero_mask] += noise[:, :, 1][nonzero_mask]

        return augmented

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        base_idx = idx % len(self.base_samples)
        npy_path, label_str = self.base_samples[base_idx]
        seq = np.load(npy_path)  # Shape (T, 137, 3)

        if self.augment:
            seq = self._apply_augmentation(seq)

        # Temporal resample to target_frames
        resampled = resample_sequence(seq, self.target_frames)

        tensor = torch.from_numpy(resampled.astype(np.float32))
        label_idx = self.label_encoder.transform(label_str)

        return tensor, label_idx

