"""Unit tests for lsec_sdk.models."""

from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from lsec_sdk.models import (
    LabelEncoder,
    LiveInferenceEngine,
    SignGRUClassifier,
    SignLSTMClassifier,
    SignModelTrainer,
    SignSequenceDataset,
    TemporalAttention,
    resample_sequence,
)


def test_label_encoder():
    labels = ["CÓMO TE SIENTES", "HOLA", "GRACIAS", "HOLA"]
    le = LabelEncoder(labels)
    assert le.num_classes == 3
    encoded = le.transform("CÓMO TE SIENTES")
    assert le.inverse_transform(encoded) == "CÓMO TE SIENTES"


def test_resample_sequence():
    orig_seq = np.random.randn(17, 137, 3).astype(np.float32)
    resampled = resample_sequence(orig_seq, target_length=32)
    assert resampled.shape == (32, 137, 3)

    # Test longer sequence downsampled
    long_seq = np.random.randn(112, 137, 3).astype(np.float32)
    resampled_down = resample_sequence(long_seq, target_length=32)
    assert resampled_down.shape == (32, 137, 3)


def test_temporal_attention():
    attn = TemporalAttention(hidden_dim=64)
    x = torch.randn(2, 10, 64)
    context, weights = attn(x)
    assert context.shape == (2, 64)
    assert weights.shape == (2, 10, 1)
    # Weights along time axis sum to 1.0
    assert torch.allclose(weights.sum(dim=1), torch.ones(2, 1), atol=1e-5)


def test_lstm_forward_pass():
    model = SignLSTMClassifier(input_dim=411, hidden_dim=64, num_classes=5)
    # Test with flattened 3D input (batch, time, 411)
    x3d = torch.randn(4, 32, 411)
    logits, weights = model(x3d)
    assert logits.shape == (4, 5)
    assert weights.shape == (4, 32, 1)

    # Test with 4D input (batch, time, 137, 3)
    x4d = torch.randn(4, 32, 137, 3)
    logits4d, _ = model(x4d)
    assert logits4d.shape == (4, 5)


def test_gru_forward_pass():
    model = SignGRUClassifier(input_dim=411, hidden_dim=48, num_classes=3)
    x = torch.randn(2, 20, 411)
    logits, _ = model(x)
    assert logits.shape == (2, 3)


def test_trainer_and_inference(tmp_path):
    # Create synthetic dataset with 2 classes
    num_samples = 8
    samples = []
    classes = ["HOLA", "GRACIAS"]

    for i in range(num_samples):
        lbl = classes[i % 2]
        fake_data = np.random.randn(25, 137, 3).astype(np.float32)
        npy_file = tmp_path / f"sample_{i}.npy"
        np.save(npy_file, fake_data)
        samples.append((npy_file, lbl))

    dataset = SignSequenceDataset(samples, target_frames=20)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = SignLSTMClassifier(input_dim=411, hidden_dim=32, num_classes=len(classes))
    trainer = SignModelTrainer(model, learning_rate=1e-2)

    loss, acc = trainer.train_epoch(loader)
    assert loss > 0.0

    # Save checkpoint
    ckpt_file = tmp_path / "model_ckpt.pt"
    trainer.save_checkpoint(ckpt_file, dataset.label_encoder, extra_config={"input_dim": 411, "hidden_dim": 32})
    assert ckpt_file.exists()

    # Test LiveInferenceEngine
    engine = LiveInferenceEngine(ckpt_file, window_size=20, min_prediction_confidence=0.1)
    # Push 20 frames with hand confidence
    pred_label, pred_conf = None, 0.0
    for _ in range(20):
        frame = np.random.randn(137, 3).astype(np.float32)
        frame[95:, 2] = 0.8  # Hand confidence high
        lbl, conf = engine.push_frame(frame)
        if lbl is not None:
            pred_label, pred_conf = lbl, conf

    assert pred_label in classes or pred_label is None

