"""Trainer module for sign language neural models."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import LabelEncoder, SignSequenceDataset


class SignModelTrainer:
    """Handles training loop, evaluation, checkpointing, and metrics calculation."""

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Trains the model for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            logits, _ = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total_samples += x.size(0)

        epoch_loss = total_loss / max(1, total_samples)
        epoch_acc = correct / max(1, total_samples)
        return epoch_loss, epoch_acc

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Evaluates model on validation or test dataset."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                logits, _ = self.model(x)
                loss = self.criterion(logits, y)

                total_loss += loss.item() * x.size(0)
                preds = logits.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total_samples += x.size(0)

        val_loss = total_loss / max(1, total_samples)
        val_acc = correct / max(1, total_samples)
        return val_loss, val_acc

    def save_checkpoint(
        self,
        save_path: Union[str, Path],
        label_encoder: LabelEncoder,
        extra_config: Optional[dict] = None,
    ) -> Path:
        """Saves model weights, label mapping, and hyperparameters."""
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "model_class": self.model.__class__.__name__,
            "label_to_idx": label_encoder.label_to_idx,
            "idx_to_label": label_encoder.idx_to_label,
            "config": extra_config or {},
        }
        torch.save(checkpoint, save_file)
        return save_file

