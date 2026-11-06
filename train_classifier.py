#!/usr/bin/env python3
"""
Training pipeline for LSEC Sign Language Classification using PyTorch & CUDA.
Loads sequences from metadata.csv, applies data augmentation, trains BiLSTM + Attention,
and saves the checkpoint for live inference.
"""

import argparse
from pathlib import Path
import sys

# Ensure src in path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from lsec_sdk.models import (
    LabelEncoder,
    SignLSTMClassifier,
    SignModelTrainer,
    SignSequenceDataset,
)


def main():
    parser = argparse.ArgumentParser(description="Train LSEC Sign Classifier")
    parser.add_argument("--csv", type=str, default="data/processed/metadata.csv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--target-frames", type=int, default=32)
    parser.add_argument("--out-model", type=str, default="models/checkpoints/lsec_classifier.pt")
    args = parser.parse_args()

    csv_file = Path(args.csv)
    if not csv_file.exists():
        print(f"ERROR: No se encontró el archivo de metadatos: {csv_file}")
        return

    df = pd.read_csv(csv_file)
    labeled_df = df[df["label"].notna() & (df["label"].str.strip() != "")]

    if len(labeled_df) == 0:
        print("ERROR: No hay secuencias con etiquetas ('label') en el CSV.")
        print("Graba señas primero usando: python record_dataset.py [NOMBRE_SEÑA]")
        return

    print("=" * 60)
    print(" ENTRENAMIENTO DE MODELO LSEC (BiLSTM + Attention)")
    print("=" * 60)
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Dispositivo de cómputo: {device_name}")
    print(f"Muestras etiquetadas encontradas: {len(labeled_df)}")

    samples = [(row["npy_path"], row["label"]) for _, row in labeled_df.iterrows()]
    labels = [lbl for _, lbl in samples]
    unique_labels = sorted(list(set(labels)))
    print(f"Clases a entrenar ({len(unique_labels)}): {unique_labels}")

    # Build dataset with augmentation and replication
    dataset = SignSequenceDataset(
        samples,
        target_frames=args.target_frames,
        augment=True,
        replicate_factor=20,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model
    model = SignLSTMClassifier(
        input_dim=411,
        hidden_dim=128,
        num_layers=2,
        num_classes=dataset.label_encoder.num_classes,
        dropout=0.3,
        use_attention=True,
    )

    trainer = SignModelTrainer(model, learning_rate=args.lr)

    print("\nIniciando optimización...")
    for epoch in range(1, args.epochs + 1):
        loss, acc = trainer.train_epoch(loader)
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"  * Época [{epoch:02d}/{args.epochs:02d}] -> Pérdida: {loss:.4f} | Precisión: {acc * 100:.1f}%")

    out_path = Path(args.out_model)
    extra_config = {
        "input_dim": 411,
        "hidden_dim": 128,
        "num_layers": 2,
        "use_attention": True,
        "target_frames": args.target_frames,
    }
    trainer.save_checkpoint(out_path, dataset.label_encoder, extra_config)

    print("\n" + "=" * 60)
    print(f" ¡MODELO ENTRENADO Y GUARDADO!")
    print(f" Archivo de pesos: {out_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()

