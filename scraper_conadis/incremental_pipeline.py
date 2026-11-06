#!/usr/bin/env python3
"""
Incremental 1-to-1 Pipeline for LSEC:
1. Scrapes term with Scrapling
2. Downloads video with yt-dlp (mobile player client)
3. Compresses video with FFmpeg (CRF 26, optimal quality-to-size ratio)
4. Extracts 137 keypoints tensor (T, 137, 3) with MediaPipe Holistic
5. Registers metadata in CSV
6. Retrains/fine-tunes BiLSTM + Attention model 1-by-1 on GPU RTX 3060!
"""

import argparse
from pathlib import Path
import sys

# Ensure paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from lsec_sdk import FeatureProvider
from lsec_sdk.models import (
    SignLSTMClassifier,
    SignModelTrainer,
    SignSequenceDataset,
)
from scrapling_crawler import ScraplingCrawler
from video_compressor import VideoCompressor
from video_downloader import ConadisVideoDownloader


def train_step_one_by_one(csv_path: Path, model_out_path: Path, epochs: int = 25):
    """Retrains the model with the updated dataset including the newly extracted sign."""
    df = pd.read_csv(csv_path)
    labeled_df = df[df["label"].notna() & (df["label"].str.strip() != "")]

    samples = [(row["npy_path"], row["label"]) for _, row in labeled_df.iterrows()]
    labels = [lbl for _, lbl in samples]
    unique_labels = sorted(list(set(labels)))

    print(f"\n[Entrenamiento 1-a-1] Re-entrenando modelo con {len(unique_labels)} clases...")
    print(f"  Vocabulario actual: {unique_labels}")

    dataset = SignSequenceDataset(
        samples,
        target_frames=32,
        augment=True,
        replicate_factor=20,
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = SignLSTMClassifier(
        input_dim=411,
        hidden_dim=128,
        num_layers=2,
        num_classes=dataset.label_encoder.num_classes,
        dropout=0.3,
        use_attention=True,
    )

    trainer = SignModelTrainer(model, learning_rate=1e-3)

    for ep in range(1, epochs + 1):
        loss, acc = trainer.train_epoch(loader)
        if ep % 10 == 0 or ep == epochs:
            print(f"    -> Época [{ep:02d}/{epochs:02d}] - Pérdida: {loss:.4f} | Precisión: {acc * 100:.1f}%")

    extra_config = {
        "input_dim": 411,
        "hidden_dim": 128,
        "num_layers": 2,
        "use_attention": True,
        "target_frames": 32,
    }
    trainer.save_checkpoint(model_out_path, dataset.label_encoder, extra_config)
    print(f"  [OK] Checkpoint actualizado: {model_out_path.resolve()}\n")


def run_pipeline(category: str = "frases-comunes", count: int = 3, compress: bool = True):
    print("=" * 65)
    print(" LSEC SDK - PIPELINE INCREMENTAL CON SCRAPLING & ENTRENAMIENTO 1 A 1")
    print("=" * 65)
    print(f"Categoría:            {category}")
    print(f"Cantidad a procesar:  {count}")
    print(f"Compresión FFmpeg:    {compress} (CRF 26 - Máxima eficiencia)")
    print("=" * 65)

    # 1. Scrapling Crawler
    crawler = ScraplingCrawler(delay_seconds=0.6)
    entries = crawler.crawl_category(category, max_articles=count)

    if not entries:
        print("No se encontraron artículos para procesar.")
        return

    downloader = ConadisVideoDownloader()
    compressor = VideoCompressor(crf=26, max_height=720, fps=25)
    provider = FeatureProvider(confidence_threshold=0.1, normalization="resolution")

    csv_file = Path("data/processed/metadata.csv")
    out_seq_dir = Path("data/processed/sequences")
    model_ckpt = Path("models/checkpoints/lsec_classifier.pt")

    for idx, entry in enumerate(entries, 1):
        print("-" * 65)
        print(f"[{idx}/{len(entries)}] PROCESANDO SEÑA: '{entry.title.upper()}'")
        print("-" * 65)

        # Step A: Download
        raw_video = downloader.download_entry(entry)
        if not raw_video or not raw_video.exists():
            print(f"  -> Se omite '{entry.title}' por error de descarga.")
            continue

        # Step B: Video Compression (Calidad-Precio óptimo)
        video_to_process = raw_video
        if compress:
            opt_video, stats = compressor.compress(raw_video, replace_original=True)
            video_to_process = opt_video

        # Step C: Feature Extraction with MediaPipe Holistic
        seq_id = f"conadis_{entry.category}_{entry.slug}"
        label = entry.title.upper()

        print(f"  -> Extrayendo características esqueléticas (137 keypoints)...")
        res = provider.process_video(video_to_process, sequence_id=seq_id, label=label)
        npy_path = provider.save_sequence(res, out_seq_dir)
        provider.save_metadata_csv([res], csv_file, append=True)

        print(f"  -> Tensor generado: {res.tensor.shape} ({res.num_frames} frames)")
        print(f"  -> Metadatos guardados en: {csv_file.name}")

        # Step D: Train Step 1-a-1
        train_step_one_by_one(csv_file, model_ckpt, epochs=20)


def main():
    parser = argparse.ArgumentParser(description="Incremental 1-to-1 LSEC Pipeline")
    parser.add_argument("--category", type=str, default="frases-comunes")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--no-compress", action="store_true")
    args = parser.parse_args()

    run_pipeline(category=args.category, count=args.count, compress=not args.no_compress)


if __name__ == "__main__":
    main()

