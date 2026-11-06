#!/usr/bin/env python3
"""Script to extract keypoint features from LSEC video '023 CÓMO TE SIENTES.mp4'."""

import glob
from pathlib import Path
import shutil
import sys

# Ensure src in path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
import pandas as pd

from lsec_sdk import FeatureProvider, Visualizer


def main():
    # Find the video file
    video_matches = glob.glob("*023*.*")
    if not video_matches:
        print("ERROR: No se encontró el video del archivo 023.")
        return

    video_file = Path(video_matches[0])
    sequence_id = "lsec_023_como_te_sientes"
    label = "CÓMO TE SIENTES"

    print("=" * 60)
    print(f" PROCESANDO VIDEO LSEC: {video_file.name}")
    print(f" Frase / Glosa: '{label}'")
    print("=" * 60)

    # 1. Initialize FeatureProvider
    provider = FeatureProvider(confidence_threshold=0.1, normalization="resolution")

    # 2. Extract features from video
    result = provider.process_video(video_file, sequence_id=sequence_id, label=label)

    print(f"-> Fotogramas procesados: {result.num_frames}")
    print(f"-> Tensor extraído: {result.tensor.shape}, dtype={result.tensor.dtype}")
    print(f"-> FPS del video: {result.fps}")
    print(f"-> Resolución: {result.width}x{result.height}")
    print("-> Métricas de confianza:")
    for k, v in result.metrics.items():
        print(f"   * {k}: {v}")

    # 3. Save .npy tensor
    out_dir = Path("data/processed/sequences")
    npy_path = provider.save_sequence(result, out_dir)
    print(f"\n-> Tensor guardado en: {npy_path.resolve()} ({npy_path.stat().st_size} bytes)")

    # 4. Save to metadata CSV
    csv_file = Path("data/processed/metadata.csv")
    provider.save_metadata_csv([result], csv_file, append=True)
    print(f"-> Metadatos guardados en: {csv_file.resolve()}")

    # 5. Generate Visual Side-by-Side GIF and Key Snapshots
    vis_dir = Path("data/processed/visualizations")
    vis_dir.mkdir(parents=True, exist_ok=True)
    visualizer = Visualizer(confidence_threshold=0.1)

    gif_path = vis_dir / f"{sequence_id}_validation.gif"
    print(f"\n-> Generando animación GIF comparativa...")
    visualizer.export_gif(npy_path, video_file, gif_path, fps=12, scale=0.45)
    print(f"   * GIF guardado: {gif_path} ({gif_path.stat().st_size} bytes)")

    # Generate snapshots at 25%, 50%, 75% of video
    cap = cv2.VideoCapture(str(video_file))
    snapshots = []
    key_frames = [
        int(result.num_frames * 0.25),
        int(result.num_frames * 0.50),
        int(result.num_frames * 0.75),
    ]

    for idx in range(result.num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if idx in key_frames:
            sbs = visualizer.create_side_by_side(frame, result.tensor[idx], idx, result.num_frames, scale=0.6)
            snap_path = vis_dir / f"{sequence_id}_snap_{idx:03d}.jpg"
            cv2.imwrite(str(snap_path), sbs)
            snapshots.append(snap_path)
            print(f"   * Instantánea guardada: {snap_path.name}")
    cap.release()

    # 6. Copy to Artifacts directory for user display
    artifact_dir = Path(r"C:\Users\jesuc\.gemini\antigravity\brain\212d5e7e-eafe-4f2a-a7f1-818fbc51e65b")
    if artifact_dir.exists():
        shutil.copy2(gif_path, artifact_dir / gif_path.name)
        for snap in snapshots:
            shutil.copy2(snap, artifact_dir / snap.name)
        print("   * Artefactos visuales copiados exitosamente al reporte.")

    print("\n" + "=" * 60)
    print(" ¡EXTRACCIÓN DE LSEC COMPLETADA CON ÉXITO!")
    print("=" * 60)


if __name__ == "__main__":
    main()

