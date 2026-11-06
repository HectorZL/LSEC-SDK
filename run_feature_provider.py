#!/usr/bin/env python3
"""
CLI runner script to execute FeatureProvider on OpenPose sequences.
Processes sequence, filters low confidence keypoints, normalizes,
exports NumPy (.npy) tensor and metadata CSV.
"""

import argparse
import logging
from pathlib import Path
import sys

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd

from lsec_sdk.constants import (
    DEFAULT_FPS,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    FACE_SLICE,
    HAND_LEFT_SLICE,
    HAND_RIGHT_SLICE,
    POSE_SLICE,
)
from lsec_sdk.feature_provider import FeatureProvider


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("FeatureProviderRunner")


def main():
    parser = argparse.ArgumentParser(description="LSEC FeatureProvider - OpenPose Keypoints Pipeline")
    parser.add_argument(
        "--sequence-dir",
        type=str,
        default="openpose_output/json/-fZc293MpJk_0-1-rgb_front",
        help="Path to folder containing sequence JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/sequences",
        help="Directory to save generated .npy files",
    )
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default="data/processed/metadata.csv",
        help="Path to metadata CSV file",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.1,
        help="Confidence threshold for keypoints filtering",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        choices=["resolution", "center", "none"],
        default="resolution",
        help="Normalization method",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_FRAME_WIDTH, help="Frame width in pixels")
    parser.add_argument("--height", type=int, default=DEFAULT_FRAME_HEIGHT, help="Frame height in pixels")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Frame rate")

    args = parser.parse_args()
    logger = setup_logger()

    seq_path = Path(args.sequence_dir)
    out_dir = Path(args.output_dir)
    csv_path = Path(args.metadata_csv)

    logger.info("=" * 60)
    logger.info(" INICIANDO PIPELINE: FEATURE PROVIDER")
    logger.info("=" * 60)
    logger.info(f"Directorio de entrada: {seq_path.resolve()}")
    logger.info(f"Directorio de salida:  {out_dir.resolve()}")
    logger.info(f"Archivo de metadatos:  {csv_path.resolve()}")
    logger.info(f"Parámetros: Umbral Conf={args.conf_thresh}, Normalización='{args.normalization}', Resolución={args.width}x{args.height}")

    # Step 1 & 2: Initialize Provider
    provider = FeatureProvider(
        confidence_threshold=args.conf_thresh,
        normalization=args.normalization,
        resolution=(args.width, args.height),
        fps=args.fps,
    )

    # Step 3, 4: Process sequence (order, extract body/face/hands, filter confidence, normalize)
    logger.info(f"Procesando secuencia '{seq_path.name}'...")
    result = provider.process_sequence(seq_path)

    logger.info(f"-> Fotogramas ordenados y procesados: {result.num_frames}")
    logger.info(f"-> Tensor generado: forma {result.tensor.shape}, dtype={result.tensor.dtype}")
    logger.info(f"   * Cuerpo (Pose):      {result.tensor[:, POSE_SLICE, :].shape} (25 puntos)")
    logger.info(f"   * Rostro (Face):      {result.tensor[:, FACE_SLICE, :].shape} (70 puntos)")
    logger.info(f"   * Mano Izquierda:     {result.tensor[:, HAND_LEFT_SLICE, :].shape} (21 puntos)")
    logger.info(f"   * Mano Derecha:       {result.tensor[:, HAND_RIGHT_SLICE, :].shape} (21 puntos)")

    # Display Metrics
    logger.info("-> Métricas de confianza:")
    for k, v in result.metrics.items():
        logger.info(f"   * {k}: {v}")

    # Step 5: Save .npy
    npy_path = provider.save_sequence(result, out_dir)
    logger.info(f"-> Secuencia guardada en .npy: {npy_path.resolve()} ({npy_path.stat().st_size} bytes)")

    # Step 6: Save metadata CSV
    csv_file = provider.save_metadata_csv([result], csv_path, append=True)
    logger.info(f"-> Metadatos registrados en CSV: {csv_file.resolve()}")

    logger.info("=" * 60)
    logger.info(" PROCESAMIENTO COMPLETADO EXITOSAMENTE")
    logger.info("=" * 60)

    # Summary preview
    df = pd.read_csv(csv_file)
    print("\n--- REGISTRO EN CSV (METADATOS) ---")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

