#!/usr/bin/env python3
"""
Extracts LSEC sign keypoints from videos using DWPose (Whole-Body GPU)
and creates the fresh dataset for retraining from scratch.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
import pandas as pd
from lsec_sdk.dwpose_extractor import DWPoseDetector

def extract_video(detector: DWPoseDetector, video_path: Path, label: str, sequence_id: str):
    print(f"\n>> Procesando video con DWPose GPU: {video_path.name}")
    print(f"   Etiqueta: '{label}' | ID: {sequence_id}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_tensor = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract (137, 3) tensor from DWPose
        t = detector.extract_lsec_tensor(frame)
        frames_tensor.append(t)
        frame_idx += 1
        if frame_idx % 25 == 0 or frame_idx == total_frames:
            print(f"   [Fotograma {frame_idx:03d}/{total_frames:03d}] ...")

    cap.release()

    tensor_arr = np.array(frames_tensor, dtype=np.float32)
    print(f"   Tensor completado: {tensor_arr.shape} ({tensor_arr.dtype})")

    # Metrics
    lh_conf = np.mean(tensor_arr[:, 95:116, 2])
    rh_conf = np.mean(tensor_arr[:, 116:137, 2])
    pose_conf = np.mean(tensor_arr[:, :25, 2])
    face_conf = np.mean(tensor_arr[:, 25:95, 2])
    print(f"   Confianzas medias -> Pose: {pose_conf:.3f} | Rostro: {face_conf:.3f} | Mano Izq: {lh_conf:.3f} | Mano Der: {rh_conf:.3f}")

    out_dir = Path("data/processed/sequences")
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / f"{sequence_id}.npy"
    np.save(npy_path, tensor_arr)
    print(f"   Guardado en: {npy_path.resolve()}")

    return {
        "sequence_id": sequence_id,
        "label": label,
        "num_frames": len(tensor_arr),
        "fps": fps,
        "width": width,
        "height": height,
        "npy_path": str(npy_path.resolve()),
        "pose_mean_conf": float(pose_conf),
        "face_mean_conf": float(face_conf),
        "lh_mean_conf": float(lh_conf),
        "rh_mean_conf": float(rh_conf),
    }

def main():
    print("=" * 65)
    print(" EXTRACCIÓN DE CARACTERÍSTICAS DWPose (WHOLE-BODY GPU)")
    print(" Preparación de dataset limpio para entrenamiento desde cero")
    print("=" * 65)

    detector = DWPoseDetector()

    videos_to_process = [
        (Path("023 CÓMO TE  SIENTES.mp4"), "CÓMO TE SIENTES", "dwpose_023_como_te_sientes"),
        (Path("044 DÓNDE ESTUDIÓ.mp4"), "DÓNDE ESTUDIÓ", "dwpose_044_donde_estudio"),
    ]

    records = []
    for v_path, label, seq_id in videos_to_process:
        if not v_path.exists():
            # Check with alternative spaces
            matches = list(Path(".").glob(f"*{seq_id.split('_')[1]}*.mp4"))
            if matches:
                v_path = matches[0]
            else:
                print(f"ERROR: No se encontró el archivo de video: {v_path}")
                continue
        rec = extract_video(detector, v_path, label, seq_id)
        records.append(rec)

    # 3. Create clean REPOSO sequence (idle posture baseline)
    # Extract resting posture from the first 25 frames of the videos where the signer is at rest
    como_tensor = np.load(records[0]["npy_path"])
    reposo_tensor = como_tensor[:28].copy()
    # Ensure hand confidence is very low/rest in reposo baseline
    reposo_tensor[:, 95:, 2] *= 0.1
    reposo_path = Path("data/processed/sequences/dwpose_reposo_baseline.npy")
    np.save(reposo_path, reposo_tensor)

    records.append({
        "sequence_id": "dwpose_reposo_baseline",
        "label": "REPOSO",
        "num_frames": len(reposo_tensor),
        "fps": 30.0,
        "width": 1280,
        "height": 720,
        "npy_path": str(reposo_path.resolve()),
        "pose_mean_conf": float(np.mean(reposo_tensor[:, :25, 2])),
        "face_mean_conf": float(np.mean(reposo_tensor[:, 25:95, 2])),
        "lh_mean_conf": float(np.mean(reposo_tensor[:, 95:116, 2])),
        "rh_mean_conf": float(np.mean(reposo_tensor[:, 116:137, 2])),
    })
    print(f"\n>> Creada secuencia neutral de 'REPOSO': {reposo_path.resolve()}")

    # Save fresh metadata.csv containing only the new clean dataset
    df = pd.DataFrame(records)
    csv_path = Path("data/processed/metadata.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n>> Nuevo metadata.csv generado con {len(df)} registros:")
    for _, r in df.iterrows():
        print(f"   - [{r['label']}] {r['sequence_id']} ({r['num_frames']} frames)")

    print("\n" + "=" * 65)
    print(" ¡EXTRACCIÓN CON DWPose GPU COMPLETADA CON ÉXITO!")
    print("=" * 65)

if __name__ == "__main__":
    main()

