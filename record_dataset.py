#!/usr/bin/env python3
"""
Interactive Dataset Recorder for LSEC signs.
Guides user through recording multiple repetitions of selected signs.
Saves .npy sequences and automatically updates metadata.csv.
"""

from datetime import datetime
from pathlib import Path
import sys
import time

# Ensure src in path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
import pandas as pd

from lsec_sdk.constants import TOTAL_KEYPOINTS_COUNT
from lsec_sdk.realtime_capture import RealTimeCapture


def record_sign_interactive(sign_name: str, repetitions: int = 5, record_seconds: float = 2.5):
    print("=" * 60)
    print(f" GRABADOR DE LSEC: Seña '{sign_name.upper()}'")
    print(f" Vamos a grabar {repetitions} repeticiones.")
    print("=" * 60)

    capture = RealTimeCapture(camera_index=0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: No se pudo acceder a la cámara web.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = f"LSEC Recorder - Grabando '{sign_name.upper()}'"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    output_dir = Path("data/processed/sequences")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_file = Path("data/processed/metadata.csv")

    for rep in range(1, repetitions + 1):
        print(f"\n>> Preparando Repetición {rep}/{repetitions}...")

        # Countdown phase (3 seconds)
        countdown_start = time.time()
        while time.time() - countdown_start < 3.0:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            remaining = 3.0 - (time.time() - countdown_start)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (20, 20, 20), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

            msg = f"Preparate: '{sign_name.upper()}' (Rep {rep}/{repetitions}) en {remaining:.1f}s"
            cv2.putText(frame, msg, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                print("Grabación cancelada por el usuario.")
                cap.release()
                cv2.destroyAllWindows()
                capture.close()
                return

        # Recording phase
        print(f"   * ¡GRABANDO! Realiza la seña ahora...")
        rec_start = time.time()
        recorded_frames = []

        while time.time() - rec_start < record_seconds:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = capture.holistic.process(rgb)
            kp_array = capture.extract_landmarks_array(results)
            recorded_frames.append(kp_array)

            display_frame = capture.draw_skeleton(frame, results)

            # Red REC border & text
            cv2.rectangle(display_frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 6)
            cv2.putText(display_frame, f"REC: '{sign_name.upper()}'", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imshow(window_name, display_frame)
            cv2.waitKey(1)

        # Save repetition
        if len(recorded_frames) >= 10:
            seq_tensor = np.stack(recorded_frames, axis=0)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            clean_name = sign_name.lower().replace(" ", "_")
            seq_id = f"lsec_{clean_name}_{timestamp}"
            npy_file = output_dir / f"{seq_id}.npy"
            np.save(npy_file, seq_tensor)

            row = {
                "sequence_id": seq_id,
                "label": sign_name.upper(),
                "num_frames": len(recorded_frames),
                "fps": 30.0,
                "resolution_w": 1280,
                "resolution_h": 720,
                "total_keypoints": TOTAL_KEYPOINTS_COUNT,
                "tensor_shape": f"{seq_tensor.shape}",
                "normalization": "resolution",
                "confidence_threshold": 0.1,
                "npy_path": str(npy_file.resolve()),
                "mean_conf_pose": round(float(np.mean(seq_tensor[:, :25, 2])), 4),
                "mean_conf_face": round(float(np.mean(seq_tensor[:, 25:95, 2])), 4),
                "mean_conf_hand_left": round(float(np.mean(seq_tensor[:, 95:116, 2])), 4),
                "mean_conf_hand_right": round(float(np.mean(seq_tensor[:, 116:137, 2])), 4),
                "mean_conf_overall": round(float(np.mean(seq_tensor[:, :, 2])), 4),
                "low_conf_ratio": 0.0,
            }
            df_new = pd.DataFrame([row])
            if csv_file.exists():
                df_old = pd.read_csv(csv_file)
                pd.concat([df_old, df_new], ignore_index=True).to_csv(csv_file, index=False)
            else:
                df_new.to_csv(csv_file, index=False)

            print(f"   -> Repetición {rep} guardada con éxito ({len(recorded_frames)} frames).")

    cap.release()
    cv2.destroyAllWindows()
    capture.close()
    print("\n" + "=" * 60)
    print(f" ¡Listo! Se grabaron {repetitions} muestras de '{sign_name.upper()}'.")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_sign = " ".join(sys.argv[1:])
    else:
        target_sign = input("Nombre de la seña a grabar (ej: HOLA / GRACIAS / AYUDA): ").strip()

    if target_sign:
        record_sign_interactive(target_sign, repetitions=5, record_seconds=2.5)

