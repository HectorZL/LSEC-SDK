#!/usr/bin/env python3
"""
Real-time webcam sign language tracking with MediaPipe and OpenCV.
Tracks body, face, and hands at ~30 FPS on your laptop camera.

Controls:
- [ESPACIO] o [R]: Iniciar / Detener grabación de secuencia
- [Q] o [ESC]:    Cerrar la cámara
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


def main():
    print("=" * 60)
    print(" INICIANDO SEGUIMIENTO EN TIEMPO REAL CON LA CÁMARA")
    print("=" * 60)
    print("Controles:")
    print("  * [ESPACIO] o [R]: Iniciar / Pausar grabación de secuencia")
    print("  * [Q] o [ESC]:     Salir de la aplicación")
    print("=" * 60)

    capture = RealTimeCapture(camera_index=0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: No se pudo abrir la cámara web de tu laptop.")
        print("Verifica que no esté en uso por otra aplicación (Zoom, Teams, etc.).")
        return

    # Set camera resolution (standard 640x480 or 1280x720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "LSEC SDK - Captura en Vivo de Senas (MediaPipe + OpenCV)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 576)

    is_recording = False
    recorded_frames = []
    fps_time = time.time()
    fps_counter = 0
    current_fps = 0.0

    output_dir = Path("data/processed/sequences")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_file = Path("data/processed/metadata.csv")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error al leer fotograma de la cámara.")
                break

            # Mirror image for natural user interaction
            frame = cv2.flip(frame, 1)

            # Process with MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = capture.holistic.process(rgb)

            # Extract structured keypoint array
            kp_array = capture.extract_landmarks_array(results)

            if is_recording:
                recorded_frames.append(kp_array)

            # Draw skeleton overlays
            display_frame = capture.draw_skeleton(frame, results)

            # Calculate FPS
            fps_counter += 1
            now = time.time()
            if now - fps_time >= 1.0:
                current_fps = fps_counter / (now - fps_time)
                fps_counter = 0
                fps_time = now

            # Render HUD
            display_frame = capture.render_hud(
                display_frame,
                current_fps,
                is_recording,
                len(recorded_frames),
                results,
            )

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            # 'Q' or ESC to quit
            if key in (ord('q'), ord('Q'), 27):
                break

            # Spacebar or 'R' to toggle recording
            elif key in (32, ord('r'), ord('R')):
                if not is_recording:
                    is_recording = True
                    recorded_frames = []
                    print("\n>> [REC] Grabación INICIADA... ¡Haz una seña con tus manos!")
                else:
                    is_recording = False
                    n_frames = len(recorded_frames)
                    print(f"\n>> [STOP] Grabación DETENIDA ({n_frames} fotogramas capturados).")

                    if n_frames >= 5:
                        seq_tensor = np.stack(recorded_frames, axis=0)  # (T, 137, 3)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        seq_id = f"webcam_seq_{timestamp}"
                        npy_file = output_dir / f"{seq_id}.npy"
                        np.save(npy_file, seq_tensor)

                        print(f"   -> Secuencia guardada: {npy_file.resolve()}")
                        print(f"   -> Forma del tensor: {seq_tensor.shape}")

                        # Append to CSV
                        row = {
                            "sequence_id": seq_id,
                            "num_frames": n_frames,
                            "fps": round(current_fps if current_fps > 0 else 30.0, 1),
                            "resolution_w": frame.shape[1],
                            "resolution_h": frame.shape[0],
                            "total_keypoints": TOTAL_KEYPOINTS_COUNT,
                            "tensor_shape": f"{seq_tensor.shape}",
                            "normalization": "resolution",
                            "confidence_threshold": 0.5,
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

                        print(f"   -> Metadatos registrados en: {csv_file.resolve()}")
                    else:
                        print("   -> Secuencia descartada por ser demasiado corta (< 5 fotogramas).")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        capture.close()
        print("\nCámara cerrada y recursos liberados correctamente.")


if __name__ == "__main__":
    main()

