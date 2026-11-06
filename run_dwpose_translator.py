#!/usr/bin/env python3
"""
LSEC SDK - Real-Time Sign Translator & Tutor powered by DWPose (GPU Acceleration).
Runs whole-body and high-fidelity hand tracking via NVIDIA RTX 3060 (CUDA).
"""

from collections import deque
from pathlib import Path
import sys
import time

# Ensure src in path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np

from lsec_sdk.dwpose_extractor import DWPoseDetector
from lsec_sdk.models.inference import LiveInferenceEngine
from lsec_sdk.tutor_engine import LSECTutorEngine, TutorState, clean_ascii


def main():
    print("=" * 65)
    print(" LSEC SDK - TRADUCTOR Y TUTOR EN VIVO CON DWPOSE (NVIDIA GPU)")
    print("=" * 65)
    print("Iniciando DWPose Whole-Body Pose Detector en GPU (CUDA)...")

    try:
        detector = DWPoseDetector()
        print(f">> DWPose activo con: {detector.provider_name}")
    except Exception as e:
        print(f"ERROR al inicializar DWPose en GPU: {e}")
        return

    # Load LSEC Classifier
    checkpoint_path = Path("models/checkpoints/lsec_classifier.pt")
    if not checkpoint_path.exists():
        print(f"ERROR: No se encontró el modelo en: {checkpoint_path}")
        return

    engine = LiveInferenceEngine(
        checkpoint_path=checkpoint_path,
        window_size=32,
        min_prediction_confidence=0.60,
    )
    print(f"Vocabulario activo ({len(engine.label_encoder.label_to_idx)} señas): {list(engine.label_encoder.label_to_idx.keys())}")

    # Initialize Tutor Engine
    tutor = LSECTutorEngine(enabled=True)
    if tutor.lessons:
        tutor.set_section("TODAS")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: No se pudo abrir la cámara web.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "LSEC SDK - Traductor GPU (DWPose WholeBody)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1100, 620)

    current_prediction = ""
    current_confidence = 0.0
    last_detection_time = 0.0

    fps_time = time.time()
    fps_counter = 0
    current_fps = 0.0

    print("\n>> Sistema DWPose activo en tiempo real.")
    print("   [M]: Abrir / Cerrar Menú de Señas en Pantalla")
    print("   [T]: Habilitar / Deshabilitar Modo Tutor")
    print("   [C]: Cambiar Sección Temática")
    print("   [N]: Siguiente seña | [P]: Seña anterior")
    print("   [ESPACIO]: Saltar demostración de palitos")
    print("   [Q]: Salir\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            # 1. Run GPU DWPose inference
            dw_res = detector.process(frame)
            if dw_res is not None:
                kpts, scores = dw_res
                # Draw skeleton
                display_frame = detector.draw_skeleton(frame.copy(), kpts, scores)
                # Check active hands
                lh_score = np.mean(scores[92:113]) if len(scores) >= 113 else 0.0
                rh_score = np.mean(scores[113:134]) if len(scores) >= 134 else 0.0
                hands_detected = bool(lh_score > 0.20 or rh_score > 0.20)
            else:
                display_frame = frame.copy()
                hands_detected = False

            # Extract normalized (137, 3) tensor
            kp_array = detector.extract_lsec_tensor(frame)

            # 2. Push to temporal inference engine
            pred_label, pred_conf = engine.push_frame(kp_array)

            if pred_label is not None and pred_label != "REPOSO":
                if pred_label != current_prediction:
                    print(f">> [DWPose - NUEVA SEÑA]: {pred_label} ({pred_conf * 100:.1f}%)")
                current_prediction = pred_label
                current_confidence = pred_conf
                last_detection_time = time.time()

            # Target confidence for tutor mode
            target_conf = 0.0
            if tutor.enabled and tutor.current_lesson:
                target_conf = engine.get_target_confidence(tutor.current_lesson.clean_label)

            # FPS calculation
            fps_counter += 1
            now = time.time()
            if now - fps_time >= 1.0:
                current_fps = fps_counter / (now - fps_time)
                fps_counter = 0
                fps_time = now

            h, w, _ = display_frame.shape

            # 3. Render Tutor or HUD
            if tutor.enabled:
                display_frame = tutor.update_and_render(
                    display_frame,
                    pred_label=current_prediction if (time.time() - last_detection_time < 2.5) else None,
                    pred_conf=current_confidence,
                    hands_detected=hands_detected,
                    target_conf=target_conf,
                )
                # Small GPU badge on top left
                cv2.putText(display_frame, f"GPU: DWPose (CUDA) | FPS: {current_fps:4.1f}", (w - 380, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 1)
            else:
                # Top bar HUD for free translator
                cv2.rectangle(display_frame, (0, 0), (w, 55), (15, 15, 15), -1)
                cv2.putText(display_frame, f"FPS: {current_fps:4.1f} (DWPose CUDA)", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 2)
                cv2.putText(display_frame, "[T]: Modo Tutor | [M]: Menu | [Q]: Salir", (w - 380, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)

                is_active = (time.time() - last_detection_time) < 2.0
                banner_h = 90
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (0, h - banner_h), (w, h), (15, 25, 15) if is_active else (20, 20, 20), -1)
                display_frame = cv2.addWeighted(overlay, 0.85, display_frame, 0.15, 0)

                if is_active and current_prediction:
                    cv2.putText(
                        display_frame,
                        f"TRADUCCION: {clean_ascii(current_prediction)}",
                        (30, h - 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 120),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        display_frame,
                        f"Confianza: {current_confidence * 100:.1f}%",
                        (w - 240, h - 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (200, 255, 200),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    cv2.putText(
                        display_frame,
                        "DWPose listo. Haz una seña frente a la cámara...",
                        (30, h - 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (180, 180, 180),
                        2,
                        cv2.LINE_AA,
                    )

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("t") or key == ord("T"):
                tutor.enabled = not tutor.enabled
            elif key == ord("m") or key == ord("M"):
                tutor.toggle_menu()
            elif key == ord("c") or key == ord("C"):
                tutor.cycle_section()
            elif key == ord("n") or key == ord("N"):
                tutor.next_lesson()
            elif key == ord("p") or key == ord("P"):
                tutor.prev_lesson()
            elif key == ord("r") or key == ord("R"):
                tutor.reset_to_demo()
            elif key == ord("s") or key == ord("S"):
                tutor.slow_motion = not tutor.slow_motion
            elif key == 32:  # Barra espaciadora: saltar demo
                if tutor.enabled:
                    tutor.skip_demo()

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n>> Sesión DWPose finalizada.")


if __name__ == "__main__":
    main()
