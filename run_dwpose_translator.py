#!/usr/bin/env python3
"""
LSEC SDK - Real-Time Sign Translator & Tutor powered by DWPose (GPU Acceleration).
Runs whole-body and high-fidelity hand tracking via NVIDIA RTX 3060 (CUDA).
"""

import argparse
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
    parser = argparse.ArgumentParser(description="LSEC SDK - Traductor y Tutor en Vivo con DWPose (NVIDIA GPU)")
    parser.add_argument("--libre", "--free", action="store_true", help="Iniciar directamente en Modo Libre (traduccion continua sin tutor)")
    args = parser.parse_args()

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
    tutor = LSECTutorEngine(enabled=not args.libre)
    if tutor.lessons:
        tutor.set_section("TODAS")

    # En Windows, DirectShow (CAP_DSHOW) previene bloqueos de cámara por navegadores
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
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
    sign_history: deque[str] = deque(maxlen=4)

    fps_time = time.time()
    fps_counter = 0
    current_fps = 0.0

    print("\n>> Sistema DWPose activo en tiempo real.")
    print("   [T]: Alternar Modo Libre / Modo Tutor (Prueba señas libres una tras otra)")
    print("   [M]: Abrir / Cerrar Menú de Señas en Pantalla")
    print("   [C]: Cambiar Sección Temática")
    print("   [N]: Siguiente seña | [P]: Seña anterior")
    print("   [ESPACIO]: Saltar demostración de palitos")
    print("   [Q]: Salir")
    print(f">> Modo inicial: {'MODO LIBRE (Traductor Continuo)' if not tutor.enabled else 'MODO TUTOR (Guiado)'}\n")

    try:
        retry_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                retry_count += 1
                if retry_count < 15:
                    time.sleep(0.1)
                    continue
                print("\n[AVISO] No se reciben fotogramas de la cámara web.")
                print(">> Asegúrate de CERRAR la pestaña del navegador o cualquier otra app que esté usando la cámara.\n")
                break
            retry_count = 0
            frame = cv2.flip(frame, 1)

            # 1. Run GPU DWPose inference
            dw_res = detector.process(frame)
            if dw_res is not None:
                kpts, scores = dw_res
                # Draw skeleton
                display_frame = detector.draw_skeleton(frame.copy(), kpts, scores)
                # Check active hands (COCO WholeBody: 91..112 left hand, 112..133 right hand)
                lh_score = np.mean(scores[91:112]) if len(scores) >= 112 else 0.0
                rh_score = np.mean(scores[112:133]) if len(scores) >= 133 else 0.0
                hands_detected = bool(lh_score > 0.10 or rh_score > 0.10)
            else:
                display_frame = frame.copy()
                hands_detected = False

            # Extract normalized (137, 3) tensor (reusing precomputed DWPose result to avoid double GPU inference)
            kp_array = detector.extract_lsec_tensor(frame, precomputed_res=dw_res)

            # 2. Push to temporal inference engine
            pred_label, pred_conf = engine.push_frame(kp_array)

            if pred_label is not None and pred_label != "REPOSO":
                if pred_label != current_prediction:
                    print(f">> [DWPose - NUEVA SEÑA]: {pred_label} ({pred_conf * 100:.1f}%)")
                    clean_pred = clean_ascii(pred_label)
                    if not sign_history or sign_history[-1] != clean_pred:
                        sign_history.append(clean_pred)
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
                    pred_label=current_prediction if (time.time() - last_detection_time < 0.8) else None,
                    pred_conf=current_confidence,
                    hands_detected=hands_detected,
                    target_conf=target_conf,
                )
                # Small GPU badge on top right
                cv2.putText(display_frame, f"GPU: DWPose (CUDA) + CPU FaceMesh | FPS: {current_fps:4.1f}", (w - 440, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 1)
            else:
                # Top bar HUD for free translator
                cv2.rectangle(display_frame, (0, 0), (w, 48), (14, 18, 26), -1)
                cv2.line(display_frame, (0, 48), (w, 48), (0, 220, 255), 1)
                cv2.putText(display_frame, f"LSEC AI - MODO LIBRE CONTINUO | FPS: {current_fps:4.1f} (DWPose GPU)", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 120), 2)
                cv2.putText(display_frame, "[T]: Activar Modo Tutor | [M]: Catalogo | [Q]: Salir", (w - 470, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 1)

                is_active = (time.time() - last_detection_time) < 2.5
                banner_h = 100
                overlay = display_frame.copy()
                bg_color = (16, 28, 20) if (is_active and current_prediction) else (18, 20, 26)
                cv2.rectangle(overlay, (0, h - banner_h), (w, h), bg_color, -1)
                display_frame = cv2.addWeighted(overlay, 0.88, display_frame, 0.12, 0)
                border_color = (0, 255, 120) if (is_active and current_prediction) else (60, 70, 85)
                cv2.line(display_frame, (0, h - banner_h), (w, h - banner_h), border_color, 2)

                # Line 1: Main prediction readout
                if is_active and current_prediction:
                    cv2.putText(
                        display_frame,
                        f">> TRADUCCION: {clean_ascii(current_prediction).upper()}",
                        (25, h - 58),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.92,
                        (0, 255, 120),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        display_frame,
                        f"Precision: {current_confidence * 100:.1f}%",
                        (w - 240, h - 58),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.72,
                        (0, 240, 255),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    cv2.putText(
                        display_frame,
                        "[LISTO] Realiza cualquier sena en vivo... (Traduccion libre una tras otra)",
                        (25, h - 58),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.68,
                        (200, 215, 230),
                        1,
                        cv2.LINE_AA,
                    )

                # Line 2: Multi-sign continuous sequence ribbon
                hist_str = " -> ".join(list(sign_history)[-4:]) if sign_history else "(esperando primera sena...)"
                cv2.putText(
                    display_frame,
                    f"Historial continuo: {hist_str}",
                    (25, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 200, 80),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow(window_name, display_frame)

            raw_key = cv2.waitKeyEx(1)
            key = raw_key & 0xFF if raw_key != -1 else -1

            if key in (ord("q"), ord("Q")):
                if tutor.enabled and tutor.menu_open:
                    tutor.toggle_menu()
                else:
                    break
            elif raw_key == 27:  # ESC
                if tutor.enabled and tutor.menu_open:
                    tutor.toggle_menu()
                else:
                    break

            # -------------------------------------------------------------
            # Navegación del Menú en Pantalla (cuando está abierto)
            # -------------------------------------------------------------
            elif tutor.enabled and tutor.menu_open:
                from lsec_sdk.tutor_engine import SECTIONS_LIST

                # Confirmar selección: ENTER o ESPACIO
                if raw_key in (13, 10) or key in (13, 10, 32):
                    tutor.menu_select()
                    engine.reset()
                    current_prediction = None
                    current_confidence = 0.0
                    print(f">> Seña elegida: {tutor.current_lesson.clean_label if tutor.current_lesson else ''}")

                # Subir en la lista: 'W', 'w' o Flecha Arriba
                elif raw_key in (2490368, 38) or key in (ord("w"), ord("W")):
                    tutor.menu_up()

                # Bajar en la lista: 'S', 's' o Flecha Abajo
                elif raw_key in (2621440, 40) or key in (ord("s"), ord("S")):
                    tutor.menu_down()

                # Página anterior: 'A', 'a' o Flecha Izquierda
                elif raw_key in (2424832, 37) or key in (ord("a"), ord("A")):
                    tutor.menu_prev_page()

                # Página siguiente: 'D', 'd', Flecha Derecha o TAB
                elif raw_key in (2555904, 39, 9, ord("\t")) or key in (ord("d"), ord("D"), 9):
                    tutor.menu_next_page()

                # Cambiar de sección con 'C'
                elif key in (ord("c"), ord("C")):
                    sec = tutor.cycle_section()
                    print(f">> Sección cambiada a: {sec}")

                # Cerrar menú con 'M'
                elif key in (ord("m"), ord("M")):
                    tutor.toggle_menu()

                # Saltar a sección directa 1 a 6
                elif ord("1") <= key <= ord("6"):
                    sec_idx = key - ord("1")
                    if sec_idx < len(SECTIONS_LIST):
                        chosen_sec = SECTIONS_LIST[sec_idx]
                        tutor.set_section(chosen_sec)
                        print(f">> Pestaña: {chosen_sec} ({len(tutor.filtered_lessons)} señas)")

            # -------------------------------------------------------------
            # Controles normales (cuando el Menú está cerrado)
            # -------------------------------------------------------------
            else:
                if key in (ord("t"), ord("T")):
                    tutor.enabled = not tutor.enabled
                    engine.reset()
                    current_prediction = ""
                    current_confidence = 0.0
                    print(f"\n>> Modo cambiado a: {'MODO TUTOR (Guiado con Lecciones)' if tutor.enabled else 'MODO LIBRE (Traductor Continuo de Señas)'}")
                elif key in (ord("m"), ord("M")):
                    tutor.toggle_menu()
                elif key in (ord("c"), ord("C")):
                    tutor.cycle_section()
                    engine.reset()
                    current_prediction = None
                    current_confidence = 0.0
                elif key in (ord("n"), ord("N")):
                    tutor.next_lesson()
                    engine.reset()
                    current_prediction = None
                    current_confidence = 0.0
                elif key in (ord("p"), ord("P")):
                    tutor.prev_lesson()
                    engine.reset()
                    current_prediction = None
                    current_confidence = 0.0
                elif key in (ord("r"), ord("R")):
                    tutor.reset_to_demo()
                    engine.reset()
                    current_prediction = None
                    current_confidence = 0.0
                elif key in (ord("s"), ord("S")):
                    tutor.slow_motion = not tutor.slow_motion
                elif key == 32:  # Barra espaciadora: saltar demo
                    if tutor.enabled:
                        tutor.skip_demo()
                        engine.reset()
                        current_prediction = None
                        current_confidence = 0.0

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n>> Sesión DWPose finalizada.")


if __name__ == "__main__":
    main()

