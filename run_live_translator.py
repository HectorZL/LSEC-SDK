#!/usr/bin/env python3
"""
Real-time Sign Language Translator using Webcam and trained LSEC PyTorch model.
Continuously predicts signs and supports switching phrases one after another.
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

from lsec_sdk.models import LiveInferenceEngine
from lsec_sdk.realtime_capture import RealTimeCapture
from lsec_sdk.tutor_engine import LSECTutorEngine


def main():
    parser = argparse.ArgumentParser(description="LSEC Real-Time Live Translator & Guided Tutor")
    parser.add_argument("--tutor", action="store_true", help="Iniciar directamente en Modo Tutor de Enseñanza Guiada")
    parser.add_argument("--menu", action="store_true", help="Mostrar menú de selección en terminal antes de iniciar")
    args = parser.parse_args()

    checkpoint_file = Path("models/checkpoints/lsec_classifier.pt")
    if not checkpoint_file.exists():
        print(f"ERROR: No se encontró el modelo entrenado en: {checkpoint_file}")
        print("Ejecuta 'python train_classifier.py' para entrenar tu modelo.")
        return

    print("=" * 60)
    print(" LSEC SDK - TRADUCTOR MULTI-FRASE Y TUTOR DE ENSEÑANZA GUIADA")
    print("=" * 60)
    print(f"Cargando modelo desde: {checkpoint_file}")

    engine = LiveInferenceEngine(
        checkpoint_path=checkpoint_file,
        window_size=32,
        min_prediction_confidence=0.55,
    )
    classes = list(engine.label_encoder.label_to_idx.keys())
    print(f"Vocabulario activo ({len(classes)}): {classes}")

    # Initialize Guided Tutor Engine (2 loops, no ghost)
    tutor = LSECTutorEngine(enabled=args.tutor or args.menu, demo_loops=2)
    if tutor.all_lessons:
        print(f"Modo Tutor cargado: {len(tutor.all_lessons)} señas de referencia disponibles.")
    else:
        print("Aviso: No se encontraron secuencias .npy para el Modo Tutor en data/processed/metadata.csv")

    # Interactive Terminal Menu if requested
    if args.menu and tutor.all_lessons:
        print("\n" + "=" * 62)
        print(" SELECCIONA LA SECCION QUE DESEAS REPASAR:")
        print("=" * 62)
        print("  [1] TODAS     (46 señas completas)")
        print("  [2] SALUDOS   (Buenos días, Buenas tardes, Adiós, De nada...)")
        print("  [3] TRABAJO   (Dónde estudió, Abrir cuenta, Crédito, Banco...)")
        print("  [4] COMIDA    (Hambre, Agua, Ya no más, Babaco, Abrelatas...)")
        print("  [5] SALUD     (Cómo te sientes, Cómo está, Está bien, Mal...)")
        print("  [6] GENERAL   (Abecedario, Abrazar, Abrir, Abanico...)")
        print("=" * 62)

        from lsec_sdk.tutor_engine import SECTIONS_LIST
        try:
            sec_choice = input("Elige una sección [1-6] (Enter para [1] TODAS): ").strip()
            if sec_choice in ("1", "2", "3", "4", "5", "6"):
                chosen_sec = SECTIONS_LIST[int(sec_choice) - 1]
                tutor.set_section(chosen_sec)
                print(f">> Sección elegida: {chosen_sec} ({len(tutor.filtered_lessons)} señas)")

                # Option to choose specific sign inside that section
                print("\nSeñas en esta sección:")
                for i, l in enumerate(tutor.filtered_lessons, 1):
                    print(f"  [{i:2d}] {l.clean_label}")
                sign_choice = input("\nElige una seña (o presiona Enter para comenzar con la primera): ").strip()
                if sign_choice and sign_choice.isdigit():
                    s_num = int(sign_choice)
                    if 1 <= s_num <= len(tutor.filtered_lessons):
                        tutor.select_lesson_by_index(s_num - 1)
                        print(f">> Iniciando con: {tutor.current_lesson.clean_label}")
        except Exception:
            pass

    capture = RealTimeCapture(camera_index=0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: No se pudo abrir la cámara web de tu laptop.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "LSEC SDK - Traductor y Tutor de Senas en Vivo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1100, 620)

    current_prediction = ""
    current_confidence = 0.0
    last_detection_time = 0.0
    phrase_history: deque = deque(maxlen=4)

    fps_time = time.time()
    fps_counter = 0
    current_fps = 0.0

    print("\n>> Sistema activo.")
    print("   [M]: Abrir / Cerrar Menú de Señas en Pantalla")
    print("   [T]: Habilitar / Deshabilitar Modo Tutor")
    print("   [N]: Siguiente seña | [P]: Seña anterior")
    print("   [R]: Reiniciar demostración (2 repeticiones)")
    print("   [S]: Alternar velocidad (Normal / Lento)")
    print("   [Q]: Salir\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            # Process with MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = capture.holistic.process(rgb)
            kp_array = capture.extract_landmarks_array(results)

            # Push to sliding-window inference engine
            pred_label, pred_conf = engine.push_frame(kp_array)

            if pred_label is not None and pred_label != "REPOSO":
                if pred_label != current_prediction:
                    phrase_history.append(pred_label)
                    print(f">> [NUEVA SEÑA]: {pred_label} ({pred_conf * 100:.1f}%)")

                current_prediction = pred_label
                current_confidence = pred_conf
                last_detection_time = time.time()

            # Calculate target sign confidence specifically for the current tutor lesson
            target_conf = 0.0
            if tutor.enabled and tutor.current_lesson:
                target_conf = engine.get_target_confidence(tutor.current_lesson.clean_label)

            # Check if user hands are visible in camera (including inertial profile tracking)
            hands_detected = bool(
                results.left_hand_landmarks
                or results.right_hand_landmarks
                or capture.left_hand_active
                or capture.right_hand_active
            )

            # Draw user's own live skeleton tracking
            display_frame = capture.draw_skeleton(frame, results)

            # FPS calculation
            fps_counter += 1
            now = time.time()
            if now - fps_time >= 1.0:
                current_fps = fps_counter / (now - fps_time)
                fps_counter = 0
                fps_time = now

            h, w, _ = display_frame.shape

            # If Tutor Mode is enabled, render the guided learning experience
            if tutor.enabled:
                display_frame = tutor.update_and_render(
                    display_frame,
                    pred_label=current_prediction if (time.time() - last_detection_time < 2.5) else None,
                    pred_conf=current_confidence,
                    hands_detected=hands_detected,
                    target_conf=target_conf,
                )
            else:
                # Top bar HUD for free translator
                cv2.rectangle(display_frame, (0, 0), (w, 55), (15, 15, 15), -1)
                cv2.putText(display_frame, f"FPS: {current_fps:4.1f}", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, "LSEC SDK - TRADUCTOR LIBRE", (140, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)
                cv2.putText(display_frame, "[T]: Modo Tutor | [M]: Menu | [Q]: Salir", (w - 380, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)

                # Bottom Translation Banner
                is_active = (time.time() - last_detection_time) < 2.0
                banner_h = 90
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (0, h - banner_h), (w, h), (15, 25, 15) if is_active else (20, 20, 20), -1)
                display_frame = cv2.addWeighted(overlay, 0.85, display_frame, 0.15, 0)

                if is_active and current_prediction:
                    cv2.putText(
                        display_frame,
                        f"TRADUCCION: {current_prediction}",
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
                        "Esperando sena frente a la camara...",
                        (30, h - 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (120, 120, 120),
                        2,
                        cv2.LINE_AA,
                    )

                if phrase_history:
                    history_text = "Historial: " + " -> ".join(list(phrase_history)[-3:])
                    cv2.putText(display_frame, history_text, (30, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1, cv2.LINE_AA)

            cv2.imshow(window_name, display_frame)

            raw_key = cv2.waitKeyEx(1)
            key = raw_key & 0xFF if raw_key != -1 else -1

            if key in (ord("q"), ord("Q")) or raw_key == 27:
                # If menu is open, ESC closes menu instead of exiting app
                if tutor.enabled and tutor.menu_open:
                    tutor.toggle_menu()
                else:
                    break

            elif tutor.enabled and tutor.menu_open:
                from lsec_sdk.tutor_engine import SECTIONS_LIST
                # Menu navigation keys
                if raw_key in (13, 10) or key in (13, 10, 32):  # Enter or Space to select sign
                    tutor.menu_select()
                    print(f">> Seña elegida: {tutor.current_lesson.clean_label if tutor.current_lesson else ''}")
                elif raw_key in (2490368, 38) or key in (ord("w"), ord("W")):  # Up arrow or W
                    tutor.menu_up()
                elif raw_key in (2621440, 40) or key in (ord("s"), ord("S")):  # Down arrow or S
                    tutor.menu_down()
                elif raw_key in (2424832, 37) or key in (ord("a"), ord("A")):  # Left arrow or A
                    tutor.menu_prev_page()
                elif raw_key in (2555904, 39, 9, ord("\t")) or key in (ord("d"), ord("D"), 9):  # Right arrow, D or Tab
                    tutor.menu_next_page()
                elif key in (ord("c"), ord("C")):  # Cycle category section
                    sec = tutor.cycle_section()
                    print(f">> Sección cambiada a: {sec}")
                elif key in (ord("m"), ord("M")):
                    tutor.toggle_menu()
                elif ord("1") <= key <= ord("6"):
                    # Switch section tab directly by number 1-6
                    sec_idx = key - ord("1")
                    if sec_idx < len(SECTIONS_LIST):
                        chosen_sec = SECTIONS_LIST[sec_idx]
                        tutor.set_section(chosen_sec)
                        print(f">> Sección: {chosen_sec} ({len(tutor.filtered_lessons)} señas)")

            else:
                # Regular live controls
                if key == 32:  # Spacebar skips demo directly to imitation
                    tutor.skip_demo()
                    print(">> Demostración saltada: ¡Ahora hazlo tú!")
                elif key in (ord("m"), ord("M")):
                    tutor.toggle_menu()
                elif key in (ord("c"), ord("C")):
                    sec = tutor.cycle_section()
                    print(f">> Sección activa: {sec}")
                elif key in (ord("t"), ord("T")):
                    status = tutor.toggle()
                    print(f">> Modo Tutor {'ACTIVADO' if status else 'DESACTIVADO'}")
                elif key in (ord("n"), ord("N")):
                    tutor.next_lesson()
                    print(f">> Siguiente lección: {tutor.current_lesson.clean_label if tutor.current_lesson else 'None'}")
                elif key in (ord("p"), ord("P")):
                    tutor.prev_lesson()
                    print(f">> Lección anterior: {tutor.current_lesson.clean_label if tutor.current_lesson else 'None'}")
                elif key in (ord("r"), ord("R")):
                    tutor.reset_to_demo()
                    print(">> Reiniciando demostración de palitos...")
                elif key in (ord("s"), ord("S")):
                    slow = tutor.toggle_speed()
                    print(f">> Velocidad {'0.6x CÁMARA LENTA' if slow else '1.0x NORMAL'}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        capture.close()
        print("\nTraductor cerrado y recursos liberados.")


if __name__ == "__main__":
    main()
