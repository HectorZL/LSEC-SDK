"""Guided Teaching & Learning Module (Modo Tutor / Aprendizaje Guiado) for LSEC SDK.

Features:
1. Thematic Sections / Categories:
   - SALUDOS: Saludos y Cortesía (Buenos días, Adiós, De nada...)
   - TRABAJO: Trabajo, Trámites y Banco (Dónde estudió, Abrir cuenta, Crédito...)
   - COMIDA: Comida y Rutina Diaria (Hambre, Agua, Ya no más...)
   - SALUD: Salud y Estado Personal (Cómo te sientes, Está bien, Mal...)
   - GENERAL: Vocabulario Alfabético (Abecedario, Abanico, Abrir...)
   - TODAS: Catálogo completo de 46 señas.
2. Teacher Stick-Figure ("Modelo de Palitos"): Animated playback in top-right PiP (2 loops).
3. Clean Camera View: No ghost skeleton on the user.
4. Interactive Selection Menu: On-screen [M] with category tabs [1-6], arrow navigation, and Enter to choose.
5. Quick Section Switching: Press [C] anytime during live camera to jump between sections.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from .constants import (
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    FACE_SLICE,
    HAND_LEFT_SLICE,
    HAND_RIGHT_SLICE,
    POSE_SLICE,
)
from .visualizer import Visualizer

SECTIONS_LIST = ["TODAS", "SALUDOS", "TRABAJO", "COMIDA", "SALUD", "GENERAL"]


def clean_ascii(text: str) -> str:
    """Sanitizes text for OpenCV putText to avoid '??' glyphs."""
    replacements = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
        "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
        "ñ": "n", "Ñ": "N",
        "¿": "", "¡": "",
    }
    res = text
    for k, v in replacements.items():
        res = res.replace(k, v)
    return res.encode("ascii", "ignore").decode("ascii")


def matches_target(pred_label: Optional[str], target_label: str) -> bool:
    """Fuzzy matching between live predicted label and target lesson label."""
    if not pred_label:
        return False
    p = clean_ascii(pred_label).strip().upper()
    t = clean_ascii(target_label).strip().upper()
    if p == t or p in t or t in p:
        return True

    # Check key word tokens (e.g. ABRAZAR in ABRAZAR, ABRAZO)
    p_tokens = [w for w in p.replace(",", " ").replace("?", " ").split() if len(w) > 3]
    t_tokens = [w for w in t.replace(",", " ").replace("?", " ").split() if len(w) > 3]
    return any(pt in t_tokens for pt in p_tokens) or any(tt in p_tokens for tt in t_tokens)


def classify_section(label: str) -> str:
    """Categorizes sign labels into intuitive pedagogical sections."""
    lbl = clean_ascii(label).upper()
    if any(w in lbl for w in ["ADIOS", "BUENOS DIAS", "BUENAS TARDES", "BUENAS NOCHES", "DE NADA", "UN MOMENTO", "HACE TIEMPO", "AMIGO INTIMO", "FAMILIA"]):
        return "SALUDOS"
    if any(w in lbl for w in ["DONDE ESTUDIO", "CUENTA", "CREDITO", "REQUISITOS", "AYUDAR", "ABOGADO", "ABONAR", "FOLLETO", "A DIARIO", "UNIVERSIDAD", "APRENDER", "IMPORTANTE"]):
        return "TRABAJO"
    if any(w in lbl for w in ["HAMBRE", "AGUA", "BABACO", "ABRELATAS", "ABREFACIL", "ABSORBER", "YA NO MAS"]):
        return "COMIDA"
    if any(w in lbl for w in ["COMO TE SIENTES", "COMO ESTA", "ESTA BIEN", "MAL", "ABDOMEN", "ABORTAR", "ABSTINENCIA", "ABSTENERSE", "AMEDRENTARSE", "MOCO", "FELIZ", "ABRAZAR"]):
        return "SALUD"
    return "GENERAL"


class TutorState(Enum):
    DEMO = "DEMO"             # Watching stick-figure demonstration (2 loops)
    COUNTDOWN = "COUNTDOWN"   # 3.. 2.. 1.. Get ready!
    PRACTICE = "PRACTICE"     # User imitates the sign in front of webcam
    SUCCESS = "SUCCESS"       # Great job! Sign recognized


@dataclass
class TutorLesson:
    label: str
    clean_label: str
    sequence_id: str
    tensor: np.ndarray        # Shape: (T, 137, 3)
    num_frames: int
    fps: float
    section: str = "GENERAL"


class LSECTutorEngine:
    """Interactive teaching system with stick-figure animations, thematic sections, and on-screen menu."""

    def __init__(
        self,
        metadata_csv: Path = Path("data/processed/metadata.csv"),
        enabled: bool = True,
        demo_loops: int = 2,
        practice_timeout_sec: float = 9.0,
        default_section: str = "TODAS",
    ):
        self.enabled = enabled
        self.demo_loops = demo_loops
        self.practice_timeout_sec = practice_timeout_sec
        self.slow_motion = False

        self.visualizer = Visualizer(confidence_threshold=0.1)
        self.all_lessons: List[TutorLesson] = []
        self.active_section: str = default_section.upper() if default_section.upper() in SECTIONS_LIST else "TODAS"
        self.current_lesson_idx = 0

        # State management
        self.state = TutorState.DEMO
        self.state_start_time = time.time()
        self.current_demo_loop = 1
        self.demo_frame_idx = 0
        self.last_frame_tick = time.time()

        # Imitation & scoring
        self.best_practice_conf = 0.0
        self.practice_start_time = 0.0
        self.countdown_remaining = 3
        self.practice_match_streak = 0
        self.required_streak = 7  # Require 7 frames (~0.35s) of sustained execution before triggering success

        # Interactive On-Screen Menu
        self.menu_open = False
        self.menu_selected_idx = 0
        self.page_size = 7

        self._load_lessons_from_metadata(metadata_csv)

    def _load_lessons_from_metadata(self, csv_path: Path):
        """Loads available reference sequences from metadata.csv and classifies into sections."""
        if not csv_path.exists():
            return

        try:
            df = pd.read_csv(csv_path)
            # Filter non-empty labels, ignore REPOSO as lesson
            df_valid = df[df["label"].notna() & (df["label"].str.strip() != "") & (df["label"] != "REPOSO")]
            df_unique = df_valid.drop_duplicates(subset=["label"], keep="first")

            for _, row in df_unique.iterrows():
                npy_p = Path(row["npy_path"])
                if npy_p.exists():
                    tensor = np.load(npy_p)
                    raw_lbl = str(row["label"])
                    sec = classify_section(raw_lbl)
                    self.all_lessons.append(
                        TutorLesson(
                            label=raw_lbl,
                            clean_label=clean_ascii(raw_lbl),
                            sequence_id=str(row["sequence_id"]),
                            tensor=tensor,
                            num_frames=int(row["num_frames"]),
                            fps=float(row.get("fps", 25.0)),
                            section=sec,
                        )
                    )
        except Exception as e:
            print(f"[TutorEngine] Error cargando lecciones: {e}")

    @property
    def filtered_lessons(self) -> List[TutorLesson]:
        """Returns lessons filtered by active thematic section."""
        if self.active_section == "TODAS":
            return self.all_lessons
        return [l for l in self.all_lessons if l.section == self.active_section]

    @property
    def lessons(self) -> List[TutorLesson]:
        """Alias for filtered_lessons for backward compatibility."""
        return self.filtered_lessons

    @property
    def current_lesson(self) -> Optional[TutorLesson]:
        lessons = self.filtered_lessons
        if not lessons:
            return None
        idx = max(0, min(self.current_lesson_idx, len(lessons) - 1))
        return lessons[idx]

    def set_section(self, section_name: str):
        """Sets the active thematic section and resets demo to first sign."""
        sec = section_name.upper()
        if sec in SECTIONS_LIST:
            self.active_section = sec
            self.current_lesson_idx = 0
            self.menu_selected_idx = 0
            self.reset_to_demo()

    def cycle_section(self) -> str:
        """Cycles to the next thematic section (e.g. SALUDOS -> TRABAJO -> COMIDA...)."""
        curr_idx = SECTIONS_LIST.index(self.active_section)
        next_sec = SECTIONS_LIST[(curr_idx + 1) % len(SECTIONS_LIST)]
        self.set_section(next_sec)
        return next_sec

    def toggle(self) -> bool:
        """Enables or disables guided tutor mode."""
        self.enabled = not self.enabled
        if self.enabled:
            self.reset_to_demo()
        return self.enabled

    def toggle_speed(self) -> bool:
        """Toggles between normal speed (1.0x) and slow-motion (0.55x)."""
        self.slow_motion = not self.slow_motion
        return self.slow_motion

    def toggle_menu(self) -> bool:
        """Opens or closes the on-screen sign selection menu."""
        self.menu_open = not self.menu_open
        if self.menu_open:
            self.menu_selected_idx = self.current_lesson_idx
        return self.menu_open

    def menu_up(self):
        """Moves cursor up in menu."""
        lessons = self.filtered_lessons
        if not lessons:
            return
        self.menu_selected_idx = (self.menu_selected_idx - 1) % len(lessons)

    def menu_down(self):
        """Moves cursor down in menu."""
        lessons = self.filtered_lessons
        if not lessons:
            return
        self.menu_selected_idx = (self.menu_selected_idx + 1) % len(lessons)

    def menu_next_page(self):
        """Jumps to next page of signs in current section."""
        lessons = self.filtered_lessons
        if not lessons:
            return
        self.menu_selected_idx = min(len(lessons) - 1, self.menu_selected_idx + self.page_size)

    def menu_prev_page(self):
        """Jumps to previous page of signs in current section."""
        lessons = self.filtered_lessons
        if not lessons:
            return
        self.menu_selected_idx = max(0, self.menu_selected_idx - self.page_size)

    def menu_select(self):
        """Confirms selection in menu and starts teaching that sign."""
        lessons = self.filtered_lessons
        if not lessons:
            return
        self.current_lesson_idx = self.menu_selected_idx
        self.menu_open = False
        self.reset_to_demo()

    def select_lesson_by_index(self, idx: int):
        """Selects a lesson directly by index."""
        lessons = self.filtered_lessons
        if 0 <= idx < len(lessons):
            self.current_lesson_idx = idx
            self.menu_selected_idx = idx
            self.reset_to_demo()

    def next_lesson(self):
        """Advances to next sign in current section."""
        lessons = self.filtered_lessons
        if not lessons:
            return
        self.current_lesson_idx = (self.current_lesson_idx + 1) % len(lessons)
        self.reset_to_demo()

    def prev_lesson(self):
        """Goes back to previous sign in current section."""
        lessons = self.filtered_lessons
        if not lessons:
            return
        self.current_lesson_idx = (self.current_lesson_idx - 1) % len(lessons)
        self.reset_to_demo()

    def reset_to_demo(self):
        """Restarts the 2-loop demonstration phase."""
        self.state = TutorState.DEMO
        self.current_demo_loop = 1
        self.demo_frame_idx = 0
        self.state_start_time = time.time()
        self.last_frame_tick = time.time()
        self.best_practice_conf = 0.0
        self.practice_match_streak = 0

    def skip_demo(self):
        """Skips demonstration directly to practice imitation mode."""
        self.state = TutorState.PRACTICE
        self.practice_start_time = time.time()
        self.best_practice_conf = 0.0
        self.practice_match_streak = 0

    def _render_teacher_canvas(self, width: int = 340, height: int = 240) -> np.ndarray:
        """Renders dark sleek viewport with the teacher's stick figure."""
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (20, 22, 28)

        lesson = self.current_lesson
        if lesson is None or lesson.num_frames == 0:
            return canvas

        frame_idx = int(self.demo_frame_idx) % lesson.num_frames
        keypoints = lesson.tensor[frame_idx]

        canvas = self.visualizer.draw_skeleton(
            canvas,
            keypoints,
            width=width,
            height=height,
            normalized=True,
        )

        # Header tag on teacher canvas
        cv2.rectangle(canvas, (0, 0), (width, 28), (35, 38, 48), -1)
        speed_tag = "0.6x Lento" if self.slow_motion else "1.0x Normal"
        cv2.putText(canvas, f"PALITOS: {speed_tag}", (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1)

        # Loop badge: Exactly 2 loops (Rep: 1/2 or 2/2)
        loop_str = f"Rep: {self.current_demo_loop}/{self.demo_loops}"
        cv2.putText(canvas, loop_str, (width - 85, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Progress line
        prog_ratio = frame_idx / max(1, lesson.num_frames - 1)
        bar_w = int(width * prog_ratio)
        cv2.line(canvas, (0, height - 3), (bar_w, height - 3), (0, 220, 255), 3)

        return canvas

    def _render_menu_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Renders the interactive menu with thematic section tabs and sign lists."""
        h, w, _ = frame.shape
        menu_w = min(800, w - 80)
        menu_h = min(500, h - 80)
        x1 = (w - menu_w) // 2
        y1 = (h - menu_h) // 2
        x2 = x1 + menu_w
        y2 = y1 + menu_h

        # Translucent dark backdrop
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (15, 18, 24), -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 255), 2)
        frame = cv2.addWeighted(overlay, 0.93, frame, 0.07, 0)

        # Title bar
        cv2.rectangle(frame, (x1, y1), (x2, y1 + 44), (28, 34, 46), -1)
        cv2.putText(frame, "CATALOGO DE SENAS - ELIGE TU SECCION Y SENA", (x1 + 20, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 240, 255), 2)

        # Section Tabs Row
        tab_y = y1 + 72
        tab_x = x1 + 20
        for i, s_name in enumerate(SECTIONS_LIST, 1):
            is_active_sec = (s_name == self.active_section)
            sec_count = len(self.all_lessons) if s_name == "TODAS" else len([l for l in self.all_lessons if l.section == s_name])
            label_tab = f"[{i}] {s_name} ({sec_count})"

            tw = len(label_tab) * 9 + 16
            tab_bg = (0, 140, 180) if is_active_sec else (32, 38, 48)
            tab_fg = (255, 255, 255) if is_active_sec else (180, 190, 200)

            cv2.rectangle(frame, (tab_x, tab_y - 20), (tab_x + tw, tab_y + 8), tab_bg, -1)
            cv2.putText(frame, label_tab, (tab_x + 8, tab_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.48, tab_fg, 2 if is_active_sec else 1)
            tab_x += tw + 10

        # Current Page Calculation
        lessons = self.filtered_lessons
        total_p = max(1, ((len(lessons) - 1) // self.page_size) + 1)
        curr_p = max(1, (self.menu_selected_idx // self.page_size) + 1) if lessons else 1

        start_idx = (curr_p - 1) * self.page_size
        end_idx = min(len(lessons), start_idx + self.page_size)

        # List Items
        item_y = y1 + 120
        if not lessons:
            cv2.putText(frame, "No hay senas en esta seccion.", (x1 + 40, item_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 1)
        else:
            for idx in range(start_idx, end_idx):
                item = lessons[idx]
                is_selected = (idx == self.menu_selected_idx)
                is_active_lesson = (idx == self.current_lesson_idx)

                # Highlight Row
                if is_selected:
                    cv2.rectangle(frame, (x1 + 15, item_y - 22), (x2 - 15, item_y + 10), (35, 65, 85), -1)
                    prefix = ">> "
                    color = (0, 255, 220)
                else:
                    prefix = "   "
                    color = (210, 210, 210)

                tag = f"{prefix}{idx + 1:02d}. {item.clean_label}"
                if is_active_lesson:
                    tag += "  [EN PRACTICA]"

                cv2.putText(frame, tag[:50], (x1 + 30, item_y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2 if is_selected else 1)
                item_y += 36

        # Footer controls
        cv2.rectangle(frame, (x1, y2 - 45), (x2, y2), (22, 26, 36), -1)
        footer_txt = "[1-6]: Seccion | [W/S o Flechas]: Mover | [ENTER]: Practicar | [A/D]: Pagina | [M]: Cerrar"
        cv2.putText(frame, footer_txt, (x1 + 20, y2 - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 210, 225), 1)

        return frame

    def update_and_render(
        self,
        user_frame: np.ndarray,
        pred_label: Optional[str] = None,
        pred_conf: float = 0.0,
        hands_detected: bool = True,
        target_conf: float = 0.0,
    ) -> np.ndarray:
        """
        Main update step for guided teaching mode.
        Renders teacher PiP (2 loops), thematic HUD banner, live detection readout, and interactive menu.
        """
        if not self.enabled or not self.all_lessons:
            return user_frame

        h, w, _ = user_frame.shape
        lessons = self.filtered_lessons
        lesson = self.current_lesson
        now = time.time()

        is_frame_match = False
        effective_score = float(target_conf)

        if lesson is not None:
            # Check match with predicted label
            if pred_label is not None and matches_target(pred_label, lesson.clean_label):
                if pred_conf > effective_score:
                    effective_score = pred_conf

            # A frame match requires confident recognition of the target sign (>=0.70)
            if effective_score >= 0.70:
                is_frame_match = True
            elif pred_label is not None and matches_target(pred_label, lesson.clean_label) and pred_conf >= 0.68:
                is_frame_match = True

        # -------------------------------------------------------------
        # 1. State Machine Progression (Paused when menu is open)
        # -------------------------------------------------------------
        if not self.menu_open and lesson is not None:
            if self.state == TutorState.DEMO:
                self.practice_match_streak = 0
                step_dt = 1.0 / (lesson.fps * (0.55 if self.slow_motion else 1.0))
                if now - self.last_frame_tick >= step_dt:
                    self.demo_frame_idx += 1
                    self.last_frame_tick = now

                    # Completed one full video loop
                    if self.demo_frame_idx >= lesson.num_frames:
                        self.demo_frame_idx = 0
                        self.current_demo_loop += 1
                        if self.current_demo_loop > self.demo_loops:
                            # Move to Countdown after 2 loops
                            self.state = TutorState.COUNTDOWN
                            self.state_start_time = now

            elif self.state == TutorState.COUNTDOWN:
                self.practice_match_streak = 0
                elapsed = now - self.state_start_time
                self.countdown_remaining = max(1, 3 - int(elapsed))
                if elapsed >= 3.0:
                    self.state = TutorState.PRACTICE
                    self.practice_start_time = now
                    self.best_practice_conf = 0.0
                    self.practice_match_streak = 0

            elif self.state == TutorState.PRACTICE:
                if is_frame_match:
                    self.practice_match_streak += 1
                    if effective_score > self.best_practice_conf:
                        self.best_practice_conf = effective_score
                else:
                    self.practice_match_streak = max(0, self.practice_match_streak - 1)

                # Require sustained execution streak (7 frames) to prevent premature trigger
                if self.practice_match_streak >= self.required_streak:
                    self.state = TutorState.SUCCESS
                    self.state_start_time = now
                    self.practice_match_streak = 0

                # Timeout resets demo
                if (now - self.practice_start_time) > self.practice_timeout_sec:
                    self.reset_to_demo()

            elif self.state == TutorState.SUCCESS:
                self.practice_match_streak = 0
                if (now - self.state_start_time) > 3.5:
                    self.next_lesson()

        # -------------------------------------------------------------
        # 2. Render Teacher Stick-Figure PiP (Top-Right)
        # -------------------------------------------------------------
        if lesson is not None:
            teacher_w, teacher_h = 320, 220
            teacher_img = self._render_teacher_canvas(width=teacher_w, height=teacher_h)

            margin = 15
            pip_x = w - teacher_w - margin
            pip_y = 65

            cv2.rectangle(user_frame, (pip_x - 2, pip_y - 2), (pip_x + teacher_w + 2, pip_y + teacher_h + 2), (0, 220, 255), 2)
            user_frame[pip_y : pip_y + teacher_h, pip_x : pip_x + teacher_w] = teacher_img

        # -------------------------------------------------------------
        # 3. Status HUD (Thematic Section + Clean ASCII)
        # -------------------------------------------------------------
        banner_h = 50
        cv2.rectangle(user_frame, (0, 0), (w, banner_h), (20, 24, 30), -1)

        total_in_sec = len(lessons)
        curr_in_sec = (self.current_lesson_idx + 1) if lessons else 0
        lbl_text = lesson.clean_label if lesson else "SIN SENAS"

        # Section tag + Lesson Name
        lesson_tag = f"[{self.active_section}] LECCION {curr_in_sec}/{total_in_sec}: \"{lbl_text}\""
        cv2.putText(user_frame, lesson_tag, (20, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 240, 255), 2)

        # Quick navigation helpers on top right
        cv2.putText(user_frame, "[M]: Menu | [C]: Seccion", (w - 460, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 0), 2)

        # Live Detection Feedback Bar (above bottom bar)
        det_y = h - 68
        streak_pct = min(1.0, self.practice_match_streak / float(self.required_streak))
        if hands_detected:
            if self.state == TutorState.SUCCESS:
                det_txt = f"[OK] SENA CORRECTA: \"{lesson.clean_label}\" ({int(self.best_practice_conf * 100)}%)"
                cv2.rectangle(user_frame, (15, det_y - 20), (min(w - 20, 25 + len(det_txt) * 11 + 25), det_y + 8), (15, 38, 15), -1)
                cv2.putText(user_frame, det_txt, (25, det_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 120), 2)
            elif self.state == TutorState.PRACTICE and streak_pct > 0:
                det_txt = f"Imitando \"{lesson.clean_label}\": {int(streak_pct * 100)}% sostenido ({int(effective_score * 100)}%)"
                cv2.rectangle(user_frame, (15, det_y - 20), (min(w - 20, 25 + len(det_txt) * 11 + 25), det_y + 8), (20, 36, 20), -1)
                cv2.putText(user_frame, det_txt, (25, det_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 120), 2)
                # Draw imitation progress bar
                pb_x, pb_y, pb_w, pb_h = 25, det_y + 12, 280, 8
                cv2.rectangle(user_frame, (pb_x, pb_y), (pb_x + pb_w, pb_y + pb_h), (40, 45, 55), -1)
                cv2.rectangle(user_frame, (pb_x, pb_y), (pb_x + int(pb_w * streak_pct), pb_y + pb_h), (0, 255, 120), -1)
            elif effective_score >= 0.35:
                color = (0, 255, 120) if effective_score >= 0.70 else (0, 220, 255)
                det_txt = f"Practicando \"{lesson.clean_label}\" - Coincidencia: {int(effective_score * 100)}%"
                cv2.rectangle(user_frame, (15, det_y - 20), (min(w - 20, 25 + len(det_txt) * 11 + 25), det_y + 8), (20, 26, 20), -1)
                cv2.putText(user_frame, det_txt, (25, det_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)
            elif pred_label and pred_label != "REPOSO":
                det_txt = f"Postura detectada: \"{clean_ascii(pred_label)}\" ({int(pred_conf * 100)}%)"
                cv2.rectangle(user_frame, (15, det_y - 20), (min(w - 20, 25 + len(det_txt) * 11 + 25), det_y + 8), (25, 20, 15), -1)
                cv2.putText(user_frame, det_txt, (25, det_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            else:
                cv2.putText(user_frame, f"Detectando: Esperando movimiento para \"{lesson.clean_label}\"...", (25, det_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 160, 160), 1)
        else:
            warn_txt = "[!] ATENCION: Coloca tu mano frente a la camara para realizar la seña"
            cv2.rectangle(user_frame, (15, det_y - 20), (min(w - 20, 25 + len(warn_txt) * 11 + 20), det_y + 8), (15, 20, 60), -1)
            cv2.putText(user_frame, warn_txt, (25, det_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)

        # Bottom Instructions & Pedagogical Feedback
        bar_y = h - 50
        cv2.rectangle(user_frame, (0, bar_y), (w, h), (18, 18, 22), -1)

        if self.state == TutorState.DEMO and lesson is not None:
            cv2.putText(
                user_frame,
                f"1. OBSERVA LOS PALITOS: Demostracion {self.current_demo_loop}/{self.demo_loops} ([ESPACIO]: Saltar a imitar | [R]: Reiniciar)",
                (25, bar_y + 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 200),
                2,
            )

        elif self.state == TutorState.COUNTDOWN:
            cv2.putText(
                user_frame,
                f">> PREPARATE PARA IMITAR EN: {self.countdown_remaining}...",
                (25, bar_y + 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 220, 255),
                2,
            )

        elif self.state == TutorState.PRACTICE and lesson is not None:
            rem_time = max(0, int(self.practice_timeout_sec - (now - self.practice_start_time)))
            score_txt = f"{int(self.best_practice_conf * 100)}%" if self.best_practice_conf > 0 else "0%"
            cv2.putText(
                user_frame,
                f"2. AHORA HAZLO TU! Imitando: \"{lesson.clean_label}\" ({rem_time}s) | Precision: {score_txt}",
                (25, bar_y + 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 120),
                2,
            )

        elif self.state == TutorState.SUCCESS and lesson is not None:
            cv2.putText(
                user_frame,
                f"EXCELENTE! Sena reconocida con exito ({int(self.best_practice_conf * 100)}%) - [N]: Siguiente",
                (25, bar_y + 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        # -------------------------------------------------------------
        # 4. Render Menu Overlay if Active
        # -------------------------------------------------------------
        if self.menu_open:
            user_frame = self._render_menu_overlay(user_frame)

        return user_frame
