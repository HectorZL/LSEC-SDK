"""
Módulo de Detección Automática de Actividad de Señas (Sign Activity Detection - SAD).
Detecta autónomamente el inicio (onset), desarrollo y conclusión (offset) de una seña
en Lengua de Señas Ecuatoriana (LSEC) mediante análisis cinemático del espacio de señas.
"""

from enum import Enum
from typing import Optional, Tuple, Dict, Any, List, Deque
from collections import deque
import numpy as np


class ActivityState(str, Enum):
    IDLE = "IDLE"              # Usuario en reposo (brazos caídos / mesa)
    ACTIVE = "ACTIVE"          # Seña en ejecución activa dentro del espacio de señas
    COMPLETED = "COMPLETED"    # Seña finalizada con éxito (lista para envío o inferencia)


class SignActivityDetector:
    """
    Detector cinemático de límites de gestos (Sign Activity Detector).
    Elimina la necesidad de presionar teclas o botones, segmentando
    los gestos de forma transparente para el usuario.
    """

    def __init__(
        self,
        fps: float = 30.0,
        pre_buffer_frames: int = 12,
        min_gesture_frames: int = 24,       # ~0.8s a 30 FPS
        max_gesture_frames: int = 105,      # ~3.5s a 30 FPS
        end_cooldown_frames: int = 10,      # ~0.33s de quietud o descenso para cerrar
        wrist_elevation_threshold: float = 0.82,
        motion_energy_threshold: float = 0.0030,
    ):
        self.fps = fps
        self.pre_buffer_frames = pre_buffer_frames
        self.min_gesture_frames = min_gesture_frames
        self.max_gesture_frames = max_gesture_frames
        self.end_cooldown_frames = end_cooldown_frames
        self.wrist_elevation_thresh = wrist_elevation_threshold
        self.motion_energy_thresh = motion_energy_threshold

        self.state: ActivityState = ActivityState.IDLE

        # Circular buffer for pre-gesture frames (captures the onset phase)
        self.pre_buffer_kp: Deque[np.ndarray] = deque(maxlen=pre_buffer_frames)
        self.pre_buffer_frames_raw: Deque[np.ndarray] = deque(maxlen=pre_buffer_frames)

        # Active gesture storage
        self.active_kp: List[np.ndarray] = []
        self.active_raw: List[np.ndarray] = []

        # Cooldown counter
        self.inactive_streak: int = 0
        self.last_kp: Optional[np.ndarray] = None

    def reset(self):
        """Reinicia el detector al estado neutro."""
        self.state = ActivityState.IDLE
        self.pre_buffer_kp.clear()
        self.pre_buffer_frames_raw.clear()
        self.active_kp.clear()
        self.active_raw.clear()
        self.inactive_streak = 0
        self.last_kp = None

    def _is_kinematically_active(self, kp: np.ndarray) -> Tuple[bool, float, float]:
        """
        Evalúa si el fotograma actual presenta manos activas en el espacio de señas.
        Retorna (is_active, min_wrist_y, motion_energy).
        Formato de kp: (137, 3) OpenPose/LSEC o (133, 3) COCO-WholeBody.
        """
        # Puntos de muñeca en OpenPose: 4 (RWrist), 7 (LWrist)
        # O en COCO WholeBody: 10 (RWrist), 9 (LWrist)
        is_openpose = len(kp) >= 137
        rw_idx = 4 if is_openpose else 10
        lw_idx = 7 if is_openpose else 9

        rw_y = kp[rw_idx, 1] if len(kp) > rw_idx else 1.0
        lw_y = kp[lw_idx, 1] if len(kp) > lw_idx else 1.0
        rw_c = kp[rw_idx, 2] if len(kp) > rw_idx and kp.shape[1] > 2 else 0.5
        lw_c = kp[lw_idx, 2] if len(kp) > lw_idx and kp.shape[1] > 2 else 0.5

        min_wrist_y = min(rw_y if rw_c > 0.15 else 1.0, lw_y if lw_c > 0.15 else 1.0)
        elevated = bool(min_wrist_y < self.wrist_elevation_thresh)

        # Energía cinética articular (diferencia con el fotograma anterior)
        motion_energy = 0.0
        if self.last_kp is not None:
            # Manos en OpenPose: 95..116 (LHand), 116..137 (RHand)
            h_start = 95 if is_openpose else 91
            h_end = 137 if is_openpose else 133
            if len(kp) >= h_end and len(self.last_kp) >= h_end:
                diff = np.abs(kp[h_start:h_end, :2] - self.last_kp[h_start:h_end, :2])
                motion_energy = float(np.mean(diff))

        self.last_kp = kp.copy()
        is_active = elevated and (motion_energy >= self.motion_energy_thresh)
        return is_active, min_wrist_y, motion_energy

    def process_frame(
        self,
        kp_array: np.ndarray,
        raw_frame: Optional[np.ndarray] = None,
    ) -> Tuple[ActivityState, Optional[Dict[str, Any]]]:
        """
        Procesa un fotograma y avanza la máquina de estados de detección de actividad.
        Retorna (estado_actual, payload_completado_si_hubo).
        """
        is_active, wrist_y, energy = self._is_kinematically_active(kp_array)

        if self.state == ActivityState.IDLE:
            self.pre_buffer_kp.append(kp_array)
            if raw_frame is not None:
                self.pre_buffer_frames_raw.append(raw_frame)

            if is_active:
                # Transición: IDLE -> ACTIVE (Inicio de seña)
                self.state = ActivityState.ACTIVE
                self.inactive_streak = 0
                # Absorber el pre-buffer para conservar la fase de preparación
                self.active_kp = list(self.pre_buffer_kp)
                self.active_raw = list(self.pre_buffer_frames_raw)
                self.pre_buffer_kp.clear()
                self.pre_buffer_frames_raw.clear()
                return ActivityState.ACTIVE, None
            else:
                return ActivityState.IDLE, None

        elif self.state == ActivityState.ACTIVE:
            self.active_kp.append(kp_array)
            if raw_frame is not None:
                self.active_raw.append(raw_frame)

            total_frames = len(self.active_kp)

            # Verificar si continúa activa o si descendió a reposo
            if not is_active:
                self.inactive_streak += 1
            else:
                self.inactive_streak = 0

            # Criterios de cierre:
            # 1. Racha de inactividad alcanzada (manos bajaron o se detuvieron)
            # 2. Timeout de seguridad (excedió duración máxima)
            reached_cooldown = (self.inactive_streak >= self.end_cooldown_frames)
            reached_max = (total_frames >= self.max_gesture_frames)

            if reached_cooldown or reached_max:
                if total_frames >= self.min_gesture_frames:
                    # Gesto válido completado
                    seq_kp = np.array(self.active_kp, dtype=np.float32)
                    payload = {
                        "frames_count": total_frames,
                        "duration_sec": round(total_frames / self.fps, 2),
                        "keypoints_sequence": seq_kp,
                        "raw_frames": list(self.active_raw) if self.active_raw else None,
                        "forced_timeout": reached_max,
                    }
                    self.reset()
                    return ActivityState.COMPLETED, payload
                else:
                    # Movimiento espurio demasiado corto (<0.8s) -> Descartar como ruido
                    self.reset()
                    return ActivityState.IDLE, None

            return ActivityState.ACTIVE, None

        return self.state, None

