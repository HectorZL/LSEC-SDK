"""
Test Suite: Validación de Modalidades de Señas LSEC (1 Mano, 2 Manos y Complejas).
Garantiza precisión en oclusiones bimanuales, tracking de dedos y reposo silencioso.
"""

from pathlib import Path
import numpy as np
import pytest

from lsec_sdk.models.inference import LiveInferenceEngine
from lsec_sdk.tutor_engine import LSECTutorEngine


@pytest.fixture(scope="module")
def inference_engine():
    checkpoint_path = Path("models/checkpoints/lsec_classifier.pt")
    assert checkpoint_path.exists(), f"Checkpoint no encontrado: {checkpoint_path}"
    engine = LiveInferenceEngine(
        checkpoint_path=checkpoint_path,
        window_size=32,
        min_prediction_confidence=0.60,
    )
    return engine


@pytest.fixture(scope="module")
def tutor_engine():
    engine = LSECTutorEngine(enabled=True)
    return engine


class TestSignModalities:
    """Valida el rendimiento del clasificador en las tres modalidades clave."""

    def test_one_handed_signs(self, inference_engine):
        """Prueba señas unimanuales (1 mano): Adiós, De nada, Agua, Hambre, Mal, Aprender."""
        one_handed = [
            ("dwpose_adios.npy", "Adiós"),
            ("dwpose_de-nada.npy", "De nada"),
            ("dwpose_agua.npy", "Agua"),
            ("dwpose_hambre.npy", "Hambre"),
            ("dwpose_mal.npy", "Mal"),
            ("dwpose_aprender.npy", "Aprender"),
        ]
        seq_dir = Path("data/processed/sequences")

        for filename, label in one_handed:
            file_path = seq_dir / filename
            assert file_path.exists(), f"Falta archivo {filename}"
            seq = np.load(file_path)

            inference_engine.reset()
            max_target_conf = 0.0
            detected = None

            for frame in seq:
                pred, conf = inference_engine.push_frame(frame)
                if pred is not None:
                    detected = pred
                tc = inference_engine.get_target_confidence(label)
                if tc > max_target_conf:
                    max_target_conf = tc

            assert max_target_conf >= 0.70 or detected == label, (
                f"Fallo en seña de 1 mano '{label}': MaxTarget={max_target_conf:.2f}, Pred={detected}"
            )

    def test_two_handed_signs(self, inference_engine):
        """Prueba señas bimanuales estándar (2 manos): Buenos días, Buenas tardes, ¿Cómo está?, Ya no más."""
        two_handed = [
            ("dwpose_buenos-dias.npy", "Buenos días"),
            ("dwpose_buenas-tardes.npy", "Buenas tardes"),
            ("dwpose_buenas-noches.npy", "Buenas noches"),
            ("dwpose_como-esta.npy", "¿Cómo está?"),
            ("dwpose_ya-no-mas.npy", "Ya no más"),
            ("dwpose_donde.npy", "¿Dónde?"),
        ]
        seq_dir = Path("data/processed/sequences")

        for filename, label in two_handed:
            file_path = seq_dir / filename
            assert file_path.exists(), f"Falta archivo {filename}"
            seq = np.load(file_path)

            inference_engine.reset()
            max_target_conf = 0.0
            detected = None

            for frame in seq:
                pred, conf = inference_engine.push_frame(frame)
                if pred is not None:
                    detected = pred
                tc = inference_engine.get_target_confidence(label)
                if tc > max_target_conf:
                    max_target_conf = tc

            assert max_target_conf >= 0.70 or detected == label, (
                f"Fallo en seña bimanual '{label}': MaxTarget={max_target_conf:.2f}, Pred={detected}"
            )

    def test_complex_occlusion_signs(self, inference_engine):
        """
        Prueba señas complejas:
        - Abrazar: cruce bimanual profundo sobre el pecho (prueba de oclusión DWPose GPU)
        - Familia: círculo 3D con forma 'F'
        - Feliz: movimiento en pecho + rasgo no manual facial
        - Importante: contacto digital de precisión
        - Amigo íntimo: interbloqueo y cruce de dedos índices
        - Universidad: rotación de configuración 'U' sobre palma base
        """
        complex_signs = [
            ("dwpose_abrazar.npy", "Abrazar"),
            ("dwpose_familia.npy", "Familia"),
            ("dwpose_feliz.npy", "Feliz"),
            ("dwpose_importante.npy", "Importante"),
            ("dwpose_amigo-intimo.npy", "Amigo íntimo"),
            ("dwpose_universidad.npy", "Universidad"),
        ]
        seq_dir = Path("data/processed/sequences")

        for filename, label in complex_signs:
            file_path = seq_dir / filename
            assert file_path.exists(), f"Falta archivo {filename}"
            seq = np.load(file_path)

            inference_engine.reset()
            max_target_conf = 0.0
            detected = None

            for frame in seq:
                pred, conf = inference_engine.push_frame(frame)
                if pred is not None:
                    detected = pred
                tc = inference_engine.get_target_confidence(label)
                if tc > max_target_conf:
                    max_target_conf = tc

            assert max_target_conf >= 0.70 or detected == label, (
                f"Fallo en seña compleja '{label}': MaxTarget={max_target_conf:.2f}, Pred={detected}"
            )

    def test_reposo_zero_false_positives(self, inference_engine):
        """Verifica que ninguna secuencia de reposo emita predicciones espurias."""
        seq_dir = Path("data/processed/sequences")
        reposo_files = list(seq_dir.glob("dwpose_reposo_*.npy"))
        assert len(reposo_files) >= 20, "Faltan secuencias de reposo"

        false_positives = []
        for rp in reposo_files:
            seq = np.load(rp)
            inference_engine.reset()
            for frame in seq:
                pred, conf = inference_engine.push_frame(frame)
                if pred is not None and pred != "REPOSO":
                    false_positives.append((rp.name, pred, conf))
                    break

        assert len(false_positives) == 0, f"Falsos positivos detectados en reposo: {false_positives}"

    def test_tutor_catalog_integrity(self, tutor_engine):
        """Verifica que el catálogo del tutor contenga todas las 24 lecciones con secciones válidas."""
        assert len(tutor_engine.all_lessons) == 24, f"Se esperaban 24 lecciones, encontradas: {len(tutor_engine.all_lessons)}"
        labels = [l.clean_label for l in tutor_engine.all_lessons]
        assert "Abrazar" in labels
        assert "Familia" in labels
        assert "Feliz" in labels
        assert "Amigo intimo" in labels or "Amigo íntimo" in labels or any("Amigo" in l for l in labels)
