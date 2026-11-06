#!/usr/bin/env python3
"""
Pipeline de Extracción y Entrenamiento del Pack Conversacional LSEC (Ecuador CONADIS)
Entrena un clasificador BiLSTM con Atención para las 17 señas auténticas ecuatorianas + REPOSO.
Filtro cinemático de espacio de señas y velocidad de movimiento para eliminar falsos positivos.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import time
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from lsec_sdk.dwpose_extractor import DWPoseDetector
from lsec_sdk.models.lstm_classifier import SignLSTMClassifier
from lsec_sdk.models.dataset import SignSequenceDataset
from lsec_sdk.models.trainer import SignModelTrainer
from lsec_sdk.models.inference import LiveInferenceEngine

# 17 Señas Oficiales Ecuatorianas CONADIS (Sin Hola ni Chao de España)
CONVERSATIONAL_SIGNS = [
    {"slug": "buenos-dias", "label": "Buenos días", "cat": "frases-comunes"},
    {"slug": "buenas-tardes", "label": "Buenas tardes", "cat": "frases-comunes"},
    {"slug": "buenas-noches", "label": "Buenas noches", "cat": "frases-comunes"},
    {"slug": "como-esta", "label": "¿Cómo está?", "cat": "alfa-c"},
    {"slug": "como-te-sientes", "label": "CÓMO TE SIENTES", "cat": "alfa-c", "alt_video": "023 CÓMO TE  SIENTES.mp4"},
    {"slug": "esta-bien", "label": "¿Está bien?", "cat": "alfa-e"},
    {"slug": "de-nada", "label": "De nada", "cat": "alfa-d"},
    {"slug": "un-momento-por-favor", "label": "Un momento por favor", "cat": "alfa-u"},
    {"slug": "adios", "label": "Adiós", "cat": "configuracion_manual_1"},
    {"slug": "hace-tiempo-que-no-te-veo", "label": "Hace tiempo que no te veo", "cat": "alfa-h"},
    {"slug": "hambre", "label": "Hambre", "cat": "alfa-h"},
    {"slug": "agua", "label": "Agua", "cat": "configuracion_manual_3"},
    {"slug": "ya-no-mas", "label": "Ya no más", "cat": "alfa-y"},
    {"slug": "mal", "label": "Mal", "cat": "alfa-m"},
    {"slug": "donde-estudio", "label": "DÓNDE ESTUDIÓ", "cat": "alfa-d", "alt_video": "044 DÓNDE ESTUDIÓ.mp4"},
    {"slug": "donde", "label": "¿Dónde?", "cat": "alfa-d"},
    {"slug": "en-que-le-puedo-ayudar", "label": "¿En qué le puedo ayudar?", "cat": "alfa-e"},
]

def extract_all_with_dwpose():
    print("=" * 65)
    print(" 1. PREPARACIÓN Y RECORTE DE SEÑAS CONADIS (CUDA GPU)")
    print("=" * 65)
    out_dir = Path("data/processed/sequences")
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    start_total = time.time()
    total_s = len(CONVERSATIONAL_SIGNS)

    for idx, item in enumerate(CONVERSATIONAL_SIGNS, 1):
        slug = item["slug"]
        label = item["label"]
        cat = item["cat"]
        seq_id = f"dwpose_{slug}"
        npy_path = out_dir / f"{seq_id}.npy"

        # Cargar secuencia canónica
        if not npy_path.exists():
            print(f"[{idx:02d}/{total_s:02d}] ERROR: Archivo no encontrado: {npy_path}")
            continue

        arr = np.load(npy_path)

        # Recortar acolchado pasivo de inicio/fin para enfocar el gesto activo
        wy = np.minimum(arr[:, 4, 1], arr[:, 7, 1])
        active = np.where(wy < 0.82)[0]
        if len(active) > 0:
            s_f = max(0, active[0] - 6)
            e_f = min(len(arr), active[-1] + 7)
            trimmed = arr[s_f:e_f].copy()
        else:
            trimmed = arr.copy()

        # Guardar secuencia activa
        np.save(npy_path, trimmed)

        lh_c = float(np.mean(trimmed[:, 95:116, 2]))
        rh_c = float(np.mean(trimmed[:, 116:137, 2]))
        print(f"[{idx:02d}/{total_s:02d}] '{label:<25}': {len(trimmed):3d} frames (orig {len(arr):3d}) | LH: {lh_c:.2f}, RH: {rh_c:.2f}")

        records.append({
            "sequence_id": seq_id,
            "label": label,
            "num_frames": len(trimmed),
            "fps": 25.0,
            "width": 1280,
            "height": 720,
            "npy_path": str(npy_path.resolve()),
            "pose_mean_conf": float(np.mean(trimmed[:, :25, 2])),
            "face_mean_conf": float(np.mean(trimmed[:, 25:95, 2])),
            "lh_mean_conf": lh_c,
            "rh_mean_conf": rh_c,
        })

    # =========================================================================
    # RECOPILAR SECUENCIAS DE REPOSO (POSTURAS EN REPOSO/ESCRITORIO)
    # =========================================================================
    print("\n>> Registrando secuencias auténticas y sintéticas de REPOSO...")
    reposo_files = sorted(list(out_dir.glob("dwpose_reposo_*.npy")))
    for rp in reposo_files:
        r_arr = np.load(rp)
        records.append({
            "sequence_id": rp.stem,
            "label": "REPOSO",
            "num_frames": len(r_arr),
            "fps": 25.0,
            "width": 1280,
            "height": 720,
            "npy_path": str(rp.resolve()),
            "pose_mean_conf": float(np.mean(r_arr[:, :25, 2])),
            "face_mean_conf": float(np.mean(r_arr[:, 25:95, 2])),
            "lh_mean_conf": float(np.mean(r_arr[:, 95:116, 2])),
            "rh_mean_conf": float(np.mean(r_arr[:, 116:137, 2])),
        })

    print(f">> Total de secuencias de REPOSO integradas al modelo: {len(reposo_files)}")

    # Guardar a metadata.csv
    csv_path = Path("data/processed/metadata.csv")
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    total_time = time.time() - start_total
    print(f">> Metadatos guardados en: {csv_path} ({len(records)} secuencias)")
    return csv_path

def train_conversational_model(csv_path: Path):
    print("\n" + "=" * 65)
    print(" 2. ENTRENAMIENTO BI-LSTM BALANCEADO + ATTENTION (RTX 3060 GPU)")
    print("=" * 65)

    df = pd.read_csv(csv_path)
    sign_rows = df[df["label"] != "REPOSO"]
    reposo_rows = df[df["label"] == "REPOSO"]

    target_per_class = 60
    target_frames = 32
    epochs = 35
    batch_size = 16

    # Muestreo balanceado: 60 muestras por clase
    balanced_samples = []
    for _, row in sign_rows.iterrows():
        lbl = row["label"]
        p = row["npy_path"]
        for _ in range(target_per_class):
            balanced_samples.append((p, lbl))

    reposo_paths = list(reposo_rows["npy_path"])
    for i in range(target_per_class):
        p = reposo_paths[i % len(reposo_paths)]
        balanced_samples.append((p, "REPOSO"))

    dataset = SignSequenceDataset(
        balanced_samples,
        target_frames=target_frames,
        augment=True,
        replicate_factor=1,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f">> Dataset balanceado: {len(dataset)} muestras (18 clases x {target_per_class})")

    model = SignLSTMClassifier(
        input_dim=411,
        hidden_dim=128,
        num_layers=2,
        num_classes=dataset.label_encoder.num_classes,
        dropout=0.3,
        use_attention=True,
    )

    trainer = SignModelTrainer(model, learning_rate=1e-3)

    print("\nOptimizando pesos en GPU...")
    for ep in range(1, epochs + 1):
        loss, acc = trainer.train_epoch(loader)
        if ep % 5 == 0 or ep == epochs:
            print(f"  * Época [{ep:02d}/{epochs:02d}] -> Pérdida: {loss:.4f} | Precisión: {acc * 100:.1f}%")

    out_ckpt = Path("models/checkpoints/lsec_classifier.pt")
    extra_config = {
        "input_dim": 411,
        "hidden_dim": 128,
        "num_layers": 2,
        "use_attention": True,
        "target_frames": target_frames,
    }
    trainer.save_checkpoint(out_ckpt, dataset.label_encoder, extra_config)
    print(f"\n>> Checkpoint guardado exitosamente en: {out_ckpt.resolve()}")

def verify_all_signs(csv_path: Path):
    print("\n" + "=" * 65)
    print(" 3. VALIDACIÓN DE COBERTURA Y REPOSO SILENCIOSO")
    print("=" * 65)
    ckpt = Path("models/checkpoints/lsec_classifier.pt")
    engine = LiveInferenceEngine(ckpt, window_size=32)

    df = pd.read_csv(csv_path)

    # 1. Validar señas activas
    perfect = 0
    total = 0
    for _, row in df.drop_duplicates(subset=["label"]).iterrows():
        lbl = row["label"]
        p = Path(row["npy_path"])
        if not p.exists() or lbl == "REPOSO":
            continue
        total += 1
        seq = np.load(p)
        engine.reset()
        max_t_conf = 0.0
        detected_p = None
        max_c = 0.0
        for f in seq:
            pred, c = engine.push_frame(f)
            if pred is not None:
                detected_p = pred
                max_c = max(max_c, c)
            tc = engine.get_target_confidence(lbl)
            if tc > max_t_conf:
                max_t_conf = tc

        ok = (max_t_conf >= 0.70) or (detected_p == lbl)
        if ok:
            perfect += 1
        status = "[OK]" if ok else "[FAIL]"
        print(f"  {status} {lbl:<25} | MaxTarget: {max_t_conf*100:5.1f}% | Pred: {str(detected_p):<25} ({max_c*100:5.1f}%)")

    # 2. Validar que las secuencias de REPOSO sean SILENCIOSAS (0 falsos positivos)
    reposo_df = df[df["label"] == "REPOSO"]
    reposo_ok = 0
    for _, row in reposo_df.iterrows():
        p = Path(row["npy_path"])
        if not p.exists():
            continue
        seq = np.load(p)
        engine.reset()
        emitted_any = False
        for f in seq:
            pred, c = engine.push_frame(f)
            if pred is not None and pred != "REPOSO":
                emitted_any = True
                break
        if not emitted_any:
            reposo_ok += 1

    print(f"\n>> Resultado Señas Activas: {perfect}/{total} señas verificadas con éxito.")
    print(f">> Validación Reposo Silencioso: {reposo_ok}/{len(reposo_df)} secuencias de reposo con CERO falsos positivos.")

def main():
    csv_path = extract_all_with_dwpose()
    train_conversational_model(csv_path)
    verify_all_signs(csv_path)
    print("\n" + "=" * 65)
    print(" ¡PACK CONVERSACIONAL ENTRENADO CON DWPOSE Y CONADIS ECUADOR!")
    print("=" * 65)

if __name__ == "__main__":
    main()
