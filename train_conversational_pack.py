#!/usr/bin/env python3
"""
Pipeline de Extracción y Entrenamiento del Pack Conversacional LSEC
Utiliza DWPose (Whole-Body GPU con CUDA) para extraer características y
entrena un clasificador BiLSTM con Atención para las 17 señas conversacionales + REPOSO.
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
    print(" 1. EXTRACCIÓN CON DWPOSE WHOLE-BODY (CUDA GPU)")
    print("=" * 65)
    detector = DWPoseDetector()
    records = []
    out_dir = Path("data/processed/sequences")
    out_dir.mkdir(parents=True, exist_ok=True)

    start_total = time.time()
    for idx, item in enumerate(CONVERSATIONAL_SIGNS, 1):
        slug = item["slug"]
        label = item["label"]
        cat = item["cat"]
        seq_id = f"dwpose_{slug}"
        npy_path = out_dir / f"{seq_id}.npy"

        # Determine video source
        v_path = Path("data/conadis_videos") / cat / f"{slug}.mp4"
        if item.get("alt_video") and Path(item["alt_video"]).exists():
            v_path = Path(item["alt_video"])
        elif not v_path.exists():
            # Check root
            root_matches = list(Path(".").glob(f"*{slug}*.mp4"))
            if root_matches:
                v_path = root_matches[0]

        if not v_path.exists():
            print(f"[{idx:02d}/17] ERROR: Video no encontrado para '{label}': {v_path}")
            continue

        cap = cv2.VideoCapture(str(v_path))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        t0 = time.time()
        frames_list = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            tensor_frame = detector.extract_lsec_tensor(frame)
            frames_list.append(tensor_frame)
        cap.release()

        arr = np.array(frames_list, dtype=np.float32)
        np.save(npy_path, arr)
        dt = time.time() - t0
        fps_proc = len(arr) / max(0.01, dt)

        lh_c = float(np.mean(arr[:, 95:116, 2]))
        rh_c = float(np.mean(arr[:, 116:137, 2]))
        print(f"[{idx:02d}/17] '{label:<25}': {len(arr):3d} frames ({dt:4.1f}s, {fps_proc:4.1f} FPS) | LH: {lh_c:.2f}, RH: {rh_c:.2f}")

        records.append({
            "sequence_id": seq_id,
            "label": label,
            "num_frames": len(arr),
            "fps": fps,
            "width": w,
            "height": h,
            "npy_path": str(npy_path.resolve()),
            "pose_mean_conf": float(np.mean(arr[:, :25, 2])),
            "face_mean_conf": float(np.mean(arr[:, 25:95, 2])),
            "lh_mean_conf": lh_c,
            "rh_mean_conf": rh_c,
        })

    # Add REPOSO baseline
    reposo_path = out_dir / "dwpose_reposo_baseline.npy"
    if records:
        first_tensor = np.load(records[0]["npy_path"])
        reposo_arr = first_tensor[:28].copy()
        reposo_arr[:, 95:, 2] *= 0.1  # hands down
        np.save(reposo_path, reposo_arr)

        records.append({
            "sequence_id": "dwpose_reposo_baseline",
            "label": "REPOSO",
            "num_frames": len(reposo_arr),
            "fps": 25.0,
            "width": 1280,
            "height": 720,
            "npy_path": str(reposo_path.resolve()),
            "pose_mean_conf": float(np.mean(reposo_arr[:, :25, 2])),
            "face_mean_conf": float(np.mean(reposo_arr[:, 25:95, 2])),
            "lh_mean_conf": float(np.mean(reposo_arr[:, 95:116, 2])),
            "rh_mean_conf": float(np.mean(reposo_arr[:, 116:137, 2])),
        })
        print(f"[18/18] Baseline 'REPOSO' configurado con éxito.")

    # Save to data/processed/metadata.csv
    csv_path = Path("data/processed/metadata.csv")
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    total_time = time.time() - start_total
    print(f"\n>> Extracción completada en {total_time:.1f}s. Metadatos guardados en: {csv_path}")
    return csv_path

def train_conversational_model(csv_path: Path):
    print("\n" + "=" * 65)
    print(" 2. ENTRENAMIENTO BI-LSTM + ATTENTION (RTX 3060 GPU)")
    print("=" * 65)

    df = pd.read_csv(csv_path)
    labeled_df = df[df["label"].notna() & (df["label"].str.strip() != "")]
    samples = [(row["npy_path"], row["label"]) for _, row in labeled_df.iterrows()]
    labels = [lbl for _, lbl in samples]
    unique_labels = sorted(list(set(labels)))
    print(f">> Clases únicas a entrenar ({len(unique_labels)}): {unique_labels}")

    target_frames = 32
    replicate_factor = 35
    batch_size = 16
    epochs = 35

    dataset = SignSequenceDataset(
        samples,
        target_frames=target_frames,
        augment=True,
        replicate_factor=replicate_factor,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f">> Muestras sintéticas generadas con aumento de datos: {len(dataset)}")

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
    print(" 3. VALIDACIÓN DE COBERTURA Y CONFIANZA POR SEÑA")
    print("=" * 65)
    ckpt = Path("models/checkpoints/lsec_classifier.pt")
    engine = LiveInferenceEngine(ckpt, window_size=32)

    df = pd.read_csv(csv_path)
    perfect = 0
    total = 0
    for _, row in df.iterrows():
        lbl = row["label"]
        p = Path(row["npy_path"])
        if not p.exists() or lbl == "REPOSO":
            continue
        total += 1
        seq = np.load(p)
        engine.buffer.clear()
        engine.prediction_history.clear()
        last_p, last_c = None, 0.0
        for f in seq:
            pred, c = engine.push_frame(f)
            if pred is not None:
                last_p, last_c = pred, c
        t_conf = engine.get_target_confidence(lbl)
        ok = (t_conf >= 0.80) or (last_p == lbl)
        if ok:
            perfect += 1
        status = "[OK]" if ok else "[FAIL]"
        print(f"  {status} {lbl:<25} | Pred: {str(last_p):<25} ({last_c*100:5.1f}%) | Objetivo: {t_conf*100:5.1f}%")

    print(f"\n>> Resultado Global: {perfect}/{total} señas verificadas con éxito.")

def main():
    csv_path = extract_all_with_dwpose()
    train_conversational_model(csv_path)
    verify_all_signs(csv_path)
    print("\n" + "=" * 65)
    print(" ¡PACK CONVERSACIONAL ENTRENADO Y LISTO CON DWPOSE!")
    print("=" * 65)

if __name__ == "__main__":
    main()
