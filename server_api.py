"""
Servidor Microservicio FastAPI para Reconocimiento de Lengua de Señas Ecuatoriana (LSEC).
Despliegue de producción con soporte para:
1. Recepción y traducción de clips de video automáticos (SAD).
2. Inferencia temporal ultrarrápida (BiLSTM con Mecanismo de Atención).
3. Transmisión interactiva vía WebSockets con segmentación transparente.
"""

import os
import sys
import time
import tempfile
import cv2
import numpy as np
import torch
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Asegurar importación del SDK
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from lsec_sdk.dwpose_extractor import DWPoseDetector
from lsec_sdk.models.inference import LiveInferenceEngine, resample_sequence
from lsec_sdk.activity_detector import SignActivityDetector, ActivityState

app = FastAPI(
    title="LSEC SDK - API Institucional de Traducción de Señas",
    description="Microservicio de Inteligencia Artificial para el reconocimiento de LSEC (CONADIS Ecuador) acelerado por GPU.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Estado global del servidor (Lazy load para evitar demoras de arranque innecesarias)
detector: Optional[DWPoseDetector] = None
engine: Optional[LiveInferenceEngine] = None


def get_inference_engine() -> LiveInferenceEngine:
    global engine
    if engine is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt_path = os.path.join("models", "checkpoints", "lsec_classifier.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join("models", "checkpoints", "bilstm_attention_dwpose_best.pt")
        print(f"[API] Cargando LiveInferenceEngine en dispositivo: {device} desde {ckpt_path}...")
        engine = LiveInferenceEngine(checkpoint_path=ckpt_path, device=device)
    return engine


def get_detector() -> DWPoseDetector:
    global detector
    if detector is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[API] Inicializando DWPoseDetector en: {device}...")
        detector = DWPoseDetector(device=device, enable_facemesh=True)
    return get_detector_instance()

def get_detector_instance() -> DWPoseDetector:
    global detector
    if detector is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        detector = DWPoseDetector(device=device, enable_facemesh=True)
    return detector


class TensorPayload(BaseModel):
    keypoints: List[List[List[float]]]  # Matriz (T, 137, 3)


@app.on_event("startup")
def startup_event():
    print("[API] Iniciando microservicio LSEC...")
    # Precargar el motor de clasificación
    get_inference_engine()


@app.get("/")
def health_check():
    eng = get_inference_engine()
    labels = list(eng.label_encoder.idx_to_label.values())
    return {
        "status": "online",
        "service": "LSEC Recognition Microservice (CONADIS Ecuador)",
        "device": str(eng.device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "total_classes": len(labels),
        "vocabulary": labels,
    }


@app.post("/api/v1/translate_video")
async def translate_video(file: UploadFile = File(...)):
    """
    Recibe un clip de video (2 a 3 segundos) capturado manual o automáticamente por SAD.
    Ejecuta el pipeline completo DWPose GPU + BiLSTM y retorna la seña en JSON.
    """
    t0 = time.time()
    suffix = os.path.splitext(file.filename)[1] or ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if len(frames) < 12:
            raise HTTPException(status_code=400, detail="El video contiene muy pocos fotogramas (<12 frames).")

        det = get_detector_instance()
        eng = get_inference_engine()

        # Extraer tensores normalizados (137, 3)
        t_extract_start = time.time()
        kp_sequence = []
        for fr in frames:
            dw_res = det.process(fr)
            tensor = det.extract_lsec_tensor(fr, precomputed_res=dw_res)
            kp_sequence.append(tensor)
        t_extract_end = time.time()

        # Inferencia con la red BiLSTM
        seq_arr = np.array(kp_sequence, dtype=np.float32)
        resampled = resample_sequence(seq_arr, eng.window_size)
        inp = torch.from_numpy(resampled).unsqueeze(0).to(eng.device)

        with torch.no_grad():
            logits, _ = eng.model(inp)
            probs = torch.softmax(logits, dim=-1)
            conf, pred_idx = probs.max(dim=-1)

        best_label = eng.label_encoder.inverse_transform(int(pred_idx.item()))
        best_conf = float(conf.item())
        total_time_ms = round((time.time() - t0) * 1000, 1)

        return {
            "success": True,
            "prediction": best_label,
            "confidence": round(best_conf, 4),
            "total_frames": len(frames),
            "duration_seconds": round(len(frames) / 30.0, 2),
            "inference_time_ms": total_time_ms,
            "extraction_time_ms": round((t_extract_end - t_extract_start) * 1000, 1),
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/api/v1/translate_tensor")
def translate_tensor(payload: TensorPayload):
    """
    Recibe directamente el tensor (T, 137, 3) extraído por el cliente o SAD.
    Tiempo de respuesta ultrarrápido (<5 ms en GPU / CPU).
    """
    t0 = time.time()
    eng = get_inference_engine()

    seq_arr = np.array(payload.keypoints, dtype=np.float32)
    if len(seq_arr) < 12:
        raise HTTPException(status_code=400, detail="Secuencia demasiado corta (<12 frames).")

    resampled = resample_sequence(seq_arr, eng.window_size)
    inp = torch.from_numpy(resampled).unsqueeze(0).to(eng.device)

    with torch.no_grad():
        logits, _ = eng.model(inp)
        probs = torch.softmax(logits, dim=-1)
        conf, pred_idx = probs.max(dim=-1)

    best_label = eng.label_encoder.inverse_transform(int(pred_idx.item()))
    best_conf = float(conf.item())

    return {
        "success": True,
        "prediction": best_label,
        "confidence": round(best_conf, 4),
        "frames_received": len(seq_arr),
        "inference_time_ms": round((time.time() - t0) * 1000, 2),
    }


@app.websocket("/api/v1/stream_ws")
async def websocket_stream(websocket: WebSocket):
    """
    Endpoint WebSocket interactivo en tiempo real con Detección Automática de Señas (SAD).
    El cliente envía coordenadas frame a frame; el servidor detecta el inicio y fin
    de forma transparente, retornando la traducción en cuanto el usuario descansa las manos.
    """
    await websocket.accept()
    eng = get_inference_engine()
    sad = SignActivityDetector(fps=30.0)

    try:
        while True:
            data = await websocket.receive_json()
            if "keypoints" not in data:
                continue

            kp = np.array(data["keypoints"], dtype=np.float32)
            state, payload = sad.process_frame(kp)

            if state == ActivityState.ACTIVE:
                await websocket.send_json({"event": "SIGN_RECORDING", "state": "ACTIVE"})

            elif state == ActivityState.COMPLETED and payload is not None:
                # Seña finalizada de forma transparente -> clasificarla de inmediato
                t0 = time.time()
                seq = payload["keypoints_sequence"]
                resampled = resample_sequence(seq, eng.window_size)
                inp = torch.from_numpy(resampled).unsqueeze(0).to(eng.device)

                with torch.no_grad():
                    logits, _ = eng.model(inp)
                    probs = torch.softmax(logits, dim=-1)
                    conf, pred_idx = probs.max(dim=-1)

                best_label = eng.label_encoder.inverse_transform(int(pred_idx.item()))
                best_conf = float(conf.item())

                await websocket.send_json({
                    "event": "SIGN_DETECTED",
                    "prediction": best_label,
                    "confidence": round(best_conf, 4),
                    "duration_sec": payload["duration_sec"],
                    "frames_count": payload["frames_count"],
                    "inference_time_ms": round((time.time() - t0) * 1000, 2),
                })
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_api:app", host="0.0.0.0", port=8000, reload=False)
