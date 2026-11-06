# LSEC SDK - Sign Language Processing, Feature Extraction & Classification

SDK en Python para el procesamiento, normalización, entrenamiento y traducción en tiempo real de Lengua de Señas Ecuatoriana (LSEC) y datasets universales (How2Sign / OpenPose / MediaPipe).

---

## 📁 Estructura del SDK

```text
SDK/
├── lsec_sdk/                     # Módulo principal del SDK
│   ├── feature_provider.py       # Extracción de JSON (How2Sign) y Video MP4 (LSEC)
│   ├── realtime_capture.py       # Captura de webcam con MediaPipe Holistic (~30 FPS)
│   ├── visualizer.py             # Renderizado de esqueletos y animaciones comparativas
│   ├── constants.py              # Definición de keypoints (BODY_25, Face, Hands)
│   └── models/                   # Módulo de Inteligencia Artificial (PyTorch + CUDA)
│       ├── lstm_classifier.py    # Red BiLSTM con Mecanismo de Atención Temporal
│       ├── dataset.py            # Dataset de secuencias, aumento de datos e interpolación
│       ├── trainer.py            # Optimizador y entrenamiento acelerado por GPU
│       └── inference.py          # Motor de inferencia en ventana deslizante en vivo
├── tests/                        # 13 pruebas automatizadas (100% pasando)
├── data/processed/
│   ├── sequences/                # Tensores NumPy (.npy) extraídos
│   ├── metadata.csv              # Base de datos de secuencias y métricas
│   └── visualizations/           # GIFs y fotos de validación
├── models/checkpoints/           # Pesos del modelo entrenado (.pt)
├── record_dataset.py             # Asistente interactivo para grabar nuevas señas
├── train_classifier.py           # Script para entrenar el modelo en tu GPU RTX 3060
├── run_live_translator.py        # Traductor en vivo desde tu webcam
└── run_webcam.py                 # Visualizador y grabador en vivo
```

---

## 🚀 Flujo de Trabajo Rápido

### 1. Grabar nuevas señas con tu cámara
```powershell
python record_dataset.py "HOLA"
```
*(O ejecuta con doble clic el archivo `grabar_senas.bat`)*

### 2. Entrenar el clasificador de IA en tu GPU
```powershell
python train_classifier.py --epochs 25
```

### 3. Abrir el traductor en tiempo real
```powershell
python run_live_translator.py
```
*(O ejecuta con doble clic el archivo `iniciar_traductor.bat`)*
El sistema rastreará tus manos y rostro en vivo y mostrará un banner verde con la seña detectada y el porcentaje de confianza.

### 4. Extraer características de videos existentes
```powershell
python extract_lsec_video.py
```

### 5. Correr la suite de pruebas
```powershell
python -m pytest tests/ -v
```
