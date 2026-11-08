# LSEC SDK - Sign Language Processing, Whole-Body Estimation & Guided Tutor

SDK en Python y PyTorch acelerado por GPU (NVIDIA CUDA) para la extracción de características articulatorias, entrenamiento y traducción interactiva en tiempo real de la **Lengua de Señas Ecuatoriana (LSEC)** basada en el estándar oficial del **CONADIS Ecuador**.

---

## 🌟 Características Principales

1. **Estimación Postural DWPose Whole-Body (CUDA GPU):**
   * Extracción holística en tiempo real de **133 keypoints** (cuerpo COCO, rostro denso de 68 puntos y ambas manos de 21 puntos).
   * Robusto frente a cruce de manos, oclusiones y movimientos rápidos (30-45 FPS en GPU NVIDIA RTX 3060).

2. **Catálogo Canónico Oficial CONADIS Ecuador (17 Señas):**
   * **Saludos y Cortesía (7):** *Buenos días, Buenas tardes, Buenas noches, Adiós, De nada, Un momento por favor, Hace tiempo que no te veo.*
   * **Trabajo y Trámites (2):** *DÓNDE ESTUDIÓ, ¿En qué le puedo ayudar?*
   * **Comida y Rutina (3):** *Hambre, Agua, Ya no más.*
   * **Salud y Estado (4):** *CÓMO TE SIENTES, ¿Cómo está?, ¿Está bien?, Mal.*
   * **General (1):** *¿Dónde?*

3. **Modo Tutor y Aprendizaje Guiado con Modelo de Palitos (PiP):**
   * Animación en bucle del profesor virtual en tiempo real (PiP).
   * Máquina de estados desacoplada: **Demostración** $\rightarrow$ **Cuenta Regresiva** $\rightarrow$ **Práctica activa**.
   * Barra de progreso interactiva en pantalla con requisito de racha sostenida para evitar disparos prematuros.
   * Menú interactivo en pantalla con selección por teclado (`W/S`, `Flechas`, `1-6`, `ENTER`, `TAB`, `M`, `C`).

4. **Compuerta Cinemática Anti Falsos Positivos:**
   * Detección estricta de elevación de muñecas en el espacio de señas ($Y < 0.82$) y velocidad cinética ($\Delta > 0.0030$).
   * Silencio absoluto en reposo (**0 falsos positivos** cuando el usuario está sentado con las manos abajo).

5. **Clasificador BiLSTM + Temporal Attention:**
   * Entrada de 411 dimensiones ($137 \times 3$).
   * BiLSTM de 2 capas con 128 unidades ocultas y mecanismo de atención temporal para ponderar el punto clave articulatorio.
   * 100.0% de cobertura y precisión en validación experimental.

---

## 📁 Estructura del Proyecto

```text
SDK/
├── docs/
│   └── Informe_Tecnico_Tesis_LSEC_DWPose.docx   # Informe técnico formal para la tesis
├── lsec_sdk/                                     # Módulo principal del SDK
│   ├── dwpose_extractor.py                       # Extractor DWPose Whole-Body ONNX/CUDA
│   ├── tutor_engine.py                           # Máquina de estados pedagógica y menú HUD
│   ├── visualizer.py                             # Renderizado de esqueletos y animaciones
│   ├── constants.py                              # Definición de topologías y slices
│   └── models/                                   # Arquitecturas PyTorch
│       ├── lstm_classifier.py                    # Red BiLSTM con Atención Temporal
│       ├── dataset.py                            # Dataset balanceado y aumento temporal
│       ├── trainer.py                            # Optimizador y entrenamiento GPU
│       └── inference.py                          # Inferencia deslizante con filtro cinemático
├── data/
│   ├── conadis_videos/                           # Videos de referencia oficiales CONADIS
│   └── processed/
│       ├── sequences/                            # Tensores NumPy (.npy) extraídos
│       └── metadata.csv                          # Base de datos de secuencias
├── models/checkpoints/
│   └── lsec_classifier.pt                        # Pesos del modelo entrenado
├── run_dwpose_translator.py                      # Traductor y Tutor en vivo GPU (DWPose)
├── train_conversational_pack.py                  # Pipeline de entrenamiento balanceado GPU
├── pyproject.toml                                # Especificación del paquete
└── tests/                                        # Suite de pruebas unitarias
```

---

## 🚀 Inicio Rápido

### 1. Iniciar el Traductor y Tutor en Vivo (DWPose GPU)
```powershell
python run_dwpose_translator.py
```

* **[M]:** Abrir/Cerrar el Catálogo y Menú en pantalla.
* **[W / S o Flechas]:** Navegar entre señas de la lección.
* **[1 - 6]:** Cambiar de módulo temático (Saludos, Trabajo, Comida, Salud, General, Todas).
* **[ENTER o ESPACIO]:** Seleccionar la seña para practicar.
* **[ESPACIO]:** Saltar demostración e ir directo a imitar.
* **[T]:** Alternar entre modo Tutor y modo Traductor Libre.
* **[Q]:** Salir de la aplicación.

### 2. Reentrenar el Modelo Balanceado en GPU
```powershell
python train_conversational_pack.py
```

### 3. Ejecutar las Pruebas Unitarias
```powershell
python -m pytest tests/ -v
```

---

## 📄 Documentación Académica

El informe técnico detallado para la defensa y reuniones de tesis se encuentra disponible en:
* [`docs/Informe_Tecnico_Tesis_LSEC_DWPose.docx`](docs/Informe_Tecnico_Tesis_LSEC_DWPose.docx)
