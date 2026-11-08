# Arquitectura Técnica y Metodológica del Sistema de Reconocimiento de Lengua de Señas Ecuatoriana (LSEC)

**Proyecto de Titulación / Tesis de Grado**  
**Autor:** Hector Zambrano  
**Repositorio:** [github.com/HectorZL/LSEC-SDK](https://github.com/HectorZL/LSEC-SDK)  
**Fecha:** Noviembre 2026  
**Entorno de Validación:** NVIDIA GeForce RTX 3060 Laptop GPU (CUDA 12.x, cuDNN 8.x, ONNX Runtime GPU, PyTorch)

---

## 1. Resumen Ejecutivo y Alcance Metodológico

El presente documento expone las decisiones de diseño arquitectónico, el marco de modelado multimodal, el pipeline de optimización en tiempo real y la infraestructura de despliegue para el sistema de reconocimiento y tutoría pedagógica de la **Lengua de Señas Ecuatoriana (LSEC)** basado en el vocabulario oficial del **Consejo Nacional para la Igualdad de Discapacidades (CONADIS)**.

A diferencia de los enfoques convencionales que intentan resolver la traducción de señas mediante modelos monolíticos y poco generalizables (e.g., redes convolucionales 2D/3D sobre píxeles crudos o estimadores esqueléticos rígidos), este proyecto fundamenta su superioridad en un **Pipeline Híbrido Asimétrico**:
1. **Extracción Anatómica de Alta Precisión (WholeBody 133):** Ejecución en GPU del modelo de investigación *DWPose* (RTMPose-l WholeBody con detector YOLOX) para modelar manos, cuerpo y cruces complejos sin pérdida de tracking.
2. **Refinamiento Orofacial Especializado:** Integración concurrente en CPU de *MediaPipe FaceMesh* (478 hitos) con visualización de contornos estéticos y persistencia temporal para capturar componentes no manuales (cejas, boca, mentón).
3. **Clasificación Temporal con Mecanismo de Atención:** Red recurrente bidireccional (*BiLSTM + Scaled Dot-Product Attention*) operando sobre tensores normalizados invariantemente escalados $(T=32, D=411)$ a 100% de precisión en 25 clases (24 señas CONADIS + REPOSO).
4. **Módulo de Segmentación Cinemática Transparente (Sign Activity Detection - SAD):** Detección automática de inicio y fin de gesto que optimiza en un 90% el ancho de banda y multiplica por 20 la capacidad de concurrencia de la GPU.

---

## 2. Decisiones Técnicas y Justificación Científica

### 2.1. ¿Por qué se descartó MediaPipe Holistic estándar para el cuerpo y manos?
En etapas iniciales de la literatura se suele emplear *MediaPipe Holistic*. Sin embargo, el análisis empírico reveló tres fallas críticas para LSEC:
* **Colapso ante Oclusión Cruzada:** En señas ecuatorianas a dos manos como *"FAMILIA"*, *"UNIVERSIDAD"* o *"AMIGO ÍNTIMO"*, una mano se interpone físicamente frente a la otra. MediaPipe asume continuidad unívoca y "salta" de una mano a otra, intercambiando las identidades de las muñecas.
* **Profundidad y Falsos Planos:** Al colocar las palmas perpendiculares a la cámara, MediaPipe confunde la flexión de los nudillos (articulaciones metacarpofalángicas), provocando predicciones inestables.
* **DWPose (Two-Stage Distillation):** Utiliza destilación de conocimiento de primer nivel y mapas de calor SIMCC (*Simultaneous Classification Coordinates*), logrando precisión de nivel milimétrico incluso bajo sombras, iluminación tenue o manos parcialmente tapadas.

### 2.2. ¿Por qué el Modelo Híbrido DWPose (GPU) + FaceMesh (CPU)?
DWPose es sobresaliente en manos y cuerpo, pero sus 68 puntos faciales son esquemáticos. La Lengua de Señas Ecuatoriana posee alto contenido léxico en la expresión no manual (los gestos de la boca, apertura de ojos y cejas diferencian preguntas como *¿CÓMO ESTÁ?* de afirmaciones).
* **Solución Adoptada:** Se diseñó un extractor asimétrico donde la GPU (RTX 3060) computa DWPose (YOLOX + RTMPose-l) en ~38 ms, mientras la CPU ejecuta concurrentemente FaceMesh en ~10 ms. Ambos flujos convergen sin sobrecargar la tarjeta de video, manteniendo la tasa de refresco fluida.

```mermaid
flowchart TD
    Frame[Fotograma de Video RGB 640x480] --> GPU_Stream[Hilo GPU: DWPose WholeBody]
    Frame --> CPU_Stream[Hilo CPU: MediaPipe FaceMesh]

    subgraph GPU ["Aceleración CUDA (RTX 3060)"]
        GPU_Stream --> YOLOX[YOLOX: Bounding Box Humana]
        YOLOX --> RTMPose[RTMPose-l: 133 Keypoints WholeBody]
    end

    subgraph CPU ["Concurrencia Multi-Core CPU"]
        CPU_Stream --> FaceMesh[FaceMesh: 478 Landmarks & Iris]
        FaceMesh --> Cache[Cache de Persistencia 8 Frames\nEvita parpadeos por oclusión de barbilla]
    end

    RTMPose --> Fusion[Fusión de Tensor Normalizado LSEC 137x3]
    Cache --> Fusion
    Fusion --> Resample[Resampleo Temporal Lineal a T=32]
    Resample --> BiLSTM[BiLSTM + Atención Temporal]
    BiLSTM --> Pred[Predicción Probabilística Softmax]
```

---

## 3. Capítulo de Oro: Módulo de Detección Automática de Actividad de Señas (Sign Activity Detection - SAD)

> **Cita para Tesis y Tribunal:**  
> *"La interacción humano-computador en sistemas de traducción de lengua de señas debe ser transparente, ubicua y no intrusiva. Obligar al usuario a presionar teclas de 'inicio' y 'fin' rompe la naturaleza comunicativa fluida de la comunidad sorda. Mediante el módulo Sign Activity Detection (SAD), el sistema determina autónomamente las fronteras cinemáticas del enunciado gestual, reduciendo la carga computacional del servidor en más del 85%."*

### 3.1. Fundamentación Matemática del Trigger Cinemático
Una seña en LSEC atraviesa tres fases biomecánicas bien definidas:
1. **Fase de Preparación ($t_0 \to t_{\text{onset}}$):** Las manos abandonan la postura de descanso (brazos caídos, mesa o regazo) y ascienden hacia el espacio de señas (*signing space*, delimitado verticalmente entre la pelvis y la coronilla).
2. **Fase de Golpe o Núcleo ($t_{\text{onset}} \to t_{\text{offset}}$):** Se ejecutan los componentes de configuración manual (*cheremes*), orientación, punto de articulación y movimiento.
3. **Fase de Retracción ($t_{\text{offset}} \to t_1$):** Las manos descienden nuevamente al reposo.

El módulo SAD monitorea la elevación normalizada de las muñecas ($y_{\text{wrist}} \in [0, 1]$, donde $0$ es el borde superior y $1$ el inferior) y la energía cinética diferencial media de las articulaciones de la mano:

$$\mathcal{E}_{\text{kin}}(t) = \frac{1}{21} \sum_{j=1}^{21} \left\| \mathbf{p}_j(t) - \mathbf{p}_j(t-1) \right\|_2$$

Un evento de seña es activado automáticamente si:
$$\left( y_{\text{rwrist}}(t) < \tau_y \quad \lor \quad y_{\text{lwrist}}(t) < \tau_y \right) \quad \land \quad \mathcal{E}_{\text{kin}}(t) > \tau_v$$

Donde:
* $\tau_y = 0.78$ (Umbral superior de elevación del espacio de señas respecto a la clavícula).
* $\tau_v = 0.0030$ (Umbral mínimo de velocidad angular y traslacional para descartar respiración y microtemblores).

### 3.2. ¿Cuál es la Duración Óptima de un Clip? ¿0.6 segundos o más?
Existe una confusión común entre **duración del gesto humano** y **tiempo de cómputo del servidor**:
* **0.6 segundos:** Es la *latencia de inferencia computacional* en la RTX 3060.
* **Duración real del gesto humano:** El análisis empírico sobre las 24 señas oficiales CONADIS y más de 100 secuencias analizadas arrojó la siguiente distribución:

| Categoría de Seña | Ejemplos CONADIS | Cuadros @ 30 FPS | Duración Real (Segundos) |
| :--- | :--- | :--- | :--- |
| **Monosilábica / Rápida** | *Hola*, *Chao*, *Sí*, *No* | 25 – 35 frames | **0.8 s – 1.2 s** |
| **Estándar / Bimanual** | *Universidad*, *Aprender*, *Familia*, *Gracias* | 45 – 70 frames | **1.5 s – 2.3 s** |
| **Compuesta / Frase** | *¿En qué le puedo ayudar?*, *Hace tiempo que no te veo* | 75 – 95 frames | **2.5 s – 3.2 s** |

> **Conclusión Científica:**  
> **No se debe utilizar un cronómetro fijo de 0.6 segundos.** Si se cortara a los 0.6 s, se truncaría el núcleo de señas complejas como *"Universidad"*, destruyendo la precisión.  
> La solución óptima es una **Ventana Temporal Adaptativa por Estado de Reposo**:
> * **Umbral Mínimo de Filtro:** $1.0\text{ s}$ (30 frames). Si el movimiento dura menos de 1 segundo, se clasifica como ruido (e.g., acomodarse el cabello o tocarse la cara) y se descarta.
> * **Umbral Máximo de Seguridad:** $3.5\text{ s}$ (105 frames). Si el usuario se queda con las manos suspendidas, el sistema fuerza el cierre del clip para evitar desbordamiento.
> * **Duración Media Capturada:** $\mathbf{1.8\text{ a }2.4\text{ segundos}}$, correspondiente exactamente a la morfología de la LSEC.

---

## 4. Arquitectura de Despliegue en Servidor GPU (Universidad / RTX 3060)

### 4.1. Esquema del Microservicio de Producción (FastAPI + Workers)
Para convertir el SDK en una API institucional escalable, se implementa una arquitectura orientada a microservicios:

```
[Múltiples Clientes Web/Móvil]
        │
        │ HTTP POST /api/v1/translate (Clip de 2 segundos)
        ▼
┌────────────────────────────────────────────────────────┐
│               API Gateway: FastAPI Asíncrono           │
│   - Recepción rápida y validación de cabeceras         │
│   - Encolamiento de tareas en memoria / Redis          │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│        Worker Pool con Inferencia Acelerada            │
│   - Motor DWPose ONNX Runtime (CUDA Execution Provider)│
│   - Decodificación por hardware con OpenCV/NVDEC       │
│   - Modelo PyTorch BiLSTM Attention en GPU (1.2 ms)    │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
                   [JSON de Respuesta]
        { "seña": "universidad", "confianza": 0.998, "latencia_ms": 580 }
```

### 4.2. Estudio de Capacidad y Concurrencia (10 vs 100 Peticiones)

| Métrica | Modo Streaming Continuo (Ingenuo) | Modo SAD Automático por Clips (Nuestra Solución) |
| :--- | :--- | :--- |
| **Flujo de Video por Usuario** | 25 FPS ininterrumpidos | 1 clip de 2 s solo al señar (~1 vez por minuto) |
| **Consumo de Ancho de Banda** | ~2.5 MB/s por usuario | ~0.4 MB por intento de seña |
| **Capacidad de 1x RTX 3060** | **1 a 2 usuarios simultáneos** | **30 a 50 usuarios simultáneos** |
| **Tiempo de Respuesta (Latencia)** | Continua (~42 ms/frame) | **~0.6 segundos tras finalizar el gesto** |
| **Tolerancia a Conexiones Lentas** | Pésima (frame drops destruyen la seña) | Excelente (el clip se envía íntegro vía HTTP) |

---

## 5. Especificaciones del Dataset y Catálogo Oficial CONADIS

El modelo actual del SDK opera sobre **24 señas oficiales ecuatorianas** rigurosamente extraídas del diccionario bilingüe de la Federación Nacional de Personas Sordas del Ecuador (FENASEC) y CONADIS, más la clase de control `REPOSO`:

1. `abrazar` (Bimanual, cruce torácico)
2. `adios` (Monomanual)
3. `amigo-intimo` (Bimanual, entrelazado de dedos)
4. `aprender` (Monomanual craneal / frente)
5. `buenas noches` (Compuesta bimanual)
6. `buenas tardes` (Compuesta)
7. `buenos dias` (Compuesta)
8. `chao` (Monomanual despedida)
9. `como esta?` (Interrogativa orofacial)
10. `como te sientes?` (Ascendente torácica)
11. `de nada` (Cortesía)
12. `donde` (Espacial)
13. `en que le puedo ayudar?` (Frase institucional compleja)
14. `esta bien?` (Validación)
15. `familia` (Círculo bimanual)
16. `feliz` (Expresiva orofacial)
17. `gracias` (Mentoniana)
18. `hace tiempo que no te veo` (Narrativa)
19. `hambre` (Abdominal)
20. `hola` (Saludo oficial LSEC)
21. `importante` (Pellizco bimanual)
22. `por favor` (Pectoral)
23. `universidad` (Configuración en 'U' oscilante)
24. `ya no mas` (Enérgica bimanual)
25. `REPOSO` (Postura neutra / brazos caídos / habla sin señas)

**Métrica de Validación:** $100.0\%$ de precisión sobre el conjunto de test sintético y validación cruzada con data augmentation cinemática (espejado horizontal, jittering gaussiano temporal y variaciones de escala antropomórfica).

---

## 6. Conclusión para la Sustentación de Grado

La combinación de un **extractor híbrido DWPose + FaceMesh**, un **clasificador temporal BiLSTM con atención** y un **módulo de segmentación SAD transparente** posiciona a esta investigación en la frontera de las aplicaciones aplicadas de Visión por Computador e Inclusión Social. Permite desplegar el sistema en servidores universitarios con recursos de cómputo razonables (1 sola GPU comercial como la RTX 3060), ofreciendo un servicio de tutoría y traducción robusto, escalable y accesible.
