"""Master pipeline processor for CONADIS LSEC Dictionary.

Orchestrates:
1. Video download (yt-dlp with mobile bypass)
2. Compression (H.264 CRF 26, audio stripped)
3. 137 Keypoint tensor extraction (MediaPipe Holistic)
4. Model fine-tuning/training on GPU
5. Excel reporting
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional

# Add project root and scraper dir to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import torch
import yt_dlp

from lsec_sdk.feature_provider import FeatureProvider
from lsec_sdk.models.lstm_classifier import SignLSTMClassifier
from lsec_sdk.models.dataset import SignSequenceDataset
from lsec_sdk.models.trainer import SignModelTrainer
from video_compressor import VideoCompressor

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("MasterProcessor")

CATALOG_PATH = Path("scraper_conadis/master_catalog.json")
VIDEOS_DIR = Path("data/conadis_videos")
PROCESSED_DIR = Path("data/processed")
METADATA_FILE = PROCESSED_DIR / "metadata.csv"
CHECKPOINT_PATH = Path("models/checkpoints/lsec_classifier.pt")


class MasterProcessor:
    def __init__(self, catalog_path: Path = CATALOG_PATH):
        self.catalog_path = Path(catalog_path)
        self.provider = FeatureProvider()
        self.compressor = VideoCompressor()

    def load_catalog(self) -> List[Dict]:
        with open(self.catalog_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_catalog(self, catalog: List[Dict]):
        with open(self.catalog_path, "w", encoding="utf-8") as f:
            json.dump(catalog, f, ensure_ascii=False, indent=2)

    def download_video(self, item: Dict) -> Optional[Path]:
        yt_url = item.get("youtube_url")
        if not yt_url:
            return None

        category = item.get("category", "general")
        slug = item.get("slug")
        cat_dir = VIDEOS_DIR / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        target_file = cat_dir / f"{slug}.mp4"
        if target_file.exists() and target_file.stat().st_size > 5000:
            logger.info(f"Video ya existe: {target_file.name}")
            return target_file

        logger.info(f"Descargando: '{item['title']}' ({item.get('youtube_id')})...")
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "format": "mp4/best",
            "outtmpl": str(target_file),
            "extractor_args": {
                "youtube": {
                    "player_client": ["ios", "android"],
                }
            },
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([yt_url])

            if target_file.exists():
                return target_file
        except Exception as e:
            logger.error(f"Error descargando {item['title']}: {e}")

        return None

    def process_item(self, item: Dict) -> bool:
        title = item.get("title", "")
        slug = item.get("slug", "")
        category = item.get("category", "")
        seq_id = f"conadis_{category}_{slug}"

        # 1. Download
        raw_video = self.download_video(item)
        if not raw_video or not raw_video.exists():
            return False

        orig_size = raw_video.stat().st_size
        item["orig_size_bytes"] = orig_size

        # 2. Compress video (CRF 26)
        comp_video, stats = self.compressor.compress(raw_video, replace_original=True)
        comp_size = comp_video.stat().st_size if comp_video.exists() else orig_size
        item["comp_size_bytes"] = comp_size

        # 3. Extract 137 keypoints with MediaPipe Holistic
        logger.info(f"Extrayendo tensores (T, 137, 3) para '{title}'...")
        try:
            res = self.provider.process_video(comp_video, sequence_id=seq_id, label=title)
            self.provider.save_sequence(res, output_dir=PROCESSED_DIR / "sequences")
            self.provider.save_metadata_csv([res], csv_path=METADATA_FILE, append=True)

            dur = round(res.num_frames / res.fps, 2) if res.fps > 0 else 0.0
            conf = res.metrics.get("mean_conf_overall", res.metrics.get("confidence_overall", 0.0))

            item["duration_sec"] = dur
            item["frames"] = res.num_frames
            item["conf_overall"] = conf
            item["status"] = "extracted"

            logger.info(
                f"Listo '{title}': {res.num_frames} frames ({dur}s), "
                f"calidad: {conf*100:.1f}%, tamaño: {comp_size/1024:.1f} KB"
            )
            return True
        except Exception as e:
            logger.error(f"Error extrayendo '{title}': {e}")
            return False

    def train_model(self, epochs: int = 30):
        """Train or fine-tune model using all extracted sequences."""
        from torch.utils.data import DataLoader
        logger.info("Iniciando entrenamiento del clasificador LSEC BiLSTM...")
        dataset = SignSequenceDataset.from_metadata_csv(METADATA_FILE, target_frames=32, augment=True, replicate_factor=10)
        num_classes = dataset.label_encoder.num_classes
        classes = [dataset.label_encoder.inverse_transform(i) for i in range(num_classes)]
        logger.info(f"Dataset cargado: {len(dataset)} muestras aumentadas, {num_classes} clases únicas: {classes}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SignLSTMClassifier(input_dim=137 * 3, hidden_dim=128, num_classes=num_classes, num_layers=2)

        trainer = SignModelTrainer(model=model, device=device, learning_rate=0.001)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)

        for ep in range(1, epochs + 1):
            loss, acc = trainer.train_epoch(loader)
            if ep % 10 == 0 or ep == epochs:
                logger.info(f"Epoch [{ep}/{epochs}] - Loss: {loss:.4f} - Acc: {acc*100:.1f}%")

        trainer.save_checkpoint(
            CHECKPOINT_PATH,
            dataset.label_encoder,
            extra_config={
                "input_dim": 137 * 3,
                "hidden_dim": 128,
                "num_classes": num_classes,
                "num_layers": 2,
                "epochs": epochs,
                "updated_at": datetime.now().isoformat(),
            },
        )
        logger.info(f"Modelo guardado exitosamente en: {CHECKPOINT_PATH.resolve()}")

    def run(self, max_items: Optional[int] = None, category_filter: Optional[str] = None, train: bool = True):
        catalog = self.load_catalog()

        # Find items to process
        pending = []
        for it in catalog:
            if not it.get("youtube_url"):
                continue
            if category_filter and it.get("category") != category_filter:
                continue
            if it.get("status") in ("trained", "extracted"):
                continue
            pending.append(it)

        logger.info(f"Encontrados {len(pending)} elementos pendientes de procesar.")
        if max_items and max_items > 0:
            pending = pending[:max_items]

        success_count = 0
        for idx, item in enumerate(pending, 1):
            logger.info(f"[{idx}/{len(pending)}] Procesando: {item['title']} ({item['category']})")
            if self.process_item(item):
                item["status"] = "trained"
                success_count += 1
                self.save_catalog(catalog)

            # Checkpoint and update model + Excel every 25 signs during long runs
            if train and success_count > 0 and success_count % 25 == 0:
                logger.info(f"--- HITO ALCANZADO: {success_count} señas procesadas. Entrenando modelo en GPU... ---")
                try:
                    self.train_model(epochs=20)
                    from export_excel_report import create_excel_report, REPORT_PATH
                    create_excel_report(catalog, REPORT_PATH)
                except Exception as e:
                    logger.error(f"Error en checkpoint intermedio: {e}")

            time.sleep(0.5)

        logger.info(f"Procesamiento concluido: {success_count}/{len(pending)} señas extraídas y aprendidas.")

        if train and success_count > 0:
            self.train_model()

        # Export Excel
        from export_excel_report import create_excel_report, REPORT_PATH
        create_excel_report(catalog, REPORT_PATH)
        logger.info(f"Reporte Excel actualizado en: {REPORT_PATH.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=10, help="Número de señas a procesar (0 para procesar todas)")
    parser.add_argument("--all", action="store_true", help="Procesar todo el catálogo completo de CONADIS")
    parser.add_argument("--category", type=str, default=None, help="Filtrar por categoría CONADIS")
    parser.add_argument("--no-train", action="store_true", help="Omitir reentrenamiento del modelo")
    args = parser.parse_args()

    processor = MasterProcessor()
    limit = None if args.all or args.batch_size == 0 else args.batch_size
    processor.run(max_items=limit, category_filter=args.category, train=not args.no_train)
