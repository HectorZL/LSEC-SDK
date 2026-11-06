#!/usr/bin/env python3
"""
CLI runner for the CONADIS LSEC Dictionary Scraper & Pipeline.

Usage:
  # Descargar las primeras 5 frases comunes y extraer características:
  python scraper_conadis/run_scraper.py --category frases-comunes --limit 5 --extract

  # Descargar días de la semana:
  python scraper_conadis/run_scraper.py --category dias-de-la-semana --extract
"""

import argparse
from pathlib import Path
import sys

# Ensure src in path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from conadis_crawler import ConadisCrawler
from video_downloader import ConadisVideoDownloader
from lsec_sdk import FeatureProvider


def main():
    parser = argparse.ArgumentParser(description="CONADIS LSEC Dictionary Scraper & Feature Pipeline")
    parser.add_argument(
        "--category",
        type=str,
        default="frases-comunes",
        help="Category slug to crawl (e.g. 'frases-comunes', 'dias-de-la-semana', 'numeros')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of videos to download and process (None for all)",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Automatically extract MediaPipe keypoints into .npy and update metadata.csv",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Polite delay between requests in seconds",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(" SCRAPER CONADIS - DICCIONARIO OFICIAL LSEC")
    print("=" * 60)
    print(f"Categoría objetivo:  {args.category}")
    print(f"Límite de términos:  {args.limit}")
    print(f"Auto-extracción SDK: {args.extract}")
    print("=" * 60)

    # 1. Crawl category
    crawler = ConadisCrawler(delay_seconds=args.delay)
    entries = crawler.crawl_category(args.category, max_articles=args.limit)

    if not entries:
        print("No se encontraron artículos en la categoría seleccionada.")
        return

    catalog_path = Path("scraper_conadis") / f"catalog_{args.category}.json"
    crawler.save_catalog(entries, catalog_path)

    # 2. Download videos
    downloader = ConadisVideoDownloader()
    print("\nIniciando descarga de videos...")
    downloader.download_batch(entries, delay_seconds=args.delay)

    # 3. Optional: Feature extraction into LSEC SDK
    if args.extract:
        print("\n" + "=" * 60)
        print(" INICIANDO EXTRACCIÓN AUTOMÁTICA DE CARACTERÍSTICAS (SDK)")
        print("=" * 60)
        provider = FeatureProvider(confidence_threshold=0.1, normalization="resolution")
        csv_file = Path("data/processed/metadata.csv")
        out_seq_dir = Path("data/processed/sequences")

        processed_count = 0
        for idx, entry in enumerate(entries, 1):
            if not entry.video_path or not Path(entry.video_path).exists():
                continue

            video_p = Path(entry.video_path)
            seq_id = f"conadis_{entry.category}_{entry.slug}"
            label = entry.title.upper()

            print(f"[{idx:02d}/{len(entries):02d}] Procesando tensor para '{label}'...")
            try:
                res = provider.process_video(video_p, sequence_id=seq_id, label=label)
                npy_p = provider.save_sequence(res, out_seq_dir)
                provider.save_metadata_csv([res], csv_file, append=True)
                print(f"   -> Tensor: {res.tensor.shape} | Guardado: {npy_p.name}")
                processed_count += 1
            except Exception as e:
                print(f"   -> Error procesando '{label}': {e}")

        print(f"\nExtracción finalizada: {processed_count} secuencias agregadas al SDK.")

    print("\n" + "=" * 60)
    print(" PROCESO CONADIS COMPLETADO CON ÉXITO")
    print("=" * 60)


if __name__ == "__main__":
    main()

