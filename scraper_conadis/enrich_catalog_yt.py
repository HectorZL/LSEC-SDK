"""Enrich master_catalog.json by resolving YouTube video IDs for all entries in parallel."""

import concurrent.futures
from dataclasses import asdict
import json
import logging
from pathlib import Path
import re
import time
import urllib.request
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("EnrichCatalog")

CATALOG_PATH = Path("scraper_conadis/master_catalog.json")
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def fetch_youtube_id(item: dict) -> dict:
    if item.get("youtube_id") and item.get("youtube_url"):
        return item

    url = item.get("article_url")
    if not url:
        return item

    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        # Extract YouTube embed
        yt_match = re.findall(r"youtube\.com/embed/([a-zA-Z0-9_-]+)", html)
        if yt_match:
            yt_id = yt_match[0]
            item["youtube_id"] = yt_id
            item["youtube_url"] = f"https://www.youtube.com/watch?v={yt_id}"
        else:
            # Check iframe watch url
            watch_match = re.findall(r"youtube\.com/watch\?v=([a-zA-Z0-9_-]+)", html)
            if watch_match:
                yt_id = watch_match[0]
                item["youtube_id"] = yt_id
                item["youtube_url"] = f"https://www.youtube.com/watch?v={yt_id}"
    except Exception as e:
        logger.debug(f"Error fetching {url}: {e}")

    return item


def main():
    if not CATALOG_PATH.exists():
        logger.error(f"No se encuentra {CATALOG_PATH}")
        return

    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        items: List[dict] = json.load(f)

    logger.info(f"Iniciando resolución de videos YouTube para {len(items)} señas...")
    start_time = time.time()

    completed = 0
    with_yt = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        future_to_item = {executor.submit(fetch_youtube_id, item): item for item in items}

        for future in concurrent.futures.as_completed(future_to_item):
            res = future.result()
            completed += 1
            if res.get("youtube_id"):
                with_yt += 1

            if completed % 100 == 0 or completed == len(items):
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Progreso: {completed}/{len(items)} ({completed/len(items)*100:.1f}%) "
                    f"- Con video YT: {with_yt} - {rate:.1f} items/seg"
                )

    # Save enriched catalog
    with open(CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    elapsed_total = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f" RESOLUCIÓN COMPLETADA en {elapsed_total:.1f} segundos.")
    logger.info(f" Total señas con video YouTube: {with_yt}/{len(items)}")
    logger.info(f" Catálogo actualizado guardado en: {CATALOG_PATH.resolve()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

