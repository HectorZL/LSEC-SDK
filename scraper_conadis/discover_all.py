"""Discovery script: Crawls all categories of CONADIS dictionary to build master_catalog.json."""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import re
import time
from typing import Dict, List, Set

from scrapling import Fetcher

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("ConadisDiscovery")

BASE_URL = "http://www.plataformaconadis.gob.ec/~platafor/diccionario"


@dataclass
class MasterItem:
    title: str
    slug: str
    category: str
    article_url: str
    youtube_id: str = ""
    youtube_url: str = ""
    status: str = "pending"  # pending, downloaded, extracted, trained
    orig_size_bytes: int = 0
    comp_size_bytes: int = 0
    duration_sec: float = 0.0
    frames: int = 0
    conf_overall: float = 0.0


def discover_all_categories() -> List[str]:
    fetcher = Fetcher()
    # Explore alphabet index to find all subcategories
    root_url = f"{BASE_URL}/?article_category=alfabetico"
    logger.info(f"Explorando categorías desde: {root_url}")

    page = fetcher.get(root_url)
    links = page.css("a::attr(href)")

    categories: Set[str] = set()
    for h in links:
        match = re.search(r"article_category=([a-zA-Z0-9_\-]+)", str(h))
        if match:
            slug = match.group(1)
            if slug != "alfabetico":  # Skip root meta category
                categories.add(slug)

    # Always ensure frases-comunes and primary ones are included
    priority_categories = [
        "frases-comunes",
        "dias-de-la-semana",
        "meses-del-ano",
        "numeros",
        "ciudades-ecuador",
        "provincias-ecuador",
        "instituciones",
        "animales",
        "frutas",
    ]
    for p in priority_categories:
        categories.add(p)

    sorted_cats = sorted(list(categories))
    logger.info(f"Total categorías detectadas en CONADIS: {len(sorted_cats)}")
    return sorted_cats


def main():
    catalog_file = Path("scraper_conadis/master_catalog.json")
    catalog_file.parent.mkdir(parents=True, exist_ok=True)

    existing_items: Dict[str, MasterItem] = {}
    if catalog_file.exists():
        try:
            with open(catalog_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for d in data:
                    item = MasterItem(**d)
                    existing_items[item.slug] = item
            logger.info(f"Cargados {len(existing_items)} términos existentes del catálogo.")
        except Exception as e:
            logger.warning(f"No se pudo leer catálogo previo: {e}")

    categories = discover_all_categories()
    fetcher = Fetcher()

    total_discovered = 0
    new_discovered = 0

    for cat_idx, cat in enumerate(categories, 1):
        cat_url = f"{BASE_URL}/?article_category={cat}"
        logger.info(f"[{cat_idx}/{len(categories)}] Escaneando categoría: '{cat}'")

        try:
            page = fetcher.get(cat_url)
            if page.status != 200:
                continue

            # Pagination
            paged_links = page.css("a::attr(href)")
            paged_urls = set()
            for h in paged_links:
                if f"article_category={cat}" in str(h) and "paged=" in str(h):
                    paged_urls.add(str(h).replace("&#038;", "&"))

            all_cat_pages = [cat_url] + sorted(list(paged_urls))

            for p_url in all_cat_pages:
                p_page = fetcher.get(p_url) if p_url != cat_url else page
                a_tags = p_page.css("a")

                for a in a_tags:
                    href = a.attrib.get("href", "")
                    if "st_kb=" in href:
                        raw_title = a.text.strip()
                        clean_title = re.sub(r"<[^>]+>", "", raw_title).strip()
                        slug = href.split("st_kb=")[-1].split("&")[0]

                        if clean_title and slug:
                            if slug not in existing_items:
                                existing_items[slug] = MasterItem(
                                    title=clean_title,
                                    slug=slug,
                                    category=cat,
                                    article_url=href,
                                )
                                new_discovered += 1
                            total_discovered += 1

            time.sleep(0.3)
        except Exception as e:
            logger.error(f"Error escaneando categoría '{cat}': {e}")

    # Save master catalog
    catalog_list = [asdict(item) for item in existing_items.values()]
    with open(catalog_file, "w", encoding="utf-8") as f:
        json.dump(catalog_list, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info(f" ESCANEO COMPLETADO: {len(existing_items)} términos únicos encontrados.")
    logger.info(f" Nuevos descubiertos en esta pasada: {new_discovered}")
    logger.info(f" Guardado en: {catalog_file.resolve()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

