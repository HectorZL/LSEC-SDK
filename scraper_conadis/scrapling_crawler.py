"""Enhanced CONADIS dictionary crawler powered by Scrapling (D4Vinci/Scrapling)."""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import re
import time
from typing import Dict, List, Optional

from scrapling import Fetcher

logger = logging.getLogger(__name__)

BASE_URL = "http://www.plataformaconadis.gob.ec/~platafor/diccionario"


@dataclass
class ScraplingConadisEntry:
    title: str
    slug: str
    category: str
    article_url: str
    youtube_id: Optional[str] = None
    youtube_url: Optional[str] = None
    video_path: Optional[str] = None
    compressed_video_path: Optional[str] = None


class ScraplingCrawler:
    """High-speed stealth crawler for CONADIS dictionary using Scrapling."""

    def __init__(self, delay_seconds: float = 0.5):
        self.delay = delay_seconds
        self.fetcher = Fetcher()

    def crawl_category(self, category_slug: str, max_articles: Optional[int] = None) -> List[ScraplingConadisEntry]:
        """
        Crawls all articles in a given category using Scrapling CSS/XPath selectors.
        """
        cat_url = f"{BASE_URL}/?article_category={category_slug}"
        print(f"[Scrapling] Explorando categoría: '{category_slug}' -> {cat_url}")

        page = self.fetcher.get(cat_url)
        if page.status != 200:
            print(f"[Scrapling] Error al acceder a la categoría (Status {page.status})")
            return []

        # Find all pagination links
        pagination_links = page.css("a::attr(href)")
        paged_urls = set()
        for h in pagination_links:
            if f"article_category={category_slug}" in str(h) and "paged=" in str(h):
                clean_url = str(h).replace("&#038;", "&")
                paged_urls.add(clean_url)

        all_pages = [cat_url] + sorted(list(paged_urls))
        print(f"[Scrapling] Total de páginas a procesar: {len(all_pages)}")

        entries: List[ScraplingConadisEntry] = []
        seen_slugs = set()

        for page_url in all_pages:
            p = self.fetcher.get(page_url)
            links = p.css("a")

            for a in links:
                href = a.attrib.get("href", "")
                if "st_kb=" in href:
                    raw_title = a.text.strip()
                    clean_title = re.sub(r"<[^>]+>", "", raw_title).strip()
                    slug = href.split("st_kb=")[-1].split("&")[0]

                    if clean_title and slug not in seen_slugs:
                        seen_slugs.add(slug)
                        entries.append(
                            ScraplingConadisEntry(
                                title=clean_title,
                                slug=slug,
                                category=category_slug,
                                article_url=href,
                            )
                        )

                        if max_articles and len(entries) >= max_articles:
                            break

            if max_articles and len(entries) >= max_articles:
                break
            time.sleep(self.delay)

        print(f"[Scrapling] {len(entries)} términos identificados. Extrayendo IDs de YouTube...")

        # Extract YouTube embed from each article
        for idx, entry in enumerate(entries, 1):
            try:
                time.sleep(self.delay)
                article_page = self.fetcher.get(entry.article_url)
                iframes = article_page.css("iframe::attr(src)")

                yt_id = None
                for src in iframes:
                    match = re.search(r"youtube\.com/embed/([a-zA-Z0-9_-]+)", str(src))
                    if match:
                        yt_id = match.group(1)
                        break

                if yt_id:
                    entry.youtube_id = yt_id
                    entry.youtube_url = f"https://www.youtube.com/watch?v={yt_id}"
                    print(f"  [{idx:02d}/{len(entries):02d}] '{entry.title}' -> YouTube ID: {yt_id}")
                else:
                    print(f"  [{idx:02d}/{len(entries):02d}] '{entry.title}' -> Sin video YouTube detectado.")
            except Exception as e:
                print(f"  [{idx:02d}/{len(entries):02d}] Error en '{entry.title}': {e}")

        return entries

    def save_catalog(self, entries: List[ScraplingConadisEntry], save_path: Path):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(e) for e in entries]
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[Scrapling] Catálogo guardado en: {save_path.resolve()}")

