"""Crawler for CONADIS Gabriel Roman Sign Language Dictionary."""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import re
import time
import urllib.request
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

BASE_URL = "http://www.plataformaconadis.gob.ec/~platafor/diccionario"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


@dataclass
class ConadisEntry:
    title: str
    slug: str
    category: str
    article_url: str
    youtube_id: Optional[str] = None
    youtube_url: Optional[str] = None
    video_path: Optional[str] = None


class ConadisCrawler:
    """Crawls dictionary categories and extracts article links and YouTube video IDs."""

    def __init__(self, delay_seconds: float = 0.8):
        self.delay = delay_seconds
        self.headers = {"User-Agent": USER_AGENT}

    def _get_html(self, url: str) -> str:
        req = urllib.request.Request(url, headers=self.headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8", errors="ignore")

    def crawl_category(self, category_slug: str, max_articles: Optional[int] = None) -> List[ConadisEntry]:
        """
        Crawls all articles in a given category.

        Args:
            category_slug: Slug of the category (e.g. 'frases-comunes', 'dias-de-la-semana').
            max_articles: Max number of articles to extract (useful for testing/phases).
        """
        cat_url = f"{BASE_URL}/?article_category={category_slug}"
        print(f"[Crawler] Explorando categoría: '{category_slug}' -> {cat_url}")

        first_page_html = self._get_html(cat_url)

        # Detect pagination
        pages = set(
            re.findall(
                rf'href=["\']({re.escape(BASE_URL)}/\?article_category={category_slug}(?:&#038;|&)paged=\d+)["\']',
                first_page_html,
            )
        )
        all_pages = [cat_url] + [p.replace("&#038;", "&") for p in sorted(pages)]
        print(f"[Crawler] Páginas encontradas en '{category_slug}': {len(all_pages)}")

        entries: List[ConadisEntry] = []
        seen_urls = set()

        for page_url in all_pages:
            html = self._get_html(page_url)
            # Find links to knowledgebase articles: ?st_kb=...
            matches = re.findall(r'<a\s+[^>]*href=["\']([^"\']*st_kb=([^"\']+))["\'][^>]*>(.*?)</a>', html, re.DOTALL)

            for full_url, slug, raw_text in matches:
                clean_title = re.sub(r"<[^>]+>", "", raw_text).strip()
                if not clean_title or full_url in seen_urls:
                    continue

                seen_urls.add(full_url)
                entries.append(
                    ConadisEntry(
                        title=clean_title,
                        slug=slug,
                        category=category_slug,
                        article_url=full_url,
                    )
                )

                if max_articles and len(entries) >= max_articles:
                    break

            if max_articles and len(entries) >= max_articles:
                break
            time.sleep(self.delay)

        print(f"[Crawler] Extraídos {len(entries)} términos en '{category_slug}'. Obteniendo enlaces de video...")

        # Now fetch each article page to extract the YouTube video ID
        for idx, entry in enumerate(entries, 1):
            try:
                time.sleep(self.delay)
                article_html = self._get_html(entry.article_url)
                yt_ids = re.findall(r"youtube\.com/embed/([a-zA-Z0-9_-]+)", article_html)
                if yt_ids:
                    entry.youtube_id = yt_ids[0]
                    entry.youtube_url = f"https://www.youtube.com/watch?v={yt_ids[0]}"
                    print(f"  [{idx:02d}/{len(entries):02d}] '{entry.title}' -> YT: {entry.youtube_id}")
                else:
                    print(f"  [{idx:02d}/{len(entries):02d}] '{entry.title}' -> Sin video YouTube detectado.")
            except Exception as e:
                print(f"  [{idx:02d}/{len(entries):02d}] Error obteniendo '{entry.title}': {e}")

        return entries

    def save_catalog(self, entries: List[ConadisEntry], save_path: Path):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(e) for e in entries]
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[Crawler] Catálogo guardado en: {save_path.resolve()}")

