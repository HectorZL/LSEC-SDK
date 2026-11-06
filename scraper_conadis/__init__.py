"""Scraper package for CONADIS LSEC Dictionary."""

from .conadis_crawler import ConadisCrawler, ConadisEntry
from .video_downloader import ConadisVideoDownloader

__all__ = ["ConadisCrawler", "ConadisEntry", "ConadisVideoDownloader"]

