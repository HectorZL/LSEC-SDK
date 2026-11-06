"""Video downloader for CONADIS LSEC videos using yt-dlp."""

from pathlib import Path
import time
from typing import List, Optional
import yt_dlp

try:
    from .conadis_crawler import ConadisEntry
except ImportError:
    from conadis_crawler import ConadisEntry


class ConadisVideoDownloader:
    """Downloads YouTube videos for catalog entries and organizes them into category folders."""

    def __init__(self, base_output_dir: Path = Path("data/conadis_videos")):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

    def download_entry(self, entry: ConadisEntry, delay_seconds: float = 1.0) -> Optional[Path]:
        """Downloads a single video entry into data/conadis_videos/{category}/{slug}.mp4."""
        if not entry.youtube_url:
            print(f"[Downloader] Se omite '{entry.title}' (no tiene enlace de YouTube).")
            return None

        category_dir = self.base_output_dir / entry.category
        category_dir.mkdir(parents=True, exist_ok=True)

        target_file = category_dir / f"{entry.slug}.mp4"
        if target_file.exists() and target_file.stat().st_size > 1000:
            print(f"[Downloader] Ya existe: {target_file.name} ({target_file.stat().st_size} bytes).")
            entry.video_path = str(target_file.resolve())
            return target_file

        print(f"[Downloader] Descargando: '{entry.title}' ({entry.youtube_id})...")

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
                ydl.download([entry.youtube_url])

            if target_file.exists():
                entry.video_path = str(target_file.resolve())
                print(f"  -> Guardado exitosamente: {target_file.name} ({target_file.stat().st_size} bytes)")
                time.sleep(delay_seconds)
                return target_file
            else:
                print(f"  -> Falló la descarga de {target_file.name}.")
                return None
        except Exception as e:
            print(f"  -> Error al descargar '{entry.title}': {e}")
            return None

    def download_batch(self, entries: List[ConadisEntry], delay_seconds: float = 1.0) -> List[Path]:
        """Downloads a list of entries and returns the paths of downloaded files."""
        downloaded = []
        for idx, entry in enumerate(entries, 1):
            print(f"[{idx:02d}/{len(entries):02d}] Procesando descarga de '{entry.title}'...")
            path = self.download_entry(entry, delay_seconds=delay_seconds)
            if path:
                downloaded.append(path)
        return downloaded
