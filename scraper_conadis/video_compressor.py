"""Video compressor using FFmpeg to achieve optimal quality-to-size ratio for keypoint extraction."""

import logging
from pathlib import Path
import subprocess
from typing import Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class VideoCompressor:
    """
    Compresses videos using H.264 (libx264) with balanced Constant Rate Factor (CRF)
    to achieve minimal file size while preserving sharp finger and facial contours for MediaPipe.
    """

    def __init__(
        self,
        crf: int = 26,  # 24-28 is the sweet spot for computer vision
        max_height: int = 720,
        fps: int = 25,
        preset: str = "faster",
    ):
        self.crf = crf
        self.max_height = max_height
        self.fps = fps
        self.preset = preset

    def compress(
        self,
        input_video: Union[str, Path],
        output_video: Optional[Union[str, Path]] = None,
        replace_original: bool = False,
    ) -> Tuple[Path, Dict[str, Union[int, float]]]:
        """
        Compresses an MP4 video file.

        Args:
            input_video: Source video path.
            output_video: Destination path (if None, generates a _compressed.mp4 name).
            replace_original: If True, replaces the original with the compressed version.
        Returns:
            (compressed_path, stats_dict)
        """
        in_path = Path(input_video).resolve()
        if not in_path.exists():
            raise FileNotFoundError(f"Input video not found: {in_path}")

        orig_size = in_path.stat().st_size

        if output_video is None:
            out_path = in_path.parent / f"{in_path.stem}_opt.mp4"
        else:
            out_path = Path(output_video).resolve()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        temp_out = out_path if out_path != in_path else in_path.parent / f"{in_path.stem}_temp_opt.mp4"

        # FFmpeg command:
        # - Scale height to max_height preserving aspect ratio (must be even: -2)
        # - Remove audio (-an) since sign language models do not process sound
        # - CRF 26 + faster preset + 25 fps
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite
            "-i", str(in_path),
            "-vcodec", "libx264",
            "-crf", str(self.crf),
            "-preset", self.preset,
            "-vf", f"scale=-2:min({self.max_height}\\,ih)",
            "-r", str(self.fps),
            "-an",  # Strip audio track for sign language
            "-movflags", "+faststart",  # Web-streamable
            str(temp_out),
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg compression failed: {result.stderr}")
            raise RuntimeError(f"FFmpeg error: {result.stderr[-300:]}")

        new_size = temp_out.stat().st_size
        compression_ratio = round((1.0 - (new_size / max(1, orig_size))) * 100, 1)

        final_path = out_path
        if replace_original:
            temp_out.replace(in_path)
            final_path = in_path
        elif temp_out != out_path:
            temp_out.replace(out_path)

        stats = {
            "orig_size_bytes": orig_size,
            "new_size_bytes": new_size,
            "saved_percent": compression_ratio,
            "crf": self.crf,
            "max_height": self.max_height,
        }

        print(f"[VideoCompressor] '{in_path.name}': {orig_size // 1024} KB -> {new_size // 1024} KB (Ahorro: {compression_ratio}%)")
        return final_path, stats

