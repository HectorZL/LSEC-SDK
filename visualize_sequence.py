#!/usr/bin/env python3
"""Script to generate visual comparisons (GIF & snapshots) between raw video and extracted keypoints."""

from pathlib import Path
import shutil
import sys

# Ensure src in path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np

from lsec_sdk.visualizer import Visualizer


def main():
    seq_id = "-fZc293MpJk_0-1-rgb_front"
    video_path = Path(f"openpose_output/video/{seq_id}.mp4")
    npy_path = Path(f"data/processed/sequences/{seq_id}.npy")
    out_dir = Path("data/processed/visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating visual validation for: {seq_id}")
    visualizer = Visualizer(confidence_threshold=0.1)

    # 1. Export animated GIF
    gif_file = out_dir / f"{seq_id}_validation.gif"
    print(f"Creating comparison GIF...")
    visualizer.export_gif(npy_path, video_path, gif_file, fps=8, scale=0.5)
    print(f"GIF saved: {gif_file} ({gif_file.stat().st_size} bytes)")

    # 2. Export key frame snapshots (Frame 0, Frame 8, Frame 16)
    tensor = np.load(npy_path)
    cap = cv2.VideoCapture(str(video_path))
    num_frames = tensor.shape[0]

    key_frames = [0, num_frames // 2, num_frames - 1]
    saved_snapshots = []

    for idx in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if idx in key_frames:
            sbs = visualizer.create_side_by_side(frame, tensor[idx], idx, num_frames, scale=0.7)
            snap_file = out_dir / f"snapshot_frame_{idx:02d}.jpg"
            cv2.imwrite(str(snap_file), sbs)
            saved_snapshots.append(snap_file)
            print(f"Saved snapshot frame {idx}: {snap_file}")

    cap.release()

    # 3. Copy artifacts to appData artifact directory if it exists
    artifact_dir = Path(r"C:\Users\jesuc\.gemini\antigravity\brain\212d5e7e-eafe-4f2a-a7f1-818fbc51e65b")
    if artifact_dir.exists():
        for snap in saved_snapshots:
            dest = artifact_dir / snap.name
            shutil.copy2(snap, dest)
            print(f"Copied {snap.name} to artifact directory: {dest}")
        dest_gif = artifact_dir / gif_file.name
        shutil.copy2(gif_file, dest_gif)
        print(f"Copied GIF to artifact directory: {dest_gif}")

    print("Visual validation complete!")


if __name__ == "__main__":
    main()

