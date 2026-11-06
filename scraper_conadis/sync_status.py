"""Synchronizes already processed data/conadis_videos and metadata.csv into master_catalog.json."""

import glob
import json
from pathlib import Path
import pandas as pd

CATALOG_PATH = Path("scraper_conadis/master_catalog.json")
METADATA_PATH = Path("data/processed/metadata.csv")

with open(CATALOG_PATH, "r", encoding="utf-8") as f:
    catalog = json.load(f)

slug_to_item = {item["slug"]: item for item in catalog}

# Load metadata.csv
if METADATA_PATH.exists():
    df = pd.read_csv(METADATA_PATH)
    for _, row in df.iterrows():
        seq_id = str(row["sequence_id"])
        # Format might be conadis_category_slug or lsec_023_como_te_sientes
        for slug, item in slug_to_item.items():
            if slug in seq_id or slug.replace("-", "_") in seq_id:
                item["status"] = "trained"
                n_frames = int(row.get("num_frames", row.get("total_frames", 0)))
                fps_val = float(row.get("fps", 25.0))
                dur_val = round(n_frames / fps_val, 2) if fps_val > 0 else 0.0
                conf_val = float(row.get("mean_conf_overall", row.get("confidence_overall", 0.0)))
                item["duration_sec"] = dur_val
                item["frames"] = n_frames
                item["conf_overall"] = conf_val

# Check video files
for mp4 in glob.glob("data/conadis_videos/**/*.mp4", recursive=True):
    p = Path(mp4)
    slug = p.stem
    if slug in slug_to_item:
        size = p.stat().st_size
        item = slug_to_item[slug]
        item["comp_size_bytes"] = size
        if item["orig_size_bytes"] == 0:
            item["orig_size_bytes"] = int(size * 1.8)  # Estimated original before CRF 26
        if item["status"] == "pending":
            item["status"] = "downloaded"

with open(CATALOG_PATH, "w", encoding="utf-8") as f:
    json.dump(catalog, f, ensure_ascii=False, indent=2)

print("Sincronización completada.")

