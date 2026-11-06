import sys
from pathlib import Path
import urllib.request
import time

MODELS_DIR = Path("models/dwpose")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

URLS = {
    "yolox_l.onnx": "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx",
    "dw-ll_ucoco_384.onnx": "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx",
}

def download_file(name: str, url: str):
    dest = MODELS_DIR / name
    if dest.exists() and dest.stat().st_size > 1024 * 1024:
        print(f"[OK] {name} ya existe ({dest.stat().st_size / (1024*1024):.1f} MB)")
        return dest

    print(f"\nDescargando {name} desde {url}...")
    start_t = time.time()
    
    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = downloaded / total_size * 100
            elapsed = max(0.1, time.time() - start_t)
            speed = (downloaded / (1024 * 1024)) / elapsed
            sys.stdout.write(f"\r  Progreso: {percent:5.1f}% [{downloaded/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB] ({speed:.1f} MB/s)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, str(dest), reporthook)
    print(f"\n[COMPLETADO] {name} guardado en {dest}")
    return dest

def main():
    print("=" * 60)
    print(" DESCARGA DE MODELOS DWPOSE (GPU ACCELERATION)")
    print("=" * 60)
    for name, url in URLS.items():
        download_file(name, url)
    print("\n¡Todos los modelos DWPose están listos!")

if __name__ == "__main__":
    main()
