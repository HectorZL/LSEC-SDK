"""Process conversational pack of LSEC signs for interactive testing."""

from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from master_processor import MasterProcessor

CONVERSATIONAL_SLUGS = [
    "buenos-dias",
    "buenas-tardes",
    "buenas-noches",
    "como-esta",
    "como-te-sientes",
    "esta-bien",
    "de-nada",
    "un-momento-por-favor",
    "adios",
    "hace-tiempo-que-no-te-veo",
    "hambre",
    "agua",
    "ya-no-mas",
    "mal",
    "donde-estudio",
    "donde",
    "en-que-le-puedo-ayudar",
]


def main():
    processor = MasterProcessor()
    catalog = processor.load_catalog()

    # Filter catalog items
    targets = [it for it in catalog if it.get("slug") in CONVERSATIONAL_SLUGS]
    print(f">> Se encontraron {len(targets)} señas del Pack Conversacional.")

    count = 0
    for idx, item in enumerate(targets, 1):
        if item.get("status") in ("trained", "extracted"):
            print(f"[{idx}/{len(targets)}] Ya aprendida: {item['title']}")
            continue

        print(f"[{idx}/{len(targets)}] Procesando: {item['title']} ({item['category']})...")
        if processor.process_item(item):
            item["status"] = "trained"
            count += 1
            processor.save_catalog(catalog)
        time.sleep(0.5)

    print(f"\n>> Finalizada extracción de {count} señas conversacionales.")
    print(">> Entrenando modelo en GPU NVIDIA RTX 3060...")
    processor.train_model(epochs=35)

    from export_excel_report import create_excel_report, REPORT_PATH
    create_excel_report(catalog, REPORT_PATH)
    print(">> ¡Pack conversacional listo para probar con iniciar_traductor.bat!")


if __name__ == "__main__":
    main()

