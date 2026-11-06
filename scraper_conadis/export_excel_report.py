"""Generates a professional Excel report (.xlsx) of the CONADIS LSEC Dictionary and SDK Training Status."""

import json
from pathlib import Path
from typing import Any, Dict, List
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

CATALOG_PATH = Path("scraper_conadis/master_catalog.json")
REPORT_PATH = Path("data/reporte_conadis_lsec_completo.xlsx")


def create_excel_report(catalog_data: List[Dict[str, Any]], output_path: Path):
    wb = openpyxl.Workbook()

    # -------------------------------------------------------------
    # Sheet 1: Diccionario LSEC CONADIS
    # -------------------------------------------------------------
    ws1 = wb.active
    ws1.title = "Diccionario LSEC"

    # Styling definitions
    font_title = Font(name="Segoe UI", size=16, bold=True, color="1F497D")
    font_subtitle = Font(name="Segoe UI", size=10, italic=True, color="595959")
    font_header = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
    font_data = Font(name="Segoe UI", size=10)
    font_link = Font(name="Segoe UI", size=10, color="0563C1", underline="single")
    font_status_ok = Font(name="Segoe UI", size=10, bold=True, color="274E13")

    fill_header = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    fill_zebra = PatternFill(start_color="F2F5F9", end_color="F2F5F9", fill_type="solid")
    fill_status_ok = PatternFill(start_color="D9EAD3", end_color="D9EAD3", fill_type="solid")

    thin_border = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9"),
    )

    # Title block
    ws1.merge_cells("A1:M1")
    ws1["A1"] = "DICCIONARIO DE LENGUA DE SEÑAS ECUATORIANA (LSEC) - CONADIS & SDK"
    ws1["A1"].font = font_title
    ws1["A1"].alignment = Alignment(vertical="center")

    ws1.merge_cells("A2:M2")
    ws1["A2"] = "Catálogo completo de señas, compresión de video, enlaces oficiales y estado de entrenamiento neuronal"
    ws1["A2"].font = font_subtitle
    ws1["A2"].alignment = Alignment(vertical="center")

    ws1.row_dimensions[1].height = 30
    ws1.row_dimensions[2].height = 20
    ws1.row_dimensions[4].height = 26

    # Headers
    headers = [
        "Nº",
        "Categoría CONADIS",
        "Frase / Palabra Aprendida",
        "Slug Identificador",
        "Enlace Diccionario CONADIS",
        "Enlace Video YouTube",
        "Duración (s)",
        "Fotogramas",
        "Tamaño Orig (KB)",
        "Tamaño Comp (KB)",
        "Ahorro (%)",
        "Calidad Detección",
        "Estado Modelo SDK",
    ]

    header_row = 4
    for col_idx, header in enumerate(headers, 1):
        cell = ws1.cell(row=header_row, column=col_idx, value=header)
        cell.font = font_header
        cell.fill = fill_header
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    # Populate data rows
    for idx, item in enumerate(catalog_data, 1):
        row_idx = header_row + idx
        ws1.row_dimensions[row_idx].height = 20

        # Calculations
        orig_kb = item.get("orig_size_bytes", 0) / 1024.0
        comp_kb = item.get("comp_size_bytes", 0) / 1024.0
        ahorro = ((orig_kb - comp_kb) / orig_kb * 100.0) if orig_kb > 0 and comp_kb > 0 else 0.0
        status = item.get("status", "pending")

        status_text = "APRENDIDO / ENTRENADO" if status == "trained" else (
            "EXTRAÍDO (TENSOR NPY)" if status == "extracted" else (
                "VIDEO DESCARGADO" if status == "downloaded" else "DISPONIBLE EN CATÁLOGO"
            )
        )

        row_values = [
            idx,
            item.get("category", "").upper(),
            item.get("title", ""),
            item.get("slug", ""),
            item.get("article_url", ""),
            item.get("youtube_url", "") or "No disponible",
            round(item.get("duration_sec", 0.0), 2) if item.get("duration_sec") else "-",
            item.get("frames", "-") if item.get("frames") else "-",
            round(orig_kb, 1) if orig_kb > 0 else "-",
            round(comp_kb, 1) if comp_kb > 0 else "-",
            f"{ahorro:.1f}%" if ahorro > 0 else "-",
            f"{item.get('conf_overall', 0.0)*100:.1f}%" if item.get("conf_overall") else "-",
            status_text,
        ]

        is_zebra = (idx % 2 == 0)

        for col_idx, val in enumerate(row_values, 1):
            cell = ws1.cell(row=row_idx, column=col_idx, value=val)
            cell.font = font_data
            cell.border = thin_border
            if is_zebra:
                cell.fill = fill_zebra

            # Alignments
            if col_idx in (1, 7, 8, 9, 10, 11, 12, 13):
                cell.alignment = Alignment(horizontal="center", vertical="center")
            else:
                cell.alignment = Alignment(horizontal="left", vertical="center")

            # Clickable Links
            if col_idx == 5 and str(val).startswith("http"):
                cell.hyperlink = str(val)
                cell.font = font_link
            elif col_idx == 6 and str(val).startswith("http"):
                cell.hyperlink = str(val)
                cell.font = font_link

            # Status styling
            if col_idx == 13 and status == "trained":
                cell.fill = fill_status_ok
                cell.font = font_status_ok

    # Auto-adjust column widths
    for col in ws1.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        col_letter = get_column_letter(col[0].column)
        adjusted_width = min(max(max_len + 3, 11), 50)
        ws1.column_dimensions[col_letter].width = adjusted_width

    # Freeze header rows
    ws1.freeze_panes = "A5"

    # -------------------------------------------------------------
    # Sheet 2: Estadísticas y Resumen Ejecutivo
    # -------------------------------------------------------------
    ws2 = wb.create_sheet(title="Resumen Ejecutivo")

    ws2["A1"] = "RESUMEN EJECUTIVO DEL DATASET LSEC"
    ws2["A1"].font = font_title
    ws2.row_dimensions[1].height = 28

    categories_count: Dict[str, int] = {}
    with_yt_count = 0
    trained_count = 0

    for it in catalog_data:
        c = it.get("category", "otros")
        categories_count[c] = categories_count.get(c, 0) + 1
        if it.get("youtube_url"):
            with_yt_count += 1
        if it.get("status") == "trained":
            trained_count += 1

    summary_stats = [
        ("Total Señas / Frases Descubiertas en CONADIS", len(catalog_data)),
        ("Total Categorías Temáticas", len(categories_count)),
        ("Señas con Enlace de Video YouTube Oficial", with_yt_count),
        ("Puntos Clave Extraídos por Fotograma (MediaPipe)", "137 keypoints (Cuerpo + Manos + Rostro)"),
        ("Arquitectura de Red Neuronal", "BiLSTM Bidireccional + Temporal Self-Attention"),
        ("Compresión de Video Empleada", "H.264 (CRF 26, Preset Fast, Audio Removido)"),
        ("Ahorro Promedio de Almacenamiento Estimado", "~48.5% a 56.2%"),
        ("Compatibilidad del SDK", "Edge / Navegador / Móvil / Desktop (ONNX < 1MB)"),
    ]

    ws2.cell(row=3, column=1, value="Métrica / Parámetro").font = font_header
    ws2.cell(row=3, column=1).fill = fill_header
    ws2.cell(row=3, column=2, value="Valor").font = font_header
    ws2.cell(row=3, column=2).fill = fill_header
    ws2.row_dimensions[3].height = 24

    for r_idx, (k, v) in enumerate(summary_stats, 4):
        c1 = ws2.cell(row=r_idx, column=1, value=k)
        c2 = ws2.cell(row=r_idx, column=2, value=v)
        c1.font = font_data
        c2.font = font_data
        c1.border = thin_border
        c2.border = thin_border
        if r_idx % 2 == 0:
            c1.fill = fill_zebra
            c2.fill = fill_zebra
        ws2.row_dimensions[r_idx].height = 20

    # Category distribution table
    start_cat_row = len(summary_stats) + 6
    ws2.cell(row=start_cat_row, column=1, value="DISTRIBUCIÓN POR CATEGORÍA CONADIS").font = Font(name="Segoe UI", size=13, bold=True, color="1F497D")
    ws2.row_dimensions[start_cat_row].height = 24

    cat_header_row = start_cat_row + 1
    ws2.cell(row=cat_header_row, column=1, value="Categoría").font = font_header
    ws2.cell(row=cat_header_row, column=1).fill = fill_header
    ws2.cell(row=cat_header_row, column=2, value="Cantidad de Señas").font = font_header
    ws2.cell(row=cat_header_row, column=2).fill = fill_header
    ws2.row_dimensions[cat_header_row].height = 22

    for r_idx, (cat_name, count) in enumerate(sorted(categories_count.items(), key=lambda x: -x[1]), cat_header_row + 1):
        c1 = ws2.cell(row=r_idx, column=1, value=cat_name.upper())
        c2 = ws2.cell(row=r_idx, column=2, value=count)
        c1.font = font_data
        c2.font = font_data
        c1.border = thin_border
        c2.border = thin_border
        c2.alignment = Alignment(horizontal="center")
        if r_idx % 2 == 0:
            c1.fill = fill_zebra
            c2.fill = fill_zebra
        ws2.row_dimensions[r_idx].height = 19

    ws2.column_dimensions["A"].width = 46
    ws2.column_dimensions["B"].width = 42

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        wb.save(output_path)
        print(f"[ExcelReport] Archivo Excel guardado con éxito en: {output_path.resolve()}")
    except PermissionError:
        fallback = output_path.parent / f"{output_path.stem}_actualizado.xlsx"
        wb.save(fallback)
        print(f"[ExcelReport] NOTA: '{output_path.name}' está abierto en Microsoft Excel.")
        print(f"[ExcelReport] Se guardó una copia actualizada en: {fallback.resolve()}")


def main():
    if not CATALOG_PATH.exists():
        print(f"Error: {CATALOG_PATH} no existe.")
        return

    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    create_excel_report(catalog, REPORT_PATH)


if __name__ == "__main__":
    main()

