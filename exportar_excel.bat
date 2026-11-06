@echo off
title LSEC SDK - Generador de Reporte Excel
echo ========================================================
echo  LSEC SDK - GENERADOR DE REPORTE EXCEL CONADIS
echo ========================================================
echo.
echo Generando reporte completo en data/reporte_conadis_lsec_completo.xlsx...
python scraper_conadis/export_excel_report.py
echo.
echo Reporte generado con exito.
start data\reporte_conadis_lsec_completo.xlsx
pause

