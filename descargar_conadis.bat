@echo off
title LSEC SDK - Descargador y Entrenador CONADIS
echo ========================================================
echo  LSEC SDK - DESCARGADOR DE VIDEOS Y TENSORES CONADIS
echo ========================================================
echo.
echo Categorias populares:
echo   - frases-comunes
echo   - dias-de-la-semana
echo   - meses-del-ano
echo   - numeros
echo   - ciudades-ecuador
echo   - provincias-ecuador
echo.
set /p CAT="Ingresa la categoria a procesar (Enter para 'frases-comunes'): "
if "%CAT%"=="" set CAT=frases-comunes

set /p LIM="Cuantos videos deseas descargar y aprender (Enter para 10): "
if "%LIM%"=="" set LIM=10

echo.
echo Iniciando descarga, compresion CRF 26, extraccion y entrenamiento...
python scraper_conadis/master_processor.py --category "%CAT%" --batch-size %LIM%
pause
