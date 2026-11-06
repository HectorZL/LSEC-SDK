@echo off
title LSEC SDK - Descarga y Entrenamiento CONADIS
cls
echo ================================================================
echo  LSEC SDK - DESCARGA Y ENTRENAMIENTO POR LOTE (CONADIS ECUADOR)
echo ================================================================
echo.
echo El catalogo contiene 1,096 senas oficiales de CONADIS.
echo.
echo Selecciona que deseas hacer:
echo.
echo   [1] Procesar un lote de 10 senas (Rapido, ~1 minuto)
echo   [2] Procesar un lote personalizado (ej. 25, 50 senas)
echo   [3] Procesar una categoria especifica (ej. frases-comunes, dias-de-la-semana)
echo   [4] Procesar TODO CONADIS completo (1,085 senas restantes)
echo.
set /p OPCION="Elige una opcion (1-4, por defecto [1]): "

if "%OPCION%"=="" set OPCION=1
if "%OPCION%"=="1" goto LOTE_10
if "%OPCION%"=="2" goto LOTE_CUSTOM
if "%OPCION%"=="3" goto CATEGORIA
if "%OPCION%"=="4" goto TODO_CONADIS

:LOTE_10
echo.
echo >> Procesando lote de 10 senas...
python scraper_conadis/master_processor.py --batch-size 10
goto FIN

:LOTE_CUSTOM
set /p CANT="Cuantas senas deseas procesar?: "
if "%CANT%"=="" set CANT=10
echo.
echo >> Procesando lote de %CANT% senas...
python scraper_conadis/master_processor.py --batch-size %CANT%
goto FIN

:CATEGORIA
echo.
echo Categorias populares: frases-comunes, dias-de-la-semana, meses-del-ano,
echo                      ciudades-ecuador, provincias-ecuador, numeros, animales, etc.
echo.
set /p CAT="Ingresa el nombre exacto de la categoria: "
if "%CAT%"=="" set CAT=frases-comunes
echo.
echo >> Procesando categoria '%CAT%'...
python scraper_conadis/master_processor.py --category "%CAT%" --batch-size 0
goto FIN

:TODO_CONADIS
echo.
echo ================================================================
echo ATENCION: Procesar las ~1,085 senas tomara entre 1.5 y 2 horas.
echo El proceso ira guardando y re-entrenando el modelo en tu GPU
echo cada 25 senas, por lo que puedes detenerlo cuando quieras con Ctrl+C
echo sin perder ningun avance.
echo ================================================================
echo.
pause
python scraper_conadis/master_processor.py --all
goto FIN

:FIN
echo.
echo ================================================================
echo  Proceso finalizado. El modelo y el Excel han sido actualizados.
echo ================================================================
pause
