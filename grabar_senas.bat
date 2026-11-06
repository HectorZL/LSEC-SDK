@echo off
title LSEC SDK - Grabador de Senas
echo ========================================================
echo  LSEC SDK - GRABADOR DE SENAS INTERACTIVO
echo ========================================================
echo.
set /p SENA="Ingresa el nombre de la sena a grabar (ej: HOLA, GRACIAS, AYUDA): "
python record_dataset.py "%SENA%"
pause

