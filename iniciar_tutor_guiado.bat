@echo off
title LSEC SDK - Tutor de Ensenanza Guiada por Secciones
cls
echo ================================================================
echo  LSEC SDK - TUTOR DE ENSENANZA GUIADA (PALITOS Y SECCIONES)
echo ================================================================
echo.
echo Secciones tematicas disponibles:
echo   - [1] SALUDOS   : Buenos dias, Buenas noches, Adios, De nada...
echo   - [2] TRABAJO   : Donde estudio, Abrir cuenta, Credito, Banco...
echo   - [3] COMIDA    : Hambre, Agua, Ya no mas, Babaco, Abrelatas...
echo   - [4] SALUD     : Como te sientes, Como esta, Esta bien, Mal...
echo   - [5] GENERAL   : Vocabulario alfabetico LSEC
echo   - [6] TODAS     : Catalogo completo (46 senas)
echo.
echo Modos de inicio:
echo   [1] Abrir camara directamente (Usa [M] o [C] en vivo para cambiar)
echo   [2] Seleccionar seccion y sena aqui antes de abrir la camara
echo.
set /p M_OPT="Elige una opcion (1 o 2, por defecto [1]): "
if "%M_OPT%"=="" set M_OPT=1

if "%M_OPT%"=="2" (
    python run_live_translator.py --tutor --menu
) else (
    echo.
    echo Iniciando camara y tutor de ensenanza...
    echo Controles en vivo:
    echo   [M] : Abrir Menu con pestanas de Secciones
    echo   [C] : Cambiar de Seccion al instante (SALUDOS - TRABAJO - COMIDA - SALUD)
    echo   [N] : Siguiente sena de la seccion
    echo.
    python run_live_translator.py --tutor
)
pause
