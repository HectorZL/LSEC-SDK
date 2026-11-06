@echo off
chcp 65001 > nul
title LSEC SDK - Traductor y Tutor DWPose (Acelerado por GPU NVIDIA RTX 3060)

echo ================================================================
echo  LSEC SDK - TRADUCTOR Y TUTOR EN VIVO CON DWPOSE (GPU ACCELERATION)
echo ================================================================
echo  Hardware detectado: NVIDIA GeForce RTX 3060 Laptop GPU
echo  Motor de Vision: DWPose Whole-Body + YOLOX (CUDA)
echo  Seguimiento de manos: 3D de alta precision (No falla de lado)
echo.
echo  Controles en vivo:
echo    [M] : Abrir Menu con pestanas de Secciones tematicas
echo    [T] : Alternar Modo Tutor / Modo Traductor Libre
echo    [C] : Cambiar de Seccion (SALUDOS, TRABAJO, COMIDA, SALUD...)
echo    [N] : Siguiente seña ^| [P] : Seña anterior
echo    [ESPACIO] : Saltar demostracion de palitos a imitar
echo    [Q] : Salir
echo ================================================================
echo.

python run_dwpose_translator.py

pause
