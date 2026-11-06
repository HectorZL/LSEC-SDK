@echo off
title LSEC SDK - Pack Conversacional
echo ================================================================
echo  LSEC SDK - APRENDIZAJE DEL PACK CONVERSACIONAL (LSEC ECUADOR)
echo ================================================================
echo.
echo Se procesaran las frases conversacionales clave del diccionario:
echo   - Saludos: Buenos dias, Buenas tardes, Buenas noches, Adios
echo   - Preguntas: Como esta?, Donde estudio?, En que le puedo ayudar?
echo   - Respuestas: Esta bien? (OK), Mal, De nada, Un momento por favor
echo   - Rutina: Hambre (Ya comiste/Comida), Agua, Ya no mas
echo.
echo Descargando, comprimiendo y entrenando en tu GPU RTX 3060...
python scraper_conadis/process_conversational.py
echo.
echo ================================================================
echo  Listo. Ya puedes abrir 'iniciar_traductor.bat' para probarlas!
echo ================================================================
pause

