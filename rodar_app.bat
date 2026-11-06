@echo off
REM Sobe o app Streamlit no Docker (Windows) e abre o navegador em http://localhost:8501
cd /d "%~dp0"
docker build -t carga-alta:1.0 . || exit /b 1
start "" http://localhost:8501
docker run --rm -p 8501:8501 -v "%cd%\saida:/app/saida" carga-alta:1.0 app
