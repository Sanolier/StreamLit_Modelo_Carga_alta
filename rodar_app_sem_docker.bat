@echo off
REM Roda o app Streamlit sem Docker, num ambiente Python separado (nao mexe no seu Anaconda).
REM Usa o Python do Anaconda se existir; senao o launcher "py" ou o "python" do PATH. Precisa de Python 3.11+.
setlocal
cd /d "%~dp0"
set "VENV=%LOCALAPPDATA%\carga_alta_venv"

if exist "%VENV%\Scripts\python.exe" goto instalar

set "PY="
if exist "C:\ProgramData\anaconda3\python.exe" set "PY=C:\ProgramData\anaconda3\python.exe"
if not defined PY if exist "%USERPROFILE%\anaconda3\python.exe" set "PY=%USERPROFILE%\anaconda3\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\anaconda3\python.exe" set "PY=%LOCALAPPDATA%\anaconda3\python.exe"
if not defined PY (where py >nul 2>nul && set "PY=py")
if not defined PY set "PY=python"

"%PY%" -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
if errorlevel 1 (
    echo.
    echo Precisa de Python 3.11 ou mais novo. Python encontrado: %PY%
    "%PY%" --version
    pause
    exit /b 1
)
echo Criando ambiente em %VENV% com %PY% ...
"%PY%" -m venv "%VENV%"
if errorlevel 1 (pause & exit /b 1)

:instalar
echo Instalando dependencias (so demora na primeira vez)...
"%VENV%\Scripts\python.exe" -m pip install --upgrade pip >nul
"%VENV%\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (pause & exit /b 1)
REM evita a pergunta de e-mail que o Streamlit faz na primeira execucao
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
    > "%USERPROFILE%\.streamlit\credentials.toml" echo [general]
    >> "%USERPROFILE%\.streamlit\credentials.toml" echo email = ""
)
echo.
echo Abrindo o app em http://localhost:8501  (feche esta janela para parar)
"%VENV%\Scripts\python.exe" -m streamlit run app_streamlit.py --browser.gatherUsageStats false
pause
