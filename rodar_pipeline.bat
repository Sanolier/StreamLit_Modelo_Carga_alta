@echo off
REM
cd /d "%~dp0"
docker build -t carga-alta:1.0 . || exit /b 1
docker run --rm -v "%cd%\saida:/app/saida" carga-alta:1.0 tudo || exit /b 1
echo.
echo Pronto. Veja saida\avaliacao\relatorio.md e saida\previsoes\previsoes.csv
