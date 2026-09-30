@echo off
chcp 65001 >nul
cd /d C:\CashAI

echo ============================================================
echo CASH-AI KOMPLETT-UPDATE
echo Datenmodell + XGBoost + SARIMA + Power-BI-Datei
echo ============================================================
echo.

set /p STICHTAG=Neuen Datenstichtag eingeben (YYYY-MM-DD): 

if "%STICHTAG%"=="" (
    echo Kein Stichtag eingegeben. Abbruch.
    pause
    exit /b 1
)

python UPDATE_CASH_AI_MODELS.py --prepare-data --datenstichtag %STICHTAG%

echo.
echo ============================================================
echo Fertig. Ergebnisse:
echo C:\CashAI\Ergebnisse\AUTO_UPDATE
echo Power BI:
echo C:\CashAI\Ergebnisse\AUTO_UPDATE\PowerBI\PowerBI_Cashflow_Gesamt.csv
echo ============================================================
pause
