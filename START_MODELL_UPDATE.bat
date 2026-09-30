@echo off
chcp 65001 >nul
cd /d C:\CashAI

echo ============================================================
echo CASH-AI: XGBoost + SARIMA aktualisieren
echo Grundlage: bereits aufbereitete Daten in C:\Ergebniss_XGBoost
echo ============================================================
echo.

python UPDATE_CASH_AI_MODELS.py

echo.
echo ============================================================
echo Fertig. Ergebnisse:
echo C:\CashAI\Ergebnisse\AUTO_UPDATE
echo ============================================================
pause
