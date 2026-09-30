import pandas as pd
import os


# ==================================================
# ORDNER ERSTELLEN
# ==================================================

os.makedirs(
    "Ergebnisse/SARIMA",
    exist_ok=True
)


# ==================================================
# 1. MODELLBEWERTUNG
# ==================================================

modellvergleich = pd.DataFrame({

    "Datenreihe": [
        "Angebote",
        "Angebote",
        "Bestellungen",
        "Bestellungen",
        "Rechnungen",
        "Rechnungen"
    ],

    "Modell": [
        "Baseline",
        "SARIMA",
        "Baseline",
        "SARIMA",
        "Baseline",
        "SARIMA"
    ],

    "MAE_Euro": [
        643980.45,
        347574.08,
        142226.63,
        385312.17,
        181614.20,
        211761.18
    ],

    "RMSE_Euro": [
        768560.24,
        406796.83,
        191139.67,
        479134.44,
        239749.99,
        241075.50
    ],

    "MAPE_Prozent": [
        42.79,
        26.59,
        22.08,
        54.72,
        19.35,
        36.28
    ]
})


# ==================================================
# 2. SARIMA-MODELLE
# ==================================================

sarima_modelle = pd.DataFrame({

    "Datenreihe": [
        "Angebote",
        "Bestellungen",
        "Rechnungen"
    ],

    "order": [
        "(2,1,2)",
        "(1,0,2)",
        "(0,1,2)"
    ],

    "seasonal_order": [
        "(0,0,0,12)",
        "(1,0,0,12)",
        "(1,1,0,12)"
    ]
})


# ==================================================
# 3. FORECAST JUL-DEZ 2026
# ==================================================

monate = pd.date_range(
    start="2026-07-01",
    periods=6,
    freq="MS"
)

forecast = pd.DataFrame({

    "Monat": monate,

    "Angebote_SARIMA": [
        1483738.07,
        1393386.27,
        1362985.74,
        1378912.52,
        1416074.51,
        1451937.13
    ],

    "Bestellungen_SARIMA": [
        491116.07,
        766064.22,
        841852.85,
        1012184.20,
        769306.13,
        614205.15
    ],

    "Rechnungen_SARIMA": [
        1706630.31,
        1117942.69,
        1075553.19,
        1097650.56,
        1653319.24,
        1300711.55
    ]
})


# ==================================================
# 4. CSV-DATEIEN SPEICHERN
# ==================================================

modellvergleich.to_csv(
    "Ergebnisse/SARIMA/sarima_modellvergleich.csv",
    index=False,
    decimal=",",
    sep=";"
)

sarima_modelle.to_csv(
    "Ergebnisse/SARIMA/sarima_modelle.csv",
    index=False,
    sep=";"
)

forecast.to_csv(
    "Ergebnisse/SARIMA/sarima_forecast_2026.csv",
    index=False,
    decimal=",",
    sep=";"
)


# ==================================================
# 5. GEMEINSAME EXCEL-DATEI
# ==================================================

excel_datei = (
    "Ergebnisse/SARIMA/"
    "SARIMA_Gesamtergebnisse.xlsx"
)

with pd.ExcelWriter(
    excel_datei,
    engine="openpyxl"
) as writer:

    modellvergleich.to_excel(
        writer,
        sheet_name="Modellvergleich",
        index=False
    )

    sarima_modelle.to_excel(
        writer,
        sheet_name="Modelle",
        index=False
    )

    forecast.to_excel(
        writer,
        sheet_name="Forecast_2026",
        index=False
    )


print("\n======================================")
print("SARIMA ERGEBNISEXPORT ABGESCHLOSSEN")
print("======================================")

print("\nErstellt:")

print(
    "Ergebnisse/SARIMA/"
    "sarima_modellvergleich.csv"
)

print(
    "Ergebnisse/SARIMA/"
    "sarima_modelle.csv"
)

print(
    "Ergebnisse/SARIMA/"
    "sarima_forecast_2026.csv"
)

print(
    "Ergebnisse/SARIMA/"
    "SARIMA_Gesamtergebnisse.xlsx"
)