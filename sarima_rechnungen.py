import pandas as pd
import numpy as np

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# 1. AUSGANGSRECHNUNGEN EINLESEN
# --------------------------------------------------

rechnungen = pd.read_excel(
    "Ausgangsrechnungen/ausgangsrechnungen.xlsx"
)

print("\nDatei erfolgreich eingelesen.")


# --------------------------------------------------
# 2. MONATLICHEN UMSATZ BILDEN
# --------------------------------------------------

monatsumsatz = (
    rechnungen
    .set_index("rechnungsdatum")["rechnungsbetrag"]
    .resample("MS")
    .sum()
)

print("\nMONATSUMSATZ:")
print(monatsumsatz)


# --------------------------------------------------
# 3. TRAINING UND TEST
# --------------------------------------------------

train = monatsumsatz[
    monatsumsatz.index < "2026-01-01"
]

test = monatsumsatz[
    (monatsumsatz.index >= "2026-01-01") &
    (monatsumsatz.index <= "2026-06-01")
]

print("\nAnzahl Trainingsmonate:", len(train))
print("Anzahl Testmonate:", len(test))


# --------------------------------------------------
# 4. SARIMA MODELL
# --------------------------------------------------

modell = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(0, 0, 0, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)

modell_fit = modell.fit(disp=False)

print("\nSARIMA-Modell wurde trainiert.")


# --------------------------------------------------
# 5. PROGNOSE JANUAR BIS JUNI 2026
# --------------------------------------------------

prognose_test = modell_fit.forecast(
    steps=len(test)
)

print("\nPROGNOSE JAN-JUN 2026:")
print(prognose_test)


# --------------------------------------------------
# 6. ABWEICHUNGSANALYSE
# --------------------------------------------------

vergleich = pd.DataFrame({
    "Ist": test,
    "SARIMA_Prognose": prognose_test
})

vergleich["Abweichung_Euro"] = (
    vergleich["Ist"]
    - vergleich["SARIMA_Prognose"]
)

vergleich["Absolute_Abweichung"] = (
    vergleich["Abweichung_Euro"].abs()
)

vergleich["Abweichung_Prozent"] = (
    vergleich["Abweichung_Euro"]
    / vergleich["Ist"]
) * 100

print("\nABWEICHUNGSANALYSE:")
print(vergleich.round(2))


# --------------------------------------------------
# 7. MODELLBEWERTUNG
# --------------------------------------------------

mae = mean_absolute_error(
    test,
    prognose_test
)

rmse = np.sqrt(
    mean_squared_error(
        test,
        prognose_test
    )
)

mape = np.mean(
    np.abs(
        (test - prognose_test) / test
    )
) * 100

print("\nMODELLBEWERTUNG:")
print(f"MAE: {mae:,.2f} Euro")
print(f"RMSE: {rmse:,.2f} Euro")
print(f"MAPE: {mape:.2f} %")
# ==================================================
# BASELINE: SAISONAL NAIVE PROGNOSE
# ==================================================

baseline = train[
    (train.index >= "2025-01-01") &
    (train.index <= "2025-06-01")
].copy()

# Index auf 2026 setzen
baseline.index = test.index

baseline_mae = mean_absolute_error(
    test,
    baseline
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        test,
        baseline
    )
)

baseline_mape = np.mean(
    np.abs(
        (test - baseline) / test
    )
) * 100


print("\n======================================")
print("BASELINE JAN-JUN 2026")
print("======================================")

baseline_vergleich = pd.DataFrame({
    "Ist": test,
    "Baseline": baseline
})

print(baseline_vergleich.round(2))


print("\nBASELINE MODELLBEWERTUNG:")
print(f"MAE:  {baseline_mae:,.2f} Euro")
print(f"RMSE: {baseline_rmse:,.2f} Euro")
print(f"MAPE: {baseline_mape:.2f} %")
# ==================================================
# SARIMA PARAMETEROPTIMIERUNG
# ==================================================

import itertools
import warnings

warnings.filterwarnings("ignore")


# --------------------------------------------------
# TRAINING UND VALIDIERUNG
# --------------------------------------------------

train_opt = monatsumsatz[
    monatsumsatz.index < "2025-07-01"
]

validierung = monatsumsatz[
    (monatsumsatz.index >= "2025-07-01") &
    (monatsumsatz.index <= "2025-12-01")
]


print("\n======================================")
print("SARIMA PARAMETEROPTIMIERUNG")
print("======================================")

print("Trainingsmonate:", len(train_opt))
print("Validierungsmonate:", len(validierung))


# --------------------------------------------------
# PARAMETER DEFINIEREN
# --------------------------------------------------

p = [0, 1, 2]
d = [0, 1]
q = [0, 1, 2]

P = [0, 1]
D = [0, 1]
Q = [0, 1]

parameter = list(
    itertools.product(p, d, q)
)

saisonale_parameter = list(
    itertools.product(P, D, Q)
)


# --------------------------------------------------
# MODELLE TESTEN
# --------------------------------------------------

ergebnisse = []

for order in parameter:

    for saison in saisonale_parameter:

        seasonal_order = (
            saison[0],
            saison[1],
            saison[2],
            12
        )

        try:

            modell_test = SARIMAX(
                train_opt,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            modell_test_fit = modell_test.fit(
                disp=False,
                maxiter=200
            )

            prognose_validierung = (
                modell_test_fit.forecast(
                    steps=len(validierung)
                )
            )

            mae_test = mean_absolute_error(
                validierung,
                prognose_validierung
            )

            rmse_test = np.sqrt(
                mean_squared_error(
                    validierung,
                    prognose_validierung
                )
            )

            ergebnisse.append({
                "order": order,
                "seasonal_order": seasonal_order,
                "MAE": mae_test,
                "RMSE": rmse_test
            })

        except Exception:
            pass


# --------------------------------------------------
# ERGEBNISSE SORTIEREN
# --------------------------------------------------

ergebnisse_df = pd.DataFrame(ergebnisse)

ergebnisse_df = ergebnisse_df.sort_values(
    by="MAE"
)


print("\nDIE 10 BESTEN SARIMA-MODELLE:")
print(
    ergebnisse_df.head(10).to_string(
        index=False
    )
)
# ==================================================
# FINALES SARIMA-MODELL
# ==================================================

print("\n======================================")
print("FINALER SARIMA-TEST")
print("======================================")


# --------------------------------------------------
# 1. Training bis Dezember 2025
# --------------------------------------------------

train_final = monatsumsatz[
    monatsumsatz.index < "2026-01-01"
]

test_final = monatsumsatz[
    (monatsumsatz.index >= "2026-01-01") &
    (monatsumsatz.index <= "2026-06-01")
]


# --------------------------------------------------
# 2. Bestes SARIMA-Modell trainieren
# --------------------------------------------------

finales_modell = SARIMAX(
    train_final,
    order=(0, 1, 2),
    seasonal_order=(1, 1, 0, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)

finales_modell_fit = finales_modell.fit(
    disp=False,
    maxiter=200
)


# --------------------------------------------------
# 3. Prognose Jan-Jun 2026
# --------------------------------------------------

finale_prognose = finales_modell_fit.forecast(
    steps=len(test_final)
)


# --------------------------------------------------
# 4. Abweichungsanalyse
# --------------------------------------------------

finaler_vergleich = pd.DataFrame({
    "Ist": test_final,
    "SARIMA_Prognose": finale_prognose
})

finaler_vergleich["Abweichung_Euro"] = (
    finaler_vergleich["Ist"]
    - finaler_vergleich["SARIMA_Prognose"]
)

finaler_vergleich["Absolute_Abweichung"] = (
    finaler_vergleich["Abweichung_Euro"].abs()
)

finaler_vergleich["Abweichung_Prozent"] = (
    finaler_vergleich["Abweichung_Euro"]
    / finaler_vergleich["Ist"]
) * 100


print("\nFINALER VERGLEICH JAN-JUN 2026:")
print(
    finaler_vergleich.round(2)
)


# --------------------------------------------------
# 5. Modellbewertung
# --------------------------------------------------

final_mae = mean_absolute_error(
    test_final,
    finale_prognose
)

final_rmse = np.sqrt(
    mean_squared_error(
        test_final,
        finale_prognose
    )
)

final_mape = np.mean(
    np.abs(
        (test_final - finale_prognose)
        / test_final
    )
) * 100


print("\nFINALE SARIMA-MODELLBEWERTUNG:")
print(f"MAE:  {final_mae:,.2f} Euro")
print(f"RMSE: {final_rmse:,.2f} Euro")
print(f"MAPE: {final_mape:.2f} %")
# ==================================================
# ECHTE ZUKUNFTSPROGNOSE JUL-DEZ 2026
# ==================================================

print("\n======================================")
print("SARIMA FORECAST JUL-DEZ 2026")
print("======================================")


# Alle verfügbaren Ist-Daten bis Juni 2026
train_zukunft = monatsumsatz[
    monatsumsatz.index <= "2026-06-01"
]


# Finales SARIMA neu mit allen verfügbaren Daten trainieren
zukunft_modell = SARIMAX(
    train_zukunft,
    order=(0, 1, 2),
    seasonal_order=(1, 1, 0, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)

zukunft_modell_fit = zukunft_modell.fit(
    disp=False,
    maxiter=200
)


# 6 Monate prognostizieren
forecast_zukunft = zukunft_modell_fit.forecast(
    steps=6
)


print("\nPROGNOSE JUL-DEZ 2026:")

for datum, wert in forecast_zukunft.items():
    print(
        datum.strftime("%Y-%m"),
        f"{wert:,.2f} Euro"
    )
    # ==================================================
# ERGEBNISSE SPEICHERN
# ==================================================

import os

os.makedirs("Ergebnisse/SARIMA", exist_ok=True)


# --------------------------------------------------
# Forecast Jul-Dez 2026 als Tabelle
# --------------------------------------------------

forecast_tabelle = pd.DataFrame({
    "SARIMA_Prognose": forecast_zukunft
})

forecast_tabelle.index.name = "Monat"


# --------------------------------------------------
# Modellvergleich
# --------------------------------------------------

modellvergleich = pd.DataFrame({
    "Modell": [
        "Baseline",
        "SARIMA"
    ],
    "MAE": [
        baseline_mae,
        final_mae
    ],
    "RMSE": [
        baseline_rmse,
        final_rmse
    ],
    "MAPE_Prozent": [
        baseline_mape,
        final_mape
    ]
})


# --------------------------------------------------
# Excel-Datei erstellen
# --------------------------------------------------

dateiname = "Ergebnisse/SARIMA/SARIMA_Ergebnisse.xlsx"

with pd.ExcelWriter(
    dateiname,
    engine="openpyxl"
) as writer:

    finaler_vergleich.to_excel(
        writer,
        sheet_name="Abweichungsanalyse"
    )

    forecast_tabelle.to_excel(
        writer,
        sheet_name="Forecast_Jul_Dez"
    )

    modellvergleich.to_excel(
        writer,
        sheet_name="Modellbewertung",
        index=False
    )


print("\nSARIMA-Ergebnisse gespeichert:")
print(dateiname)
# ==================================================
# SARIMA GRAFIK
# ==================================================

import matplotlib.pyplot as plt


plt.figure(figsize=(12, 6))

plt.plot(
    monatsumsatz.index,
    monatsumsatz.values,
    marker="o",
    label="Ist-Umsatz"
)

plt.plot(
    forecast_zukunft.index,
    forecast_zukunft.values,
    marker="o",
    label="SARIMA Forecast"
)

plt.title(
    "Monatlicher Umsatz und SARIMA-Prognose"
)

plt.xlabel("Monat")
plt.ylabel("Umsatz in Euro")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "Ergebnisse/SARIMA/SARIMA_Forecast.png",
    dpi=300
)

plt.show()