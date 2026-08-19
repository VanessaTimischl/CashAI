# ==================================================
# SARIMA - ANGEBOTE
# ==================================================

import pandas as pd
import numpy as np
import itertools
import warnings

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


# ==================================================
# 1. DATEN EINLESEN
# ==================================================

angebote = pd.read_excel(
    "Angebote/Angebot_Aufbereitet.xlsx"
)

print("\n======================================")
print("SARIMA - ANGEBOTE")
print("======================================")

print("\nSPALTENNAMEN:")
print(angebote.columns.tolist())


# ==================================================
# 2. DATENTYPEN SICHERSTELLEN
# ==================================================

angebote["angebotsdatum"] = pd.to_datetime(
    angebote["angebotsdatum"],
    errors="coerce"
)

angebote["angebotswert"] = pd.to_numeric(
    angebote["angebotswert"],
    errors="coerce"
)


print("\nZEITRAUM:")
print(
    "Erstes Angebot:",
    angebote["angebotsdatum"].min()
)

print(
    "Letztes Angebot:",
    angebote["angebotsdatum"].max()
)

print("\nANZAHL ANGEBOTE:")
print(len(angebote))

print("\nFEHLENDE WERTE:")
print(
    angebote[
        ["angebotsdatum", "angebotswert"]
    ].isnull().sum()
)


# ==================================================
# 3. MONATLICHES ANGEBOTSVOLUMEN
# ==================================================

monatsangebotswert = (
    angebote
    .dropna(
        subset=["angebotsdatum", "angebotswert"]
    )
    .set_index("angebotsdatum")["angebotswert"]
    .resample("MS")
    .sum()
)

print("\nMONATLICHER ANGEBOTSWERT:")
print(monatsangebotswert)


# ==================================================
# 4. TRAINING UND FINALER TEST
# ==================================================

train = monatsangebotswert[
    monatsangebotswert.index < "2026-01-01"
]

test = monatsangebotswert[
    (monatsangebotswert.index >= "2026-01-01") &
    (monatsangebotswert.index <= "2026-06-01")
]


print("\n======================================")
print("TRAINING / TEST")
print("======================================")

print("Trainingsmonate:", len(train))
print("Testmonate:", len(test))


# ==================================================
# 5. SAISONALE BASELINE
# ==================================================

baseline = train[
    (train.index >= "2025-01-01") &
    (train.index <= "2025-06-01")
].copy()

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


baseline_vergleich = pd.DataFrame({
    "Ist": test,
    "Baseline": baseline
})

baseline_vergleich["Abweichung_Euro"] = (
    baseline_vergleich["Ist"]
    - baseline_vergleich["Baseline"]
)

baseline_vergleich["Abweichung_Prozent"] = (
    baseline_vergleich["Abweichung_Euro"]
    / baseline_vergleich["Ist"]
) * 100


print("\n======================================")
print("BASELINE JAN-JUN 2026")
print("======================================")

print(
    baseline_vergleich.round(2)
)

print("\nBASELINE MODELLBEWERTUNG:")

print(
    f"MAE:  {baseline_mae:,.2f} Euro"
)

print(
    f"RMSE: {baseline_rmse:,.2f} Euro"
)

print(
    f"MAPE: {baseline_mape:.2f} %"
)


# ==================================================
# 6. SARIMA PARAMETEROPTIMIERUNG
# ==================================================

# 2026 bleibt komplett unangetastet.
# Parameterwahl erfolgt mit Daten bis Ende 2025.

train_opt = monatsangebotswert[
    monatsangebotswert.index < "2025-07-01"
]

validierung = monatsangebotswert[
    (monatsangebotswert.index >= "2025-07-01") &
    (monatsangebotswert.index <= "2025-12-01")
]


print("\n======================================")
print("SARIMA PARAMETEROPTIMIERUNG")
print("======================================")

print(
    "Trainingsmonate Optimierung:",
    len(train_opt)
)

print(
    "Validierungsmonate:",
    len(validierung)
)


# ==================================================
# 7. PARAMETERRASTER
# ==================================================

p = [0, 1, 2]
d = [0, 1]
q = [0, 1, 2]

P = [0, 1]
D = [0, 1]
Q = [0, 1]

parameter = list(
    itertools.product(
        p, d, q
    )
)

saisonale_parameter = list(
    itertools.product(
        P, D, Q
    )
)


# ==================================================
# 8. MODELLE TESTEN
# ==================================================

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


# ==================================================
# 9. BESTE MODELLE
# ==================================================

ergebnisse_df = pd.DataFrame(
    ergebnisse
)

ergebnisse_df = (
    ergebnisse_df
    .sort_values(
        by="MAE"
    )
)


print("\n======================================")
print("DIE 10 BESTEN SARIMA-MODELLE")
print("======================================")

print(
    ergebnisse_df
    .head(10)
    .to_string(index=False)
)# ==================================================
# FINALER SARIMA-TEST ANGEBOTE
# ==================================================

print("\n======================================")
print("FINALER SARIMA-TEST ANGEBOTE")
print("======================================")


# Training bis Dezember 2025
train_final = monatsangebotswert[
    monatsangebotswert.index < "2026-01-01"
]

# Finaler Test Januar bis Juni 2026
test_final = monatsangebotswert[
    (monatsangebotswert.index >= "2026-01-01") &
    (monatsangebotswert.index <= "2026-06-01")
]


# ==================================================
# Bestes Modell aus der Validierung:
# SARIMA (2,1,2)(0,0,0,12)
# ==================================================

finales_modell = SARIMAX(
    train_final,
    order=(2, 1, 2),
    seasonal_order=(0, 0, 0, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)

finales_modell_fit = finales_modell.fit(
    disp=False,
    maxiter=200
)


# ==================================================
# Prognose Jan-Jun 2026
# ==================================================

finale_prognose = finales_modell_fit.forecast(
    steps=len(test_final)
)


# ==================================================
# Abweichungsanalyse
# ==================================================

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


# ==================================================
# Modellbewertung
# ==================================================

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

print(
    f"MAE:  {final_mae:,.2f} Euro"
)

print(
    f"RMSE: {final_rmse:,.2f} Euro"
)

print(
    f"MAPE: {final_mape:.2f} %"
)


# ==================================================
# BASELINE VS SARIMA
# ==================================================

print("\n======================================")
print("BASELINE VS. SARIMA")
print("======================================")

print(
    f"Baseline MAPE: {baseline_mape:.2f} %"
)

print(
    f"SARIMA MAPE:   {final_mape:.2f} %"
)# ==================================================
# SARIMA FORECAST ANGEBOTE JUL-DEZ 2026
# ==================================================

print("\n======================================")
print("SARIMA FORECAST ANGEBOTE JUL-DEZ 2026")
print("======================================")


# Alle verfügbaren Ist-Daten bis Juni 2026
train_zukunft = monatsangebotswert[
    monatsangebotswert.index <= "2026-06-01"
]


# Final ausgewähltes SARIMA-Modell
zukunft_modell = SARIMAX(
    train_zukunft,
    order=(2, 1, 2),
    seasonal_order=(0, 0, 0, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)

zukunft_modell_fit = zukunft_modell.fit(
    disp=False,
    maxiter=200
)


# Jul-Dez 2026 prognostizieren
forecast_zukunft = zukunft_modell_fit.forecast(
    steps=6
)


print("\nPROGNOSE ANGEBOTSWERT JUL-DEZ 2026:")

for datum, wert in forecast_zukunft.items():
    print(
        datum.strftime("%Y-%m"),
        f"{wert:,.2f} Euro"
    )