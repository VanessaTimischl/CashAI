import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import itertools
# Excel-Datei einlesen
df = pd.read_excel("Angebote/Angebot_Aufbereitet.xlsx")

# Datumsfelder umwandeln
df["angebotsdatum"] = pd.to_datetime(
    df["angebotsdatum"],
     format="%m/%d/%Y",
    errors="coerce"
)

df["entscheidungsdatum"] = pd.to_datetime(
    df["entscheidungsdatum"],
    format="%m/%d/%Y",
    errors="coerce"
)

# Datentypen kontrollieren
print("DATENTYPEN NACH DER UMWANDLUNG:")
print(df.dtypes)

# Fehlende Werte kontrollieren
print("\nFEHLENDE WERTE:")
print(df.isnull().sum())
print("\nSTATUS:")
print(df["status"].value_counts(dropna=False))
print("\nZEITRAUM:")
print("Erstes Angebotsdatum:", df["angebotsdatum"].min())
print("Letztes Angebotsdatum:", df["angebotsdatum"].max())

print("Erstes Entscheidungsdatum:", df["entscheidungsdatum"].min())
print("Letztes Entscheidungsdatum:", df["entscheidungsdatum"].max())
# Nur gewonnene Angebote auswählen
gewonnen = df[df["status"] == "gewonnen"].copy()

print("\nANZAHL GEWONNENER ANGEBOTE:")
print(len(gewonnen))
# Gewonnene Angebote monatlich zusammenfassen
monatlich = (
    gewonnen
    .set_index("entscheidungsdatum")
    .resample("MS")["angebotswert"]
    .sum()
)

print("\nMONATLICH GEWONNENER ANGEBOTSWERT:")
print(monatlich)
# Zeitreihe grafisch darstellen
plt.figure(figsize=(12, 6))

plt.plot(
    monatlich.index,
    monatlich.values,
    marker="o"
)

plt.title("Monatlich gewonnener Angebotswert")
plt.xlabel("Monat")
plt.ylabel("Gewonnener Angebotswert in Euro")

plt.grid(True)
plt.tight_layout()

plt.show()
# Trainingsdaten: 2023 bis 2025
train = monatlich[monatlich.index < "2026-01-01"]

# Testdaten: Jahr 2026
test = monatlich[monatlich.index >= "2026-01-01"]

print("\nTRAININGSDATEN:")
print(train)

print("\nTESTDATEN:")
print(test)

print("\nANZAHL TRAININGSMONATE:")
print(len(train))

print("\nANZAHL TESTMONATE:")
print(len(test))
# Erstes SARIMA-Modell erstellen
model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12)
)

# Modell trainieren
model_fit = model.fit()

print("\nSARIMA-MODELL WURDE TRAINIERT")
# 12 Monate prognostizieren
forecast = model_fit.forecast(steps=len(test))

print("\nSARIMA-PROGNOSE FÜR 2026:")
print(forecast)
pd.options.display.float_format = "{:,.2f}".format
# Vergleich zwischen tatsächlichen Werten und SARIMA-Prognose

plt.figure(figsize=(12, 6))

plt.plot(
    train.index,
    train.values,
    label="Training"
)

plt.plot(
    test.index,
    test.values,
    marker="o",
    label="Tatsächliche Werte 2026"
)

plt.plot(
    forecast.index,
    forecast.values,
    marker="o",
    label="SARIMA-Prognose 2026"
)

plt.title("SARIMA-Prognose im Vergleich zu den tatsächlichen Werten")
plt.xlabel("Monat")
plt.ylabel("Gewonnener Angebotswert in Euro")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()# Prognosefehler berechnen

mae = mean_absolute_error(test, forecast)

rmse = np.sqrt(
    mean_squared_error(test, forecast)
)

print("\nMODELLBEWERTUNG:")
print(f"MAE: {mae:,.2f} Euro")
print(f"RMSE: {rmse:,.2f} Euro")# --------------------------------------------------
# --------------------------------------------------
# SARIMA GRID SEARCH
# --------------------------------------------------

# Mögliche Werte für p, d und q
p = [0, 1, 2]
d = [0, 1]
q = [0, 1, 2]

# Alle Kombinationen erzeugen
pdq = list(itertools.product(p, d, q))

# Mögliche saisonale Parameter
seasonal_pdq = [
    (P, D, Q, 12)
    for P in [0, 1]
    for D in [0, 1]
    for Q in [0, 1]
]
ergebnisse = []
for order in pdq:

    for seasonal_order in seasonal_pdq:

        try:

            model = SARIMAX(
                train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            model_fit = model.fit(disp=False)

            forecast_test = model_fit.forecast(
                steps=len(test)
            )

            mae_test = mean_absolute_error(
                test,
                forecast_test
            )

            rmse_test = np.sqrt(
                mean_squared_error(
                    test,
                    forecast_test
                )
            )

            ergebnisse.append([
                order,
                seasonal_order,
                mae_test,
                rmse_test
            ])

        except:
            continue
        ergebnisse_df = pd.DataFrame(
    ergebnisse,
    columns=[
        "order",
        "seasonal_order",
        "MAE",
        "RMSE"
    ]
)
ergebnisse_df = ergebnisse_df.sort_values(
    by="MAE"
)
print("\nDIE 10 BESTEN SARIMA-MODELLE:")

print(
    ergebnisse_df.head(10).to_string(index=False)
)
# # --------------------------------------------------
# # BESTES SARIMA-MODELL
# # --------------------------------------------------

# bestes_modell = SARIMAX(
#     train,
#     order=(2, 1, 2),
#     seasonal_order=(1, 1, 1, 12),
#     enforce_stationarity=False,
#     enforce_invertibility=False
# )

# bestes_modell_fit = bestes_modell.fit(disp=False)

# # Prognose für die 12 Testmonate
# beste_prognose = bestes_modell_fit.forecast(
#     steps=len(test)
# )

# print("\nPROGNOSE DES BESTEN SARIMA-MODELLS:")
# print(beste_prognose)
# --------------------------------------------------
# BESTES SARIMA-MODELL
# --------------------------------------------------

bestes_modell = SARIMAX(
    train,
    order=(2, 1, 2),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)

# Modell trainieren
bestes_modell_fit = bestes_modell.fit(disp=False)

# Prognose für 2026 erstellen
beste_prognose = bestes_modell_fit.forecast(
    steps=len(test)
)

print("\nPROGNOSE DES BESTEN SARIMA-MODELLS:")
print(beste_prognose)


# --------------------------------------------------
# IST-WERTE MIT PROGNOSE VERGLEICHEN
# --------------------------------------------------

vergleich = pd.DataFrame({
    "Tatsaechlich": test,
    "SARIMA_Prognose": beste_prognose
})

vergleich["Abweichung"] = (
    vergleich["Tatsaechlich"]
    - vergleich["SARIMA_Prognose"]
)

vergleich["Absolute_Abweichung"] = (
    vergleich["Abweichung"].abs()
)

print("\nVERGLEICH IST VS. PROGNOSE:")
print(vergleich)


# --------------------------------------------------
# MAE UND RMSE DES BESTEN MODELLS
# --------------------------------------------------

mae_best = mean_absolute_error(
    test,
    beste_prognose
)

rmse_best = np.sqrt(
    mean_squared_error(
        test,
        beste_prognose
    )
)

print("\nMODELLBEWERTUNG BESTES SARIMA:")
print(f"MAE: {mae_best:,.2f} Euro")
print(f"RMSE: {rmse_best:,.2f} Euro")


# --------------------------------------------------
# GRAFISCHE DARSTELLUNG
# --------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    train.index,
    train.values,
    label="Training"
)

plt.plot(
    test.index,
    test.values,
    marker="o",
    label="Tatsächliche Werte 2026"
)

plt.plot(
    beste_prognose.index,
    beste_prognose.values,
    marker="o",
    label="SARIMA-Prognose 2026"
)

plt.title("SARIMA-Prognose – bestes Modell")
plt.xlabel("Monat")
plt.ylabel("Gewonnener Angebotswert in Euro")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()
# --------------------------------------------------
# BASELINE: GLEICHER MONAT DES VORJAHRES
# --------------------------------------------------

baseline = monatlich.shift(12).loc[test.index]

print("\nBASELINE-PROGNOSE:")
print(baseline)
mae_baseline = mean_absolute_error(
    test,
    baseline
)

rmse_baseline = np.sqrt(
    mean_squared_error(
        test,
        baseline
    )
)

print("\nMODELLBEWERTUNG BASELINE:")
print(f"MAE: {mae_baseline:,.2f} Euro")
print(f"RMSE: {rmse_baseline:,.2f} Euro")
# --------------------------------------------------
# WALK-FORWARD-VALIDIERUNG VORBEREITEN
# --------------------------------------------------

entwicklung = monatlich[monatlich.index < "2026-01-01"]

training_start = entwicklung[
    entwicklung.index < "2025-01-01"
]

validierung = entwicklung[
    entwicklung.index >= "2025-01-01"
]

print("\nWALK-FORWARD VORBEREITUNG:")
print("Anfängliche Trainingsmonate:", len(training_start))
print("Validierungsmonate:", len(validierung))
# --------------------------------------------------
# WALK-FORWARD-VALIDIERUNG
# --------------------------------------------------

historie = training_start.copy()

walk_forward_prognosen = []

for datum in validierung.index:

    modell = SARIMAX(
        historie,
        order=(1, 1, 1),
        seasonal_order=(0, 0, 0, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    modell_fit = modell.fit(disp=False)

    prognose = modell_fit.forecast(steps=1)

    prognose_wert = prognose.iloc[0]

    walk_forward_prognosen.append(prognose_wert)

    # Tatsächlichen Wert anschließend zur Historie hinzufügen
    neuer_wert = pd.Series(
        [validierung.loc[datum]],
        index=[datum]
    )

    historie = pd.concat([
        historie,
        neuer_wert
    ])

# Python
__pycache__/
*.pyc

# VS Code
.vscode/

# Datendateien nicht auf GitHub hochladen
*.xlsx
*.csv
