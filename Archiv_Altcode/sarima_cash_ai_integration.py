# ============================================================
# CASH-AI: SARIMA auf Basis des bestehenden Datenmodells
# ============================================================
#
# Dieses Skript verwendet NICHT erneut die Rohdaten.
# Es liest direkt die bereits aufbereiteten wöchentlichen
# CASH-AI-Datensätze:
#
#   09_Cashflow_Woechentlich_Training.csv
#   10_Cashflow_Woechentlich_Validierung.csv
#
# Cash-In und Cash-Out werden getrennt modelliert.
# Danach wird der prognostizierte Netto-Cashflow berechnet.
#
# Empfohlener Speicherort:
#   C:\CashAI\sarima_cash_ai_integration.py
#
# Das Skript sucht den Ergebnisordner automatisch in typischen
# Projektstrukturen. Alternativ DATA_DIR unten manuell setzen.
# ============================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


# ============================================================
# 1. PFADE
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Falls die automatische Suche nicht funktioniert, hier z. B.:
# DATA_DIR = Path(r"C:\CASH_AI_Python_Paket\Ergebnisse_CASH_AI")
DATA_DIR = Path(r"C:\Ergebniss_XGBoost")


OUTPUT_DIR = SCRIPT_DIR / "Ergebnisse" / "SARIMA_CASH_AI"


def finde_datenordner():
    """Sucht den Ergebnisordner des bestehenden CASH-AI-Datenmodells."""

    kandidaten = [
        SCRIPT_DIR / "Ergebnisse_CASH_AI",
        SCRIPT_DIR.parent / "Ergebnisse_CASH_AI",
        SCRIPT_DIR.parent / "CASH_AI_Python_Paket" / "Ergebnisse_CASH_AI",
        Path(r"C:\CASH_AI_Python_Paket\Ergebnisse_CASH_AI"),
    ]

    for ordner in kandidaten:
        if (
            (ordner / "09_Cashflow_Woechentlich_Training.csv").exists()
            and
            (ordner / "10_Cashflow_Woechentlich_Validierung.csv").exists()
        ):
            return ordner

    raise FileNotFoundError(
        "Der Ordner 'Ergebnisse_CASH_AI' wurde nicht gefunden.\n"
        "Setze DATA_DIR am Anfang des Skripts manuell, z. B.:\n"
        r'DATA_DIR = Path(r"C:\CASH_AI_Python_Paket\Ergebnisse_CASH_AI")'
    )


if DATA_DIR is None:
    DATA_DIR = finde_datenordner()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "09_Cashflow_Woechentlich_Training.csv"
VALID_FILE = DATA_DIR / "10_Cashflow_Woechentlich_Validierung.csv"


# ============================================================
# 2. EINSTELLUNGEN
# ============================================================

DATE_COLUMN = "periodenbeginn"
CASH_IN_COLUMN = "ziel_kundeneinzahlungen"
CASH_OUT_COLUMN = "ziel_beschaffungsauszahlungen"

# 52 = ungefähr jährliche Saisonalität bei Wochenwerten
SEASON_LENGTH = 52

# Kleines, kontrolliertes Parameterraster.
# Dadurch bleibt die Laufzeit überschaubar.
ORDERS = [
    (0, 1, 1),
    (1, 0, 0),
    (1, 1, 0),
    (1, 1, 1),
    (2, 1, 0),
    (0, 1, 2),
]

SEASONAL_ORDERS = [
    (0, 0, 0, SEASON_LENGTH),
    (1, 0, 0, SEASON_LENGTH),
    (0, 1, 1, SEASON_LENGTH),
    (1, 0, 1, SEASON_LENGTH),
]


# ============================================================
# 3. DATEN LADEN
# ============================================================

def lade_cashflow_datei(pfad):
    """Lädt die CASH-AI-CSV mit deutscher Zahlenformatierung."""

    df = pd.read_csv(
        pfad,
        sep=";",
        decimal=",",
        encoding="utf-8-sig"
    )

    if DATE_COLUMN not in df.columns:
        raise ValueError(
            f"Spalte '{DATE_COLUMN}' fehlt in {pfad.name}."
        )

    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN],
        errors="coerce"
    )

    for spalte in [CASH_IN_COLUMN, CASH_OUT_COLUMN]:
        if spalte not in df.columns:
            raise ValueError(
                f"Spalte '{spalte}' fehlt in {pfad.name}."
            )

        df[spalte] = pd.to_numeric(
            df[spalte],
            errors="coerce"
        )

    # Nur vom bestehenden Datenmodell freigegebene Wochen.
    if "fuer_spaeteres_supervised_learning_verwendbar" in df.columns:
        df = df[
            df["fuer_spaeteres_supervised_learning_verwendbar"] == 1
        ].copy()

    if "ziel_beobachtbar" in df.columns:
        df = df[
            df["ziel_beobachtbar"] == 1
        ].copy()

    if "woche_vollstaendig" in df.columns:
        df = df[
            df["woche_vollstaendig"] == 1
        ].copy()

    df = (
        df
        .dropna(
            subset=[
                DATE_COLUMN,
                CASH_IN_COLUMN,
                CASH_OUT_COLUMN
            ]
        )
        .sort_values(DATE_COLUMN)
    )

    return df


train_df = lade_cashflow_datei(TRAIN_FILE)
valid_df = lade_cashflow_datei(VALID_FILE)

print("\n======================================")
print("CASH-AI DATENBASIS")
print("======================================")
print("Datenordner:", DATA_DIR)
print("Training:", len(train_df), "Wochen")
print("Validierung:", len(valid_df), "Wochen")
print(
    "Trainingszeitraum:",
    train_df[DATE_COLUMN].min().date(),
    "bis",
    train_df[DATE_COLUMN].max().date()
)
print(
    "Validierungszeitraum:",
    valid_df[DATE_COLUMN].min().date(),
    "bis",
    valid_df[DATE_COLUMN].max().date()
)


# ============================================================
# 4. ZEITREIHEN ERSTELLEN
# ============================================================

def erstelle_serie(df, zielspalte, name):
    serie = (
        df
        .set_index(DATE_COLUMN)[zielspalte]
        .astype(float)
        .sort_index()
    )

    serie.name = name

    # Prüfen, ob die Wochen lückenlos im 7-Tage-Abstand vorliegen.
    differenzen = serie.index.to_series().diff().dropna()

    if not differenzen.eq(pd.Timedelta(days=7)).all():
        raise ValueError(
            f"Die Zeitreihe '{name}' enthält Wochenlücken."
        )

    return serie


train_cash_in = erstelle_serie(
    train_df,
    CASH_IN_COLUMN,
    "Cash_In"
)

valid_cash_in = erstelle_serie(
    valid_df,
    CASH_IN_COLUMN,
    "Cash_In"
)

train_cash_out = erstelle_serie(
    train_df,
    CASH_OUT_COLUMN,
    "Cash_Out"
)

valid_cash_out = erstelle_serie(
    valid_df,
    CASH_OUT_COLUMN,
    "Cash_Out"
)


# ============================================================
# 5. METRIKEN
# ============================================================

def berechne_metriken(ist, prognose):
    ist = np.asarray(ist, dtype=float)
    prognose = np.asarray(prognose, dtype=float)

    mae = mean_absolute_error(ist, prognose)

    rmse = np.sqrt(
        mean_squared_error(ist, prognose)
    )

    nenner = np.abs(ist).sum()

    if nenner == 0:
        wape = np.nan
    else:
        wape = (
            np.abs(ist - prognose).sum()
            / nenner
            * 100
        )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE_Prozent": wape
    }


# ============================================================
# 6. SARIMA-PARAMETER AUSWÄHLEN
# ============================================================

def finde_bestes_sarima(train, valid, reihenname):
    ergebnisse = []

    print("\n======================================")
    print("PARAMETEROPTIMIERUNG:", reihenname)
    print("======================================")

    for order in ORDERS:

        for seasonal_order in SEASONAL_ORDERS:

            try:
                modell = SARIMAX(
                    train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )

                fit = modell.fit(
                    disp=False,
                    maxiter=300
                )

                prognose = fit.forecast(
                    steps=len(valid)
                )

                metriken = berechne_metriken(
                    valid.values,
                    prognose.values
                )

                ergebnisse.append({
                    "Datenreihe": reihenname,
                    "order": str(order),
                    "seasonal_order": str(seasonal_order),
                    "MAE": metriken["MAE"],
                    "RMSE": metriken["RMSE"],
                    "WAPE_Prozent": metriken["WAPE_Prozent"],
                    "AIC": fit.aic
                })

            except Exception as fehler:
                print(
                    "Übersprungen:",
                    order,
                    seasonal_order,
                    "|",
                    str(fehler)[:100]
                )

    if not ergebnisse:
        raise RuntimeError(
            f"Für {reihenname} konnte kein SARIMA-Modell geschätzt werden."
        )

    ergebnisse_df = pd.DataFrame(ergebnisse)

    ergebnisse_df = (
        ergebnisse_df
        .sort_values(
            ["MAE", "RMSE", "AIC"]
        )
        .reset_index(drop=True)
    )

    bestes_order = eval(
        ergebnisse_df.loc[0, "order"]
    )

    bestes_seasonal_order = eval(
        ergebnisse_df.loc[0, "seasonal_order"]
    )

    print("\nBeste 5 Modelle:")
    print(
        ergebnisse_df
        .head(5)
        .round(2)
        .to_string(index=False)
    )

    print("\nAusgewählt:")
    print("order =", bestes_order)
    print("seasonal_order =", bestes_seasonal_order)

    return (
        bestes_order,
        bestes_seasonal_order,
        ergebnisse_df
    )


(
    cash_in_order,
    cash_in_seasonal,
    cash_in_grid
) = finde_bestes_sarima(
    train_cash_in,
    valid_cash_in,
    "Cash-In"
)

(
    cash_out_order,
    cash_out_seasonal,
    cash_out_grid
) = finde_bestes_sarima(
    train_cash_out,
    valid_cash_out,
    "Cash-Out"
)


# ============================================================
# 7. VALIDIERUNGSPROGNOSE
# ============================================================

def validierungsprognose(
    train,
    valid,
    order,
    seasonal_order
):
    modell = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fit = modell.fit(
        disp=False,
        maxiter=300
    )

    prognose = fit.forecast(
        steps=len(valid)
    )

    prognose.index = valid.index

    return prognose


valid_forecast_in = validierungsprognose(
    train_cash_in,
    valid_cash_in,
    cash_in_order,
    cash_in_seasonal
)

valid_forecast_out = validierungsprognose(
    train_cash_out,
    valid_cash_out,
    cash_out_order,
    cash_out_seasonal
)


validierung = pd.DataFrame({
    "Ist_Cash_In": valid_cash_in,
    "Forecast_Cash_In": valid_forecast_in,
    "Ist_Cash_Out": valid_cash_out,
    "Forecast_Cash_Out": valid_forecast_out
})

validierung["Ist_Netto_Cashflow"] = (
    validierung["Ist_Cash_In"]
    - validierung["Ist_Cash_Out"]
)

validierung["Forecast_Netto_Cashflow"] = (
    validierung["Forecast_Cash_In"]
    - validierung["Forecast_Cash_Out"]
)


# ============================================================
# 8. VALIDIERUNGSMETRIKEN
# ============================================================

metriken_in = berechne_metriken(
    validierung["Ist_Cash_In"],
    validierung["Forecast_Cash_In"]
)

metriken_out = berechne_metriken(
    validierung["Ist_Cash_Out"],
    validierung["Forecast_Cash_Out"]
)

metriken_netto = berechne_metriken(
    validierung["Ist_Netto_Cashflow"],
    validierung["Forecast_Netto_Cashflow"]
)


modellvergleich = pd.DataFrame([
    {
        "Bereich": "Cash-In",
        "order": str(cash_in_order),
        "seasonal_order": str(cash_in_seasonal),
        **metriken_in
    },
    {
        "Bereich": "Cash-Out",
        "order": str(cash_out_order),
        "seasonal_order": str(cash_out_seasonal),
        **metriken_out
    },
    {
        "Bereich": "Netto-Cashflow",
        "order": "-",
        "seasonal_order": "-",
        **metriken_netto
    }
])


print("\n======================================")
print("SARIMA VALIDIERUNGSERGEBNISSE")
print("======================================")
print(
    modellvergleich
    .round(2)
    .to_string(index=False)
)


# ============================================================
# 9. FINALES TRAINING MIT ALLEN BEKANNTEN WOCHEN
# ============================================================

gesamt_cash_in = pd.concat([
    train_cash_in,
    valid_cash_in
]).sort_index()

gesamt_cash_out = pd.concat([
    train_cash_out,
    valid_cash_out
]).sort_index()


def finales_modell_fit(
    serie,
    order,
    seasonal_order
):
    modell = SARIMAX(
        serie,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    return modell.fit(
        disp=False,
        maxiter=300
    )


final_fit_in = finales_modell_fit(
    gesamt_cash_in,
    cash_in_order,
    cash_in_seasonal
)

final_fit_out = finales_modell_fit(
    gesamt_cash_out,
    cash_out_order,
    cash_out_seasonal
)


# ============================================================
# 10. JULI-DEZEMBER 2026 PROGNOSTIZIEREN
# ============================================================

# Die vorhandenen Wochen laufen Mittwoch bis Dienstag.
# Die letzte beobachtete Woche beginnt am 24.06.2026.
# Daher startet die nächste Woche am 01.07.2026.
#
# 26 vollständige Wochen bis einschließlich Woche 23.-29.12.2026.

FORECAST_WEEKS = 26

forecast_index = pd.date_range(
    start=pd.Timestamp("2026-07-01"),
    periods=FORECAST_WEEKS,
    freq="7D"
)


forecast_in = final_fit_in.forecast(
    steps=FORECAST_WEEKS
)

forecast_out = final_fit_out.forecast(
    steps=FORECAST_WEEKS
)

forecast_in.index = forecast_index
forecast_out.index = forecast_index


forecast_2026 = pd.DataFrame({
    "periodenbeginn": forecast_index,
    "periodenende": forecast_index + pd.Timedelta(days=6),
    "SARIMA_Cash_In_roh": forecast_in.values,
    "SARIMA_Cash_Out_roh": forecast_out.values
})


# Negative Ein-/Auszahlungen sind operativ nicht sinnvoll.
# Deshalb werden zusätzlich auf 0 begrenzte Werte ausgegeben.
# Die Modellbewertung oben bleibt jedoch UNGECLIPPT.

forecast_2026["SARIMA_Cash_In"] = (
    forecast_2026["SARIMA_Cash_In_roh"]
    .clip(lower=0)
)

forecast_2026["SARIMA_Cash_Out"] = (
    forecast_2026["SARIMA_Cash_Out_roh"]
    .clip(lower=0)
)

forecast_2026["SARIMA_Netto_Cashflow"] = (
    forecast_2026["SARIMA_Cash_In"]
    - forecast_2026["SARIMA_Cash_Out"]
)


forecast_2026["Jahr"] = (
    forecast_2026["periodenbeginn"]
    .dt.isocalendar()
    .year
    .astype(int)
)

forecast_2026["KW"] = (
    forecast_2026["periodenbeginn"]
    .dt.isocalendar()
    .week
    .astype(int)
)


print("\n======================================")
print("SARIMA FORECAST JUL-DEZ 2026")
print("======================================")
print(
    forecast_2026[
        [
            "periodenbeginn",
            "SARIMA_Cash_In",
            "SARIMA_Cash_Out",
            "SARIMA_Netto_Cashflow"
        ]
    ]
    .round(2)
    .to_string(index=False)
)


# ============================================================
# 11. ERGEBNISSE SPEICHERN
# ============================================================

modellvergleich.to_csv(
    OUTPUT_DIR / "sarima_modellvergleich.csv",
    sep=";",
    decimal=",",
    index=False,
    encoding="utf-8-sig"
)

forecast_2026.to_csv(
    OUTPUT_DIR / "sarima_forecast_2026.csv",
    sep=";",
    decimal=",",
    index=False,
    encoding="utf-8-sig"
)

cash_in_grid.to_csv(
    OUTPUT_DIR / "sarima_grid_cash_in.csv",
    sep=";",
    decimal=",",
    index=False,
    encoding="utf-8-sig"
)

cash_out_grid.to_csv(
    OUTPUT_DIR / "sarima_grid_cash_out.csv",
    sep=";",
    decimal=",",
    index=False,
    encoding="utf-8-sig"
)


excel_path = OUTPUT_DIR / "SARIMA_CASH_AI_Ergebnisse.xlsx"

with pd.ExcelWriter(
    excel_path,
    engine="openpyxl"
) as writer:

    modellvergleich.to_excel(
        writer,
        sheet_name="Modellvergleich",
        index=False
    )

    validierung.to_excel(
        writer,
        sheet_name="Validierung",
        index=True
    )

    forecast_2026.to_excel(
        writer,
        sheet_name="Forecast_Jul_Dez_2026",
        index=False
    )

    cash_in_grid.to_excel(
        writer,
        sheet_name="Grid_Cash_In",
        index=False
    )

    cash_out_grid.to_excel(
        writer,
        sheet_name="Grid_Cash_Out",
        index=False
    )


# ============================================================
# 12. GRAFIK
# ============================================================

plt.figure(
    figsize=(14, 7)
)

plt.plot(
    forecast_2026["periodenbeginn"],
    forecast_2026["SARIMA_Cash_In"],
    marker="o",
    label="SARIMA Cash-In"
)

plt.plot(
    forecast_2026["periodenbeginn"],
    forecast_2026["SARIMA_Cash_Out"],
    marker="o",
    label="SARIMA Cash-Out"
)

plt.plot(
    forecast_2026["periodenbeginn"],
    forecast_2026["SARIMA_Netto_Cashflow"],
    marker="o",
    label="SARIMA Netto-Cashflow"
)

plt.axhline(
    y=0,
    linewidth=1
)

plt.title(
    "SARIMA Cashflow-Forecast Juli bis Dezember 2026"
)

plt.xlabel(
    "Wochenbeginn"
)

plt.ylabel(
    "Euro"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

grafik_path = (
    OUTPUT_DIR
    / "SARIMA_Cashflow_Forecast_2026.png"
)

plt.savefig(
    grafik_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 13. ABSCHLUSS
# ============================================================

print("\n======================================")
print("FERTIG")
print("======================================")

print("Ergebnisordner:")
print(OUTPUT_DIR)

print("\nExcel:")
print(excel_path)

print("\nGrafik:")
print(grafik_path)
