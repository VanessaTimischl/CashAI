#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CASH-AI – gemeinsames Update für XGBoost + SARIMA.

Ziel:
1) Die bisherigen 6 Validierungswochen werden nach abgeschlossener Modellwahl
   wieder in die finale Trainingshistorie aufgenommen.
2) XGBoost und SARIMA werden auf derselben vollständigen wöchentlichen
   CASH-AI-Datenbasis bis zum jeweils aktuellen Datenstichtag neu gefittet.
3) Beide Modelle erzeugen Cash-In, Cash-Out und Netto-Cashflow.
4) Ausgabeebenen:
   - täglich (aus dem nativen Wochenforecast anhand historischer Wochentagsanteile abgeleitet)
   - wöchentlich (native Modellprognose)
   - monatlich (Summe der Tagesprognosen)
5) Bereits realisierte Cashflows werden ebenfalls exportiert.
6) Eine stabile Power-BI-Datei wird bei jedem Lauf überschrieben/aktualisiert.
7) Automatischer rollierender Backtest für XGBoost und SARIMA.
8) Modellgüte MAE / RMSE / MAPE / WAPE für:
   - Cash-In / Cash-Out / Netto-Cashflow
   - täglich / wöchentlich / monatlich
9) Direkter Power-BI-Export der Backtest-Details und Modellgüte.
10) Optional kann vor dem Modelllauf die CASH_AI_Datenaufbereitung.py gestartet werden.

Empfohlener Speicherort:
    C:\\CashAI\\UPDATE_CASH_AI_MODELS.py

Standardpfade:
    Datenmodell-Ausgaben: C:\\Ergebniss_XGBoost
    CASH-AI-Datenaufbereitung: C:\\CASH_AI_Python_Paket\\CASH_AI_Datenaufbereitung.py
    Rohdaten: C:\\CASH_AI_Python_Paket\\Eingabedaten
    Ergebnisse: C:\\CashAI\\Ergebnisse\\AUTO_UPDATE
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. STANDARDKONFIGURATION
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = Path(r"C:\Ergebniss_XGBoost")
DEFAULT_CASH_AI_DIR = Path(r"C:\CASH_AI_Python_Paket")
DEFAULT_CASHAI_REPO = Path(r"C:\CashAI")
DEFAULT_OUTPUT_DIR = DEFAULT_CASHAI_REPO / "Ergebnisse" / "AUTO_UPDATE"

# Bereits ausgewählte XGBoost-Konfiguration aus dem bisherigen Projektstand.
# Falls ihr später die exakten finalen Zusatzparameter festlegt, hier ergänzen.
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}

# SARIMA: kompakter Suchraum nur beim ersten Lauf bzw. mit --reoptimize-sarima.
SARIMA_ORDERS = [
    (0, 0, 0),
    (1, 0, 0),
    (0, 0, 1),
    (0, 1, 1),
    (1, 1, 0),
    (1, 1, 1),
    (2, 0, 0),
]
SARIMA_SEASONAL_ORDERS = [
    (0, 0, 0, 52),
    (1, 0, 0, 52),
    (0, 0, 1, 52),
    (1, 0, 1, 52),
]
SARIMA_INNER_VALIDATION_WEEKS = 26

TARGET_IN = "ziel_kundeneinzahlungen"
TARGET_OUT = "ziel_beschaffungsauszahlungen"
TARGET_NET = "ziel_netto_cashflow"


# ---------------------------------------------------------------------------
# 2. ARGUMENTE
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finales CASH-AI Update: Datenbasis + XGBoost + SARIMA + Power BI"
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Ordner mit 07/08/09/10 und 15_CASH_AI_Modellspalten.json",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Zielordner für stabile Modell- und Power-BI-Ergebnisse",
    )
    parser.add_argument(
        "--forecast-weeks",
        type=int,
        default=26,
        help="Anzahl zukünftiger Wochen (Standard 26 = ca. 6 Monate)",
    )
    parser.add_argument(
        "--reoptimize-sarima",
        action="store_true",
        help="SARIMA-Parameter neu auswählen; sonst gespeicherte Parameter wiederverwenden",
    )
    parser.add_argument(
        "--backtest-weeks",
        type=int,
        default=52,
        help=(
            "Anzahl historischer Wochen für den rollierenden 1-Schritt-Backtest "
            "(Standard: 52 = ca. 1 Jahr)"
        ),
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Backtest und Modellgüte aus Zeitgründen überspringen",
    )
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Vor dem Modelllauf CASH_AI_Datenaufbereitung.py ausführen",
    )
    parser.add_argument(
        "--cash-ai-dir",
        default=str(DEFAULT_CASH_AI_DIR),
        help="Ordner mit CASH_AI_Datenaufbereitung.py und Eingabedaten",
    )
    parser.add_argument(
        "--datenstichtag",
        help="Nur mit --prepare-data: neuer Stichtag YYYY-MM-DD",
    )
    parser.add_argument(
        "--data-mode",
        choices=["dummy", "real"],
        default="dummy",
        help="Datenmodus für die Datenaufbereitung",
    )
    parser.add_argument(
        "--date-order",
        choices=["monthfirst", "dayfirst", "auto"],
        default="monthfirst",
        help="Datumsformat für die Datenaufbereitung",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 3. DATEIEN EINLESEN
# ---------------------------------------------------------------------------

def read_cashai_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Datei fehlt: {path}")
    return pd.read_csv(
        path,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )


def load_model_columns(data_dir: Path) -> list[str]:
    path = data_dir / "15_CASH_AI_Modellspalten.json"
    if not path.exists():
        raise FileNotFoundError(f"Modellspalten-Datei fehlt: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload["eingangsmerkmale"]["Cashflow_Woechentlich"]


def combine_training_and_validation(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    date_col: str,
) -> pd.DataFrame:
    """
    Finales Refit:
    Der historische Validierungszeitraum wird nach abgeschlossener Modellwahl
    wieder Teil der Trainingshistorie.
    """
    df = pd.concat([train, valid], ignore_index=True)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    if "ziel_beobachtbar" in df.columns:
        df = df[df["ziel_beobachtbar"].eq(1)]
    if "fuer_spaeteres_supervised_learning_verwendbar" in df.columns:
        df = df[df["fuer_spaeteres_supervised_learning_verwendbar"].eq(1)]
    if "woche_vollstaendig" in df.columns:
        df = df[df["woche_vollstaendig"].eq(1)]

    return (
        df.dropna(subset=[date_col, TARGET_IN, TARGET_OUT, TARGET_NET])
        .sort_values(date_col)
        .drop_duplicates(subset=[date_col], keep="last")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 4. OPTIONAL: DATENMODELL VORHER NEU AUSFÜHREN
# ---------------------------------------------------------------------------

def run_data_preparation(args: argparse.Namespace, data_dir: Path) -> None:
    if not args.prepare_data:
        return

    if not args.datenstichtag:
        raise ValueError(
            "--prepare-data benötigt zusätzlich --datenstichtag YYYY-MM-DD."
        )

    cash_ai_dir = Path(args.cash_ai_dir)
    script = cash_ai_dir / "CASH_AI_Datenaufbereitung.py"
    input_dir = cash_ai_dir / "Eingabedaten"

    if not script.exists():
        raise FileNotFoundError(f"Datenaufbereitung nicht gefunden: {script}")
    if not input_dir.exists():
        raise FileNotFoundError(f"Eingabeordner nicht gefunden: {input_dir}")

    data_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(script),
        "--input-folder", str(input_dir),
        "--output-folder", str(data_dir),
        "--data-mode", args.data_mode,
        "--datenstichtag", args.datenstichtag,
        "--validierungstage", "42",
        "--date-order", args.date_order,
    ]

    print("\n" + "=" * 72)
    print("1/7 CASH-AI DATENMODELL AKTUALISIEREN")
    print("=" * 72)
    print(" ".join(command))
    subprocess.run(command, cwd=cash_ai_dir, check=True)


# ---------------------------------------------------------------------------
# 5. XGBOOST – FINALES REFIT AUF ALLEN VERWENDBAREN AKTUELLEN WOCHEN
# ---------------------------------------------------------------------------

def fit_xgboost_models(
    weekly_full: pd.DataFrame,
    feature_columns: list[str],
):
    missing = [c for c in feature_columns if c not in weekly_full.columns]
    if missing:
        raise ValueError(f"XGBoost-Eingangsmerkmale fehlen: {missing}")

    X = (
        weekly_full[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .astype(float)
    )

    model_in = XGBRegressor(**XGB_PARAMS)
    model_out = XGBRegressor(**XGB_PARAMS)

    model_in.fit(X, weekly_full[TARGET_IN].astype(float))
    model_out.fit(X, weekly_full[TARGET_OUT].astype(float))

    return model_in, model_out


def feature_from_history(
    history: pd.DataFrame,
    next_start: pd.Timestamp,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Rekonstruiert exakt die für Cashflow_Woechentlich freigegebenen
    Kalender-, Lag- und Rolling-Features aus der bis dahin bekannten Historie.
    Rolling-Fenster schließen die zu prognostizierende Woche aus.
    """
    row: dict[str, float] = {
        "tage_enthalten": 7.0,
        "periodenjahr": float(next_start.year),
        "periodenmonat": float(next_start.month),
        "periodenquartal": float(next_start.quarter),
        "woche_vollstaendig": 1.0,
        "hist_verfuegbare_wochen": float(len(history)),
    }

    mappings = {
        TARGET_IN: "kundeneinzahlungen",
        TARGET_OUT: "beschaffungsauszahlungen",
        TARGET_NET: "netto_cashflow",
    }

    for target, base in mappings.items():
        s = history[target].astype(float).reset_index(drop=True)

        for lag in [1, 7, 14, 30]:
            row[f"{base}_lag_{lag}w"] = (
                float(s.iloc[-lag]) if len(s) >= lag else np.nan
            )

        for window in [7, 14, 30]:
            values = s.tail(window)
            row[f"{base}_summe_letzte_{window}w"] = float(values.sum())
            row[f"{base}_mittel_letzte_{window}w"] = float(values.mean())
            row[f"{base}_std_letzte_{window}w"] = float(
                values.std(ddof=0)
            ) if len(values) else np.nan

    feature_row = pd.DataFrame([row])

    missing = [c for c in feature_columns if c not in feature_row.columns]
    if missing:
        raise ValueError(
            "Für den rekursiven XGBoost-Forecast können folgende "
            f"Features nicht gebildet werden: {missing}"
        )

    return (
        feature_row[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .astype(float)
    )


def recursive_xgboost_forecast(
    weekly_full: pd.DataFrame,
    feature_columns: list[str],
    model_in: XGBRegressor,
    model_out: XGBRegressor,
    forecast_weeks: int,
) -> pd.DataFrame:
    history = weekly_full[
        ["periodenbeginn", TARGET_IN, TARGET_OUT, TARGET_NET]
    ].copy()
    history["periodenbeginn"] = pd.to_datetime(history["periodenbeginn"])
    history = history.sort_values("periodenbeginn").reset_index(drop=True)

    last_start = history["periodenbeginn"].max()
    rows = []

    for step in range(1, forecast_weeks + 1):
        start = last_start + pd.Timedelta(days=7 * step)
        features = feature_from_history(history, start, feature_columns)

        cash_in = max(0.0, float(model_in.predict(features)[0]))
        cash_out = max(0.0, float(model_out.predict(features)[0]))
        net = cash_in - cash_out

        rows.append({
            "periodenbeginn": start,
            "periodenende": start + pd.Timedelta(days=6),
            "Cash_In": cash_in,
            "Cash_Out": cash_out,
            "Netto_Cashflow": net,
        })

        history = pd.concat(
            [
                history,
                pd.DataFrame([{
                    "periodenbeginn": start,
                    TARGET_IN: cash_in,
                    TARGET_OUT: cash_out,
                    TARGET_NET: net,
                }]),
            ],
            ignore_index=True,
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. SARIMA – PARAMETER SPEICHERN / WIEDERVERWENDEN
# ---------------------------------------------------------------------------

def select_sarima_params(
    series: pd.Series,
) -> tuple[tuple[int, int, int], tuple[int, int, int, int]]:
    if len(series) <= SARIMA_INNER_VALIDATION_WEEKS + 30:
        # Fallback bei kurzer Historie.
        return (1, 0, 0), (0, 0, 0, 52)

    train = series.iloc[:-SARIMA_INNER_VALIDATION_WEEKS]
    valid = series.iloc[-SARIMA_INNER_VALIDATION_WEEKS:]

    results = []

    for order in SARIMA_ORDERS:
        for seasonal_order in SARIMA_SEASONAL_ORDERS:
            try:
                fit = SARIMAX(
                    train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False, maxiter=200)

                pred = fit.forecast(steps=len(valid))
                mae = mean_absolute_error(valid.to_numpy(), pred.to_numpy())
                results.append((mae, fit.aic, order, seasonal_order))
            except Exception:
                continue

    if not results:
        return (1, 0, 0), (0, 0, 0, 52)

    results.sort(key=lambda x: (x[0], x[1]))
    return results[0][2], results[0][3]


def load_or_select_sarima_params(
    weekly_full: pd.DataFrame,
    model_dir: Path,
    reoptimize: bool,
) -> dict:
    params_path = model_dir / "sarima_parameters.json"

    if params_path.exists() and not reoptimize:
        return json.loads(params_path.read_text(encoding="utf-8"))

    in_series = weekly_full.set_index("periodenbeginn")[TARGET_IN].astype(float)
    out_series = weekly_full.set_index("periodenbeginn")[TARGET_OUT].astype(float)

    in_order, in_seasonal = select_sarima_params(in_series)
    out_order, out_seasonal = select_sarima_params(out_series)

    params = {
        "Cash_In": {
            "order": list(in_order),
            "seasonal_order": list(in_seasonal),
        },
        "Cash_Out": {
            "order": list(out_order),
            "seasonal_order": list(out_seasonal),
        },
    }

    params_path.write_text(
        json.dumps(params, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return params


def sarima_forecast(
    weekly_full: pd.DataFrame,
    params: dict,
    forecast_weeks: int,
) -> tuple[pd.DataFrame, object, object]:
    series_in = weekly_full.set_index("periodenbeginn")[TARGET_IN].astype(float)
    series_out = weekly_full.set_index("periodenbeginn")[TARGET_OUT].astype(float)

    fit_in = SARIMAX(
        series_in,
        order=tuple(params["Cash_In"]["order"]),
        seasonal_order=tuple(params["Cash_In"]["seasonal_order"]),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False, maxiter=300)

    fit_out = SARIMAX(
        series_out,
        order=tuple(params["Cash_Out"]["order"]),
        seasonal_order=tuple(params["Cash_Out"]["seasonal_order"]),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False, maxiter=300)

    last_start = weekly_full["periodenbeginn"].max()
    index = pd.date_range(
        last_start + pd.Timedelta(days=7),
        periods=forecast_weeks,
        freq="7D",
    )

    pred_in = np.clip(
        np.asarray(fit_in.forecast(steps=forecast_weeks), dtype=float),
        0.0,
        None,
    )
    pred_out = np.clip(
        np.asarray(fit_out.forecast(steps=forecast_weeks), dtype=float),
        0.0,
        None,
    )

    forecast = pd.DataFrame({
        "periodenbeginn": index,
        "periodenende": index + pd.Timedelta(days=6),
        "Cash_In": pred_in,
        "Cash_Out": pred_out,
    })
    forecast["Netto_Cashflow"] = forecast["Cash_In"] - forecast["Cash_Out"]

    return forecast, fit_in, fit_out


# ---------------------------------------------------------------------------
# 7. REALISIERTE CASHFLOWS + TAGES-/MONATSAUSGABEN
# ---------------------------------------------------------------------------

def prepare_actual_daily(
    daily_train: pd.DataFrame,
    daily_valid: pd.DataFrame,
) -> pd.DataFrame:
    daily = pd.concat([daily_train, daily_valid], ignore_index=True)
    daily["Datum"] = pd.to_datetime(
        daily["berechnungsstichtag"],
        errors="coerce",
    )

    if "ziel_beobachtbar" in daily.columns:
        daily = daily[daily["ziel_beobachtbar"].eq(1)]

    daily = (
        daily.dropna(subset=["Datum", TARGET_IN, TARGET_OUT, TARGET_NET])
        .sort_values("Datum")
        .drop_duplicates(subset=["Datum"], keep="last")
    )

    return pd.DataFrame({
        "Datum": daily["Datum"],
        "Cash_In": daily[TARGET_IN].astype(float),
        "Cash_Out": daily[TARGET_OUT].astype(float),
        "Netto_Cashflow": daily[TARGET_NET].astype(float),
    }).reset_index(drop=True)


def weekday_shares(actual_daily: pd.DataFrame, column: str) -> pd.Series:
    """
    Anteil des historischen Betrags je Wochentag.
    Dient ausschließlich zur konsistenten Aufteilung des nativen
    Wochenforecasts auf Tageswerte.
    """
    frame = actual_daily.copy()
    frame["weekday"] = frame["Datum"].dt.weekday
    sums = frame.groupby("weekday")[column].sum().reindex(range(7), fill_value=0.0)

    total = float(sums.sum())
    if total <= 0:
        shares = pd.Series([1 / 7] * 7, index=range(7), dtype=float)
    else:
        shares = sums / total

    return shares


def weekly_to_daily_forecast(
    weekly_forecast: pd.DataFrame,
    actual_daily: pd.DataFrame,
) -> pd.DataFrame:
    in_shares = weekday_shares(actual_daily, "Cash_In")
    out_shares = weekday_shares(actual_daily, "Cash_Out")

    rows = []

    for _, week in weekly_forecast.iterrows():
        start = pd.Timestamp(week["periodenbeginn"])

        for offset in range(7):
            date = start + pd.Timedelta(days=offset)
            wd = date.weekday()

            cash_in = float(week["Cash_In"]) * float(in_shares.loc[wd])
            cash_out = float(week["Cash_Out"]) * float(out_shares.loc[wd])

            rows.append({
                "Datum": date,
                "Cash_In": cash_in,
                "Cash_Out": cash_out,
                "Netto_Cashflow": cash_in - cash_out,
            })

    return pd.DataFrame(rows)


def daily_to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        daily.set_index("Datum")[["Cash_In", "Cash_Out", "Netto_Cashflow"]]
        .resample("MS")
        .sum()
        .reset_index()
        .rename(columns={"Datum": "Monat"})
    )
    return monthly


def actual_weekly_from_model_data(weekly_full: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "periodenbeginn": pd.to_datetime(weekly_full["periodenbeginn"]),
        "periodenende": pd.to_datetime(weekly_full["periodenende"]),
        "Cash_In": weekly_full[TARGET_IN].astype(float),
        "Cash_Out": weekly_full[TARGET_OUT].astype(float),
        "Netto_Cashflow": weekly_full[TARGET_NET].astype(float),
    })



# ---------------------------------------------------------------------------
# 8. ROLLIERENDER BACKTEST + MODELLGÜTE
# ---------------------------------------------------------------------------

def error_metrics(
    actual: pd.Series | np.ndarray,
    forecast: pd.Series | np.ndarray,
) -> dict[str, float | int]:
    """
    Berechnet Modellgüte auf bereits realisierten Beobachtungen.

    WAPE:
        Summe(|Ist - Prognose|) / Summe(|Ist|)

    MAPE:
        Null-Istwerte werden ausgeschlossen, weil eine prozentuale
        Abweichung dort mathematisch nicht definiert ist.
    """
    actual_arr = np.asarray(actual, dtype=float)
    forecast_arr = np.asarray(forecast, dtype=float)

    finite = np.isfinite(actual_arr) & np.isfinite(forecast_arr)
    actual_arr = actual_arr[finite]
    forecast_arr = forecast_arr[finite]

    if len(actual_arr) == 0:
        return {
            "MAE_Euro": np.nan,
            "RMSE_Euro": np.nan,
            "MAPE_Prozent": np.nan,
            "WAPE_Prozent": np.nan,
            "Beobachtungen": 0,
            "MAPE_Beobachtungen": 0,
        }

    errors = actual_arr - forecast_arr
    mae = float(mean_absolute_error(actual_arr, forecast_arr))
    rmse = float(np.sqrt(mean_squared_error(actual_arr, forecast_arr)))

    nonzero = np.abs(actual_arr) > 1e-12
    if nonzero.any():
        mape = float(
            np.mean(
                np.abs(
                    (actual_arr[nonzero] - forecast_arr[nonzero])
                    / actual_arr[nonzero]
                )
            )
            * 100.0
        )
    else:
        mape = np.nan

    denom = float(np.abs(actual_arr).sum())
    wape = (
        float(np.abs(errors).sum() / denom * 100.0)
        if denom > 0
        else np.nan
    )

    return {
        "MAE_Euro": mae,
        "RMSE_Euro": rmse,
        "MAPE_Prozent": mape,
        "WAPE_Prozent": wape,
        "Beobachtungen": int(len(actual_arr)),
        "MAPE_Beobachtungen": int(nonzero.sum()),
    }


def one_step_sarima_forecast(
    history: pd.DataFrame,
    params: dict,
) -> tuple[float, float]:
    """
    Ein-Wochen-Prognose mit fest gewählter SARIMA-Konfiguration.
    Bei jedem historischen Prognosezeitpunkt wird nur die bis dahin
    bekannte Historie gefittet.
    """
    series_in = (
        history.set_index("periodenbeginn")[TARGET_IN]
        .astype(float)
        .sort_index()
    )
    series_out = (
        history.set_index("periodenbeginn")[TARGET_OUT]
        .astype(float)
        .sort_index()
    )

    fit_in = SARIMAX(
        series_in,
        order=tuple(params["Cash_In"]["order"]),
        seasonal_order=tuple(params["Cash_In"]["seasonal_order"]),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False, maxiter=200)

    fit_out = SARIMAX(
        series_out,
        order=tuple(params["Cash_Out"]["order"]),
        seasonal_order=tuple(params["Cash_Out"]["seasonal_order"]),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False, maxiter=200)

    cash_in = max(0.0, float(np.asarray(fit_in.forecast(steps=1))[0]))
    cash_out = max(0.0, float(np.asarray(fit_out.forecast(steps=1))[0]))

    return cash_in, cash_out


def rolling_weekly_backtest(
    weekly_full: pd.DataFrame,
    feature_columns: list[str],
    sarima_params: dict,
    backtest_weeks: int,
) -> pd.DataFrame:
    """
    Expanding-window / rolling-origin Backtest.

    Für jede Backtest-Woche:
    1) Es werden ausschließlich frühere Wochen als Training verwendet.
    2) XGBoost wird neu gefittet und prognostiziert genau eine Woche.
    3) SARIMA wird mit der fixierten Modellkonfiguration auf derselben
       Historie neu gefittet und prognostiziert genau eine Woche.
    4) Erst danach wird mit dem realisierten Wert verglichen.

    Die letzten 'backtest_weeks' der verfügbaren Historie werden getestet.
    """
    history_all = weekly_full.copy()
    history_all["periodenbeginn"] = pd.to_datetime(
        history_all["periodenbeginn"]
    )
    history_all["periodenende"] = pd.to_datetime(
        history_all["periodenende"]
    )
    history_all = (
        history_all
        .sort_values("periodenbeginn")
        .reset_index(drop=True)
    )

    # 30 Wochen werden mindestens benötigt, um alle freigegebenen
    # XGBoost-Lags bilden zu können. Für SARIMA sind deutlich mehr
    # Beobachtungen sinnvoll; 60 ist ein konservativer Mindestwert.
    min_train_weeks = 60

    possible = len(history_all) - min_train_weeks
    if possible <= 0:
        raise ValueError(
            "Zu wenig Wochen für einen rollierenden Backtest. "
            f"Vorhanden: {len(history_all)}, benötigt > {min_train_weeks}."
        )

    n_test = min(int(backtest_weeks), possible)
    first_test_idx = len(history_all) - n_test

    rows: list[dict] = []

    for i, idx in enumerate(range(first_test_idx, len(history_all)), start=1):
        history = history_all.iloc[:idx].copy()
        actual_row = history_all.iloc[idx]
        forecast_start = pd.Timestamp(actual_row["periodenbeginn"])

        # -------------------------
        # XGBOOST
        # -------------------------
        xgb_in, xgb_out = fit_xgboost_models(
            history,
            feature_columns,
        )

        history_targets = history[
            ["periodenbeginn", TARGET_IN, TARGET_OUT, TARGET_NET]
        ].copy()

        x_features = feature_from_history(
            history_targets,
            forecast_start,
            feature_columns,
        )

        xgb_cash_in = max(
            0.0,
            float(xgb_in.predict(x_features)[0]),
        )
        xgb_cash_out = max(
            0.0,
            float(xgb_out.predict(x_features)[0]),
        )

        # -------------------------
        # SARIMA
        # -------------------------
        try:
            sarima_cash_in, sarima_cash_out = one_step_sarima_forecast(
                history,
                sarima_params,
            )
        except Exception as exc:
            # Der Backtest soll nicht vollständig abbrechen, falls ein einzelner
            # historischer SARIMA-Fit numerisch scheitert.
            print(
                f"WARNUNG: SARIMA-Backtest {forecast_start.date()} "
                f"fehlgeschlagen: {exc}"
            )
            sarima_cash_in = np.nan
            sarima_cash_out = np.nan

        actual_in = float(actual_row[TARGET_IN])
        actual_out = float(actual_row[TARGET_OUT])
        actual_net = float(actual_row[TARGET_NET])

        for model, pred_in, pred_out in [
            ("XGBoost", xgb_cash_in, xgb_cash_out),
            ("SARIMA", sarima_cash_in, sarima_cash_out),
        ]:
            pred_net = (
                float(pred_in - pred_out)
                if np.isfinite(pred_in) and np.isfinite(pred_out)
                else np.nan
            )

            rows.append({
                "periodenbeginn": forecast_start,
                "periodenende": pd.Timestamp(actual_row["periodenende"]),
                "Modell": model,
                "Trainingswochen": int(len(history)),
                "Ist_Cash_In": actual_in,
                "Prognose_Cash_In": pred_in,
                "Ist_Cash_Out": actual_out,
                "Prognose_Cash_Out": pred_out,
                "Ist_Netto_Cashflow": actual_net,
                "Prognose_Netto_Cashflow": pred_net,
            })

        print(
            f"Backtest {i:02d}/{n_test}: "
            f"{forecast_start.date()} abgeschlossen"
        )

    return pd.DataFrame(rows)


def historical_weekday_shares_before(
    actual_daily: pd.DataFrame,
    before_date: pd.Timestamp,
    column: str,
) -> pd.Series:
    """
    Wochentagsanteile ausschließlich aus Tagen VOR dem jeweiligen
    Backtest-Prognosezeitpunkt.
    """
    historical = actual_daily[
        pd.to_datetime(actual_daily["Datum"]) < pd.Timestamp(before_date)
    ].copy()

    if historical.empty:
        return pd.Series([1 / 7] * 7, index=range(7), dtype=float)

    return weekday_shares(historical, column)


def weekly_backtest_to_daily(
    weekly_backtest: pd.DataFrame,
    actual_daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    Leitet tägliche Backtest-Prognosen aus den nativen Wochenprognosen ab.
    Die Verteilung nutzt pro Prognosewoche nur davor bekannte
    historische Wochentagsanteile.
    """
    actual_lookup = (
        actual_daily
        .copy()
        .assign(Datum=lambda d: pd.to_datetime(d["Datum"]))
        .set_index("Datum")
        [["Cash_In", "Cash_Out", "Netto_Cashflow"]]
    )

    rows: list[dict] = []

    for _, week in weekly_backtest.iterrows():
        start = pd.Timestamp(week["periodenbeginn"])
        model = str(week["Modell"])

        in_shares = historical_weekday_shares_before(
            actual_daily,
            start,
            "Cash_In",
        )
        out_shares = historical_weekday_shares_before(
            actual_daily,
            start,
            "Cash_Out",
        )

        for offset in range(7):
            date = start + pd.Timedelta(days=offset)

            if date not in actual_lookup.index:
                continue

            wd = date.weekday()
            pred_in = float(week["Prognose_Cash_In"]) * float(
                in_shares.loc[wd]
            )
            pred_out = float(week["Prognose_Cash_Out"]) * float(
                out_shares.loc[wd]
            )
            pred_net = pred_in - pred_out

            actual = actual_lookup.loc[date]

            rows.append({
                "Datum": date,
                "Modell": model,
                "Ist_Cash_In": float(actual["Cash_In"]),
                "Prognose_Cash_In": pred_in,
                "Ist_Cash_Out": float(actual["Cash_Out"]),
                "Prognose_Cash_Out": pred_out,
                "Ist_Netto_Cashflow": float(actual["Netto_Cashflow"]),
                "Prognose_Netto_Cashflow": pred_net,
            })

    return pd.DataFrame(rows)


def daily_backtest_to_complete_months(
    daily_backtest: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregiert Backtest-Tageswerte auf Kalendermonate.
    Teilmonate am Anfang/Ende des Backtest-Zeitraums werden ausgeschlossen.
    """
    if daily_backtest.empty:
        return pd.DataFrame()

    df = daily_backtest.copy()
    df["Datum"] = pd.to_datetime(df["Datum"])
    df["Monat"] = df["Datum"].dt.to_period("M").dt.to_timestamp()

    value_cols = [
        "Ist_Cash_In",
        "Prognose_Cash_In",
        "Ist_Cash_Out",
        "Prognose_Cash_Out",
        "Ist_Netto_Cashflow",
        "Prognose_Netto_Cashflow",
    ]

    grouped = (
        df.groupby(["Modell", "Monat"], as_index=False)
        .agg(
            **{col: (col, "sum") for col in value_cols},
            Anzahl_Tage=("Datum", "nunique"),
        )
    )

    grouped["Tage_im_Monat"] = grouped["Monat"].dt.days_in_month
    grouped["Monat_vollstaendig"] = (
        grouped["Anzahl_Tage"] == grouped["Tage_im_Monat"]
    )

    return (
        grouped[grouped["Monat_vollstaendig"]]
        .drop(columns=["Tage_im_Monat"])
        .reset_index(drop=True)
    )


def calculate_backtest_metrics(
    weekly_backtest: pd.DataFrame,
    daily_backtest: pd.DataFrame,
    monthly_backtest: pd.DataFrame,
) -> pd.DataFrame:
    """
    MAE / RMSE / MAPE / WAPE je
    Modell × Intervall × Cash-In/Cash-Out/Netto-Cashflow.
    """
    interval_frames = {
        "Taeglich": daily_backtest,
        "Woechentlich": weekly_backtest,
        "Monatlich": monthly_backtest,
    }

    areas = {
        "Cash-In": ("Ist_Cash_In", "Prognose_Cash_In"),
        "Cash-Out": ("Ist_Cash_Out", "Prognose_Cash_Out"),
        "Netto-Cashflow": (
            "Ist_Netto_Cashflow",
            "Prognose_Netto_Cashflow",
        ),
    }

    rows = []

    for interval, frame in interval_frames.items():
        if frame is None or frame.empty:
            continue

        for model in sorted(frame["Modell"].dropna().unique()):
            model_frame = frame[frame["Modell"].eq(model)]

            for area, (actual_col, forecast_col) in areas.items():
                metrics = error_metrics(
                    model_frame[actual_col],
                    model_frame[forecast_col],
                )

                rows.append({
                    "Modell": model,
                    "Intervall": interval,
                    "Bereich": area,
                    **metrics,
                    "Hinweis": (
                        "Native Modellprognose"
                        if interval == "Woechentlich"
                        else (
                            "Aus Wochenprognose mit historischen "
                            "Wochentagsanteilen abgeleitet"
                            if interval == "Taeglich"
                            else "Aus Tagesprognosen aggregiert"
                        )
                    ),
                })

    return pd.DataFrame(rows)


def backtest_detail_long(
    interval: str,
    frame: pd.DataFrame,
    stichtag: pd.Timestamp,
    run_time: datetime,
) -> pd.DataFrame:
    """
    Long-Format für Power BI:
    eine Zeile je Modell × Periode × Bereich.
    """
    if frame.empty:
        return pd.DataFrame()

    if interval == "Taeglich":
        date_col = "Datum"
        end_col = None
    elif interval == "Woechentlich":
        date_col = "periodenbeginn"
        end_col = "periodenende"
    elif interval == "Monatlich":
        date_col = "Monat"
        end_col = None
    else:
        raise ValueError(interval)

    mappings = {
        "Cash-In": ("Ist_Cash_In", "Prognose_Cash_In"),
        "Cash-Out": ("Ist_Cash_Out", "Prognose_Cash_Out"),
        "Netto-Cashflow": (
            "Ist_Netto_Cashflow",
            "Prognose_Netto_Cashflow",
        ),
    }

    rows = []

    for _, source_row in frame.iterrows():
        for area, (actual_col, forecast_col) in mappings.items():
            actual = float(source_row[actual_col])
            forecast = float(source_row[forecast_col])

            rows.append({
                "Datum": pd.Timestamp(source_row[date_col]),
                "Periodenende": (
                    pd.Timestamp(source_row[end_col])
                    if end_col is not None
                    else pd.NaT
                ),
                "Intervall": interval,
                "Modell": str(source_row["Modell"]),
                "Bereich": area,
                "Ist": actual,
                "Prognose": forecast,
                "Fehler": actual - forecast,
                "Absoluter_Fehler": abs(actual - forecast),
                "Datenstichtag": stichtag,
                "Aktualisiert_am": run_time,
                "Backtest_Typ": (
                    "Rolling-Origin / Expanding Window / "
                    "1-Woche-voraus"
                ),
            })

    return pd.DataFrame(rows)


def write_backtest_outputs(
    weekly_backtest: pd.DataFrame,
    daily_backtest: pd.DataFrame,
    monthly_backtest: pd.DataFrame,
    metrics: pd.DataFrame,
    output_root: Path,
    stichtag: pd.Timestamp,
    run_time: datetime,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    backtest_dir = output_root / "Backtest"
    powerbi_dir = output_root / "PowerBI"

    backtest_dir.mkdir(parents=True, exist_ok=True)
    powerbi_dir.mkdir(parents=True, exist_ok=True)

    weekly_backtest.to_csv(
        backtest_dir / "Backtest_Woechentlich.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )
    daily_backtest.to_csv(
        backtest_dir / "Backtest_Taeglich.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )
    monthly_backtest.to_csv(
        backtest_dir / "Backtest_Monatlich.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )
    metrics.to_csv(
        backtest_dir / "Modellguete_MAE_RMSE_MAPE_WAPE.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    detail_frames = [
        backtest_detail_long(
            "Taeglich",
            daily_backtest,
            stichtag,
            run_time,
        ),
        backtest_detail_long(
            "Woechentlich",
            weekly_backtest,
            stichtag,
            run_time,
        ),
        backtest_detail_long(
            "Monatlich",
            monthly_backtest,
            stichtag,
            run_time,
        ),
    ]

    detail_frames = [
        frame for frame in detail_frames
        if frame is not None and not frame.empty
    ]

    detail_powerbi = (
        pd.concat(detail_frames, ignore_index=True)
        if detail_frames
        else pd.DataFrame()
    )

    metrics_powerbi = metrics.copy()
    metrics_powerbi["Datenstichtag"] = stichtag
    metrics_powerbi["Aktualisiert_am"] = run_time
    metrics_powerbi["Backtest_Typ"] = (
        "Rolling-Origin / Expanding Window / 1-Woche-voraus"
    )

    detail_powerbi.to_csv(
        powerbi_dir / "PowerBI_Backtest_Detail.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )
    metrics_powerbi.to_csv(
        powerbi_dir / "PowerBI_Modellguete.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    return detail_powerbi, metrics_powerbi


# ---------------------------------------------------------------------------
# 9. EXPORT
# ---------------------------------------------------------------------------

def write_model_forecasts(
    model_name: str,
    weekly: pd.DataFrame,
    actual_daily: pd.DataFrame,
    output_root: Path,
) -> dict[str, pd.DataFrame]:
    model_dir = output_root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    daily = weekly_to_daily_forecast(weekly, actual_daily)
    monthly = daily_to_monthly(daily)

    daily.to_csv(
        model_dir / f"{model_name}_Prognose_Taeglich.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )
    weekly.to_csv(
        model_dir / f"{model_name}_Prognose_Woechentlich.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )
    monthly.to_csv(
        model_dir / f"{model_name}_Prognose_Monatlich.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "Taeglich": daily,
        "Woechentlich": weekly,
        "Monatlich": monthly,
    }


def write_actuals(
    actual_daily: pd.DataFrame,
    actual_weekly: pd.DataFrame,
    output_root: Path,
) -> dict[str, pd.DataFrame]:
    actual_dir = output_root / "IST"
    actual_dir.mkdir(parents=True, exist_ok=True)

    actual_monthly = daily_to_monthly(actual_daily)

    actual_daily.to_csv(
        actual_dir / "Realisierte_Cashflows_Taeglich.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )
    actual_weekly.to_csv(
        actual_dir / "Realisierte_Cashflows_Woechentlich.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )
    actual_monthly.to_csv(
        actual_dir / "Realisierte_Cashflows_Monatlich.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "Taeglich": actual_daily,
        "Woechentlich": actual_weekly,
        "Monatlich": actual_monthly,
    }


def to_powerbi_rows(
    model: str,
    status: str,
    interval: str,
    df: pd.DataFrame,
    stichtag: pd.Timestamp,
    run_time: datetime,
) -> pd.DataFrame:
    if interval == "Taeglich":
        date_col = "Datum"
        end_col = None
    elif interval == "Woechentlich":
        date_col = "periodenbeginn"
        end_col = "periodenende"
    elif interval == "Monatlich":
        date_col = "Monat"
        end_col = None
    else:
        raise ValueError(interval)

    out = pd.DataFrame({
        "Datum": pd.to_datetime(df[date_col]),
        "Intervall": interval,
        "Modell": model,
        "Status": status,
        "Cash_In": df["Cash_In"].astype(float),
        "Cash_Out": df["Cash_Out"].astype(float),
        "Netto_Cashflow": df["Netto_Cashflow"].astype(float),
        "Datenstichtag": stichtag,
        "Aktualisiert_am": run_time,
    })

    if end_col:
        out["Periodenende"] = pd.to_datetime(df[end_col])
    else:
        out["Periodenende"] = pd.NaT

    return out[
        [
            "Datum", "Periodenende", "Intervall", "Modell", "Status",
            "Cash_In", "Cash_Out", "Netto_Cashflow",
            "Datenstichtag", "Aktualisiert_am",
        ]
    ]


def create_powerbi_file(
    actuals: dict[str, pd.DataFrame],
    xgb: dict[str, pd.DataFrame],
    sarima: dict[str, pd.DataFrame],
    output_root: Path,
    stichtag: pd.Timestamp,
    run_time: datetime,
) -> pd.DataFrame:
    powerbi_dir = output_root / "PowerBI"
    powerbi_dir.mkdir(parents=True, exist_ok=True)

    frames = []

    for interval in ["Taeglich", "Woechentlich", "Monatlich"]:
        frames.append(
            to_powerbi_rows(
                "IST", "Realisiert", interval,
                actuals[interval], stichtag, run_time
            )
        )
        frames.append(
            to_powerbi_rows(
                "XGBoost", "Prognose", interval,
                xgb[interval], stichtag, run_time
            )
        )
        frames.append(
            to_powerbi_rows(
                "SARIMA", "Prognose", interval,
                sarima[interval], stichtag, run_time
            )
        )

    combined = pd.concat(frames, ignore_index=True).sort_values(
        ["Intervall", "Datum", "Modell"]
    )

    stable_path = powerbi_dir / "PowerBI_Cashflow_Gesamt.csv"
    combined.to_csv(
        stable_path,
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    return combined


def archive_run(output_root: Path, run_time: datetime) -> Path:
    archive_dir = output_root / "Archiv" / run_time.strftime("%Y%m%d_%H%M%S")
    archive_dir.mkdir(parents=True, exist_ok=True)

    # Nur die stabilen Prognose-/PowerBI-Dateien archivieren.
    for folder_name in ["XGBoost", "SARIMA", "Backtest", "PowerBI"]:
        src = output_root / folder_name
        if src.exists():
            dst = archive_dir / folder_name
            shutil.copytree(src, dst, dirs_exist_ok=True)

    return archive_dir


# ---------------------------------------------------------------------------
# 9. HAUPTPROGRAMM
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_root = Path(args.output_dir)
    model_dir = output_root / "Modelle"
    output_root.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    run_data_preparation(args, data_dir)

    print("\n" + "=" * 72)
    print("2/7 TRAINING + VALIDIERUNG ZUR FINALEN HISTORIE ZUSAMMENFÜHREN")
    print("=" * 72)

    weekly_train = read_cashai_csv(data_dir / "09_Cashflow_Woechentlich_Training.csv")
    weekly_valid = read_cashai_csv(data_dir / "10_Cashflow_Woechentlich_Validierung.csv")
    daily_train = read_cashai_csv(data_dir / "07_Cashflow_Taeglich_Training.csv")
    daily_valid = read_cashai_csv(data_dir / "08_Cashflow_Taeglich_Validierung.csv")

    weekly_full = combine_training_and_validation(
        weekly_train, weekly_valid, "periodenbeginn"
    )
    weekly_full["periodenende"] = pd.to_datetime(weekly_full["periodenende"])
    daily_full = prepare_actual_daily(daily_train, daily_valid)

    stichtag = pd.Timestamp(daily_full["Datum"].max()).normalize()
    run_time = datetime.now()

    print(f"Finale Trainingswochen: {len(weekly_full)}")
    print(
        f"Historie: {weekly_full['periodenbeginn'].min().date()} "
        f"bis {weekly_full['periodenende'].max().date()}"
    )
    print(f"Datenstichtag: {stichtag.date()}")

    # Nachweisdatei: jetzt inklusive ehemaliger Validierung.
    weekly_full.to_csv(
        output_root / "00_Finale_Trainingsbasis_Woechentlich.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 72)
    print("3/7 XGBOOST FINAL FIT + REKURSIVER WOCHENFORECAST")
    print("=" * 72)

    feature_columns = load_model_columns(data_dir)
    xgb_in, xgb_out = fit_xgboost_models(weekly_full, feature_columns)
    xgb_weekly = recursive_xgboost_forecast(
        weekly_full,
        feature_columns,
        xgb_in,
        xgb_out,
        args.forecast_weeks,
    )

    xgb_in.save_model(model_dir / "XGBoost_Cash_In.json")
    xgb_out.save_model(model_dir / "XGBoost_Cash_Out.json")
    (model_dir / "XGBoost_Features.json").write_text(
        json.dumps(feature_columns, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("4/7 SARIMA FINAL FIT + WOCHENFORECAST")
    print("=" * 72)

    weekly_for_sarima = weekly_full.copy()
    weekly_for_sarima["periodenbeginn"] = pd.to_datetime(
        weekly_for_sarima["periodenbeginn"]
    )

    sarima_params = load_or_select_sarima_params(
        weekly_for_sarima,
        model_dir,
        args.reoptimize_sarima,
    )
    sarima_weekly, sarima_fit_in, sarima_fit_out = sarima_forecast(
        weekly_for_sarima,
        sarima_params,
        args.forecast_weeks,
    )

    joblib.dump(sarima_fit_in, model_dir / "SARIMA_Cash_In.pkl")
    joblib.dump(sarima_fit_out, model_dir / "SARIMA_Cash_Out.pkl")

    # ---------------------------------------------------------------
    # ROLLIERENDER BACKTEST
    # ---------------------------------------------------------------
    backtest_metrics = pd.DataFrame()
    backtest_weekly = pd.DataFrame()
    backtest_daily = pd.DataFrame()
    backtest_monthly = pd.DataFrame()

    if not args.skip_backtest:
        print("\n" + "=" * 72)
        print("5/7 ROLLIERENDER BACKTEST + MAE / RMSE / MAPE / WAPE")
        print("=" * 72)

        backtest_weekly = rolling_weekly_backtest(
            weekly_for_sarima,
            feature_columns,
            sarima_params,
            args.backtest_weeks,
        )

        backtest_daily = weekly_backtest_to_daily(
            backtest_weekly,
            daily_full,
        )

        backtest_monthly = daily_backtest_to_complete_months(
            backtest_daily
        )

        backtest_metrics = calculate_backtest_metrics(
            backtest_weekly,
            backtest_daily,
            backtest_monthly,
        )

        write_backtest_outputs(
            backtest_weekly,
            backtest_daily,
            backtest_monthly,
            backtest_metrics,
            output_root,
            stichtag,
            run_time,
        )

        print("\nMODELLGÜTE – ÜBERSICHT")
        print(
            backtest_metrics[
                [
                    "Modell",
                    "Intervall",
                    "Bereich",
                    "MAE_Euro",
                    "RMSE_Euro",
                    "MAPE_Prozent",
                    "WAPE_Prozent",
                    "Beobachtungen",
                ]
            ]
            .round(2)
            .to_string(index=False)
        )
    else:
        print("\nBacktest wurde mit --skip-backtest übersprungen.")

    print("\n" + "=" * 72)
    print("6/7 TÄGLICH / WÖCHENTLICH / MONATLICH + REALISIERTE CASHFLOWS")
    print("=" * 72)

    actual_weekly = actual_weekly_from_model_data(weekly_full)

    actuals = write_actuals(daily_full, actual_weekly, output_root)
    xgb_outputs = write_model_forecasts(
        "XGBoost", xgb_weekly, daily_full, output_root
    )
    sarima_outputs = write_model_forecasts(
        "SARIMA", sarima_weekly, daily_full, output_root
    )

    powerbi = create_powerbi_file(
        actuals,
        xgb_outputs,
        sarima_outputs,
        output_root,
        stichtag,
        run_time,
    )

    print("\n" + "=" * 72)
    print("7/7 ARCHIVIEREN + ABSCHLUSS")
    print("=" * 72)

    archive_dir = archive_run(output_root, run_time)

    summary = {
        "Aktualisiert_am": run_time.isoformat(timespec="seconds"),
        "Datenstichtag": stichtag.strftime("%Y-%m-%d"),
        "Finale_Trainingswochen": int(len(weekly_full)),
        "Forecast_Wochen": int(args.forecast_weeks),
        "Forecast_Start": xgb_weekly["periodenbeginn"].min().strftime("%Y-%m-%d"),
        "Forecast_Ende": xgb_weekly["periodenende"].max().strftime("%Y-%m-%d"),
        "XGBoost_Parameter": XGB_PARAMS,
        "SARIMA_Parameter": sarima_params,
        "Backtest_Aktiv": bool(not args.skip_backtest),
        "Backtest_Wochen": (
            int(args.backtest_weeks)
            if not args.skip_backtest
            else 0
        ),
        "Modellguete_Datei": (
            str(
                output_root
                / "PowerBI"
                / "PowerBI_Modellguete.csv"
            )
            if not args.skip_backtest
            else None
        ),
        "Hinweis_Modellguete": (
            "Wöchentlich = native Modellgüte. "
            "Täglich = aus Wochenprognose über historische "
            "Wochentagsanteile abgeleitet. "
            "Monatlich = aus Backtest-Tageswerten aggregiert; "
            "Teilmonate werden ausgeschlossen."
        ),
        "Hinweis_Tagesforecast": (
            "Tageswerte werden konsistent aus dem nativen Wochenforecast "
            "über historische Wochentagsanteile abgeleitet."
        ),
    }

    (output_root / "Aktualisierungsstatus.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print("\nERFOLGREICH.")
    print(f"Ergebnisordner: {output_root}")
    print(
        "Power BI Cashflows: "
        f"{output_root / 'PowerBI' / 'PowerBI_Cashflow_Gesamt.csv'}"
    )
    if not args.skip_backtest:
        print(
            "Power BI Modellgüte: "
            f"{output_root / 'PowerBI' / 'PowerBI_Modellguete.csv'}"
        )
        print(
            "Power BI Backtest-Detail: "
            f"{output_root / 'PowerBI' / 'PowerBI_Backtest_Detail.csv'}"
        )
    print(f"Archiv: {archive_dir}")
    print(
        "\nFür Power BI bleibt der Dateipfad stabil. "
        "Nach diesem Lauf in Power BI nur noch 'Aktualisieren' wählen."
    )


if __name__ == "__main__":
    main()
