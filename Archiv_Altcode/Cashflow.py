import pandas as pd
import numpy as np
import os


print("\n======================================")
print("WÖCHENTLICHE CASHFLOW-ANALYSE")
print("======================================")


# ==================================================
# 1. DATEN EINLESEN
# ==================================================

rechnungen = pd.read_excel(
    "Ausgangsrechnungen/ausgangsrechnungen.xlsx"
)

bestellungen = pd.read_excel(
    "Bestellungen/bestellungen.xlsx"
)# ==================================================
# DATEN PRÜFEN
# ==================================================

print("\nRECHNUNGEN:")
print("Anzahl Zeilen:", len(rechnungen))
print("Spalten:", rechnungen.columns.tolist())

print("\nBESTELLUNGEN:")
print("Anzahl Zeilen:", len(bestellungen))
print("Spalten:", bestellungen.columns.tolist())
# ==================================================
# 3. DATENTYPEN SICHERSTELLEN
# ==================================================

rechnungen["zahlungsdatum_ist"] = pd.to_datetime(
    rechnungen["zahlungsdatum_ist"],
    errors="coerce"
)

rechnungen["faelligkeitsdatum"] = pd.to_datetime(
    rechnungen["faelligkeitsdatum"],
    errors="coerce"
)

rechnungen["bezahlter_betrag"] = pd.to_numeric(
    rechnungen["bezahlter_betrag"],
    errors="coerce"
)


bestellungen["zahlungsdatum_ist"] = pd.to_datetime(
    bestellungen["zahlungsdatum_ist"],
    errors="coerce"
)

bestellungen["faelligkeitsdatum"] = pd.to_datetime(
    bestellungen["faelligkeitsdatum"],
    errors="coerce"
)

bestellungen["bezahlter_betrag"] = pd.to_numeric(
    bestellungen["bezahlter_betrag"],
    errors="coerce"
)


# ==================================================
# 4. WÖCHENTLICHE EINZAHLUNGEN
# ==================================================

einzahlungen = (
    rechnungen
    .dropna(
        subset=[
            "zahlungsdatum_ist",
            "bezahlter_betrag"
        ]
    )
    .set_index("zahlungsdatum_ist")["bezahlter_betrag"]
    .resample("W-SUN")
    .sum()
)

einzahlungen.name = "Einzahlungen"


# ==================================================
# 5. WÖCHENTLICHE AUSZAHLUNGEN
# ==================================================

auszahlungen = (
    bestellungen
    .dropna(
        subset=[
            "zahlungsdatum_ist",
            "bezahlter_betrag"
        ]
    )
    .set_index("zahlungsdatum_ist")["bezahlter_betrag"]
    .resample("W-SUN")
    .sum()
)

auszahlungen.name = "Auszahlungen"


# ==================================================
# 6. CASHFLOW ZUSAMMENFÜHREN
# ==================================================

cashflow = pd.concat(
    [
        einzahlungen,
        auszahlungen
    ],
    axis=1
).fillna(0)


cashflow["Netto_Cashflow"] = (
    cashflow["Einzahlungen"]
    - cashflow["Auszahlungen"]
)


cashflow["Kumulierter_Cashflow"] = (
    cashflow["Netto_Cashflow"].cumsum()
)


# ==================================================
# 7. KALENDERWOCHE UND JAHR
# ==================================================

cashflow["Jahr"] = (
    cashflow.index
    .isocalendar()
    .year
    .astype(int)
)

cashflow["KW"] = (
    cashflow.index
    .isocalendar()
    .week
    .astype(int)
)


# ==================================================
# 8. AUSGABE
# ==================================================

print("\n======================================")
print("WÖCHENTLICHER IST-CASHFLOW")
print("======================================")

print(
    cashflow.tail(20).round(2)
)# ==================================================
# 9. ZAHLUNGSVERHALTEN ANALYSIEREN
# ==================================================

# --------------------------------------------------
# KUNDEN / EINZAHLUNGEN
# --------------------------------------------------

rechnungen["zahlungsabweichung_tage"] = (
    rechnungen["zahlungsdatum_ist"]
    - rechnungen["faelligkeitsdatum"]
).dt.days


kunden_zahlungsverhalten = (
    rechnungen
    .dropna(
        subset=["zahlungsabweichung_tage"]
    )
)


print("\n======================================")
print("ZAHLUNGSVERHALTEN KUNDEN")
print("======================================")

print(
    "Durchschnittliche Abweichung:",
    round(
        kunden_zahlungsverhalten[
            "zahlungsabweichung_tage"
        ].mean(),
        2
    ),
    "Tage"
)

print(
    "Median:",
    round(
        kunden_zahlungsverhalten[
            "zahlungsabweichung_tage"
        ].median(),
        2
    ),
    "Tage"
)


# --------------------------------------------------
# LIEFERANTEN / AUSZAHLUNGEN
# --------------------------------------------------

bestellungen["zahlungsabweichung_tage"] = (
    bestellungen["zahlungsdatum_ist"]
    - bestellungen["faelligkeitsdatum"]
).dt.days


lieferanten_zahlungsverhalten = (
    bestellungen
    .dropna(
        subset=["zahlungsabweichung_tage"]
    )
)


print("\n======================================")
print("ZAHLUNGSVERHALTEN LIEFERANTEN")
print("======================================")

print(
    "Durchschnittliche Abweichung:",
    round(
        lieferanten_zahlungsverhalten[
            "zahlungsabweichung_tage"
        ].mean(),
        2
    ),
    "Tage"
)

print(
    "Median:",
    round(
        lieferanten_zahlungsverhalten[
            "zahlungsabweichung_tage"
        ].median(),
        2
    ),
    "Tage"
)


# ==================================================
# 10. ZAHLUNGSKATEGORIEN
# ==================================================

def zahlungskategorie(tage):

    if pd.isna(tage):
        return "Keine Zahlung"

    elif tage < -3:
        return "Früh"

    elif tage <= 3:
        return "Pünktlich"

    else:
        return "Verspätet"


rechnungen["zahlungskategorie"] = (
    rechnungen[
        "zahlungsabweichung_tage"
    ].apply(zahlungskategorie)
)

bestellungen["zahlungskategorie"] = (
    bestellungen[
        "zahlungsabweichung_tage"
    ].apply(zahlungskategorie)
)


print("\n======================================")
print("ZAHLUNGSKATEGORIEN KUNDEN")
print("======================================")

print(
    rechnungen[
        "zahlungskategorie"
    ].value_counts()
)


print("\n======================================")
print("ZAHLUNGSKATEGORIEN LIEFERANTEN")
print("======================================")

print(
    bestellungen[
        "zahlungskategorie"
    ].value_counts()
)# ==================================================
# 11. PLAUSIBILITÄTSPRÜFUNG LIEFERANTENZAHLUNGEN
# ==================================================

pruefung_lieferanten = bestellungen[
    [
        "bestellungs_id",
        "bestelldatum",
        "tatsaechlicher_wareneingang",
        "zahlungsziel_tage",
        "faelligkeitsdatum",
        "skontofrist_tage",
        "skontosatz",
        "zahlungsdatum_ist",
        "zahlungsabweichung_tage",
        "bezahlter_betrag"
    ]
].dropna(
    subset=[
        "faelligkeitsdatum",
        "zahlungsdatum_ist"
    ]
)

print("\n======================================")
print("PRÜFUNG LIEFERANTENZAHLUNGEN")
print("======================================")

print(
    pruefung_lieferanten.head(20).to_string(index=False)
)
# ==================================================
# 12. SKONTOANALYSE LIEFERANTEN
# ==================================================

# Tage zwischen Wareneingang und tatsächlicher Zahlung
bestellungen["tage_wareneingang_bis_zahlung"] = (
    bestellungen["zahlungsdatum_ist"]
    - bestellungen["tatsaechlicher_wareneingang"]
).dt.days


# theoretisches Ende der Skontofrist
bestellungen["skonto_bis"] = (
    bestellungen["tatsaechlicher_wareneingang"]
    + pd.to_timedelta(
        bestellungen["skontofrist_tage"],
        unit="D"
    )
)


# Wurde innerhalb der Skontofrist bezahlt?
bestellungen["skonto_genutzt"] = (
    bestellungen["zahlungsdatum_ist"]
    <= bestellungen["skonto_bis"]
)


# Nur tatsächlich bezahlte Bestellungen betrachten
bezahlte_bestellungen = bestellungen.dropna(
    subset=[
        "zahlungsdatum_ist",
        "tatsaechlicher_wareneingang"
    ]
)


print("\n======================================")
print("SKONTOANALYSE LIEFERANTEN")
print("======================================")

print(
    "Durchschnitt Tage Wareneingang bis Zahlung:",
    round(
        bezahlte_bestellungen[
            "tage_wareneingang_bis_zahlung"
        ].mean(),
        2
    )
)

print(
    "Median Tage Wareneingang bis Zahlung:",
    round(
        bezahlte_bestellungen[
            "tage_wareneingang_bis_zahlung"
        ].median(),
        2
    )
)


anzahl_bezahlt = len(bezahlte_bestellungen)

anzahl_skonto = (
    bezahlte_bestellungen["skonto_genutzt"].sum()
)

skonto_quote = (
    anzahl_skonto
    / anzahl_bezahlt
    * 100
)


print(
    "Bezahlte Bestellungen:",
    anzahl_bezahlt
)

print(
    "Innerhalb Skontofrist bezahlt:",
    anzahl_skonto
)

print(
    "Skontonutzungsquote:",
    round(skonto_quote, 2),
    "%"
)


# ==================================================
# 13. POTENZIELLER SKONTOBETRAG
# ==================================================

bezahlte_bestellungen = bezahlte_bestellungen.copy()

bezahlte_bestellungen["theoretischer_skonto"] = (
    bezahlte_bestellungen["bestellwert"]
    * bezahlte_bestellungen["skontosatz"]
    / 100
)


genutzter_skonto = (
    bezahlte_bestellungen.loc[
        bezahlte_bestellungen["skonto_genutzt"],
        "theoretischer_skonto"
    ].sum()
)


print(
    "Theoretisch genutztes Skontovolumen:",
    f"{genutzter_skonto:,.2f}",
    "Euro"
)
# ==================================================
# 14. WÖCHENTLICHER SOLL-CASHFLOW
# ==================================================

# --------------------------------------------------
# SOLL-EINZAHLUNGEN
# nach Fälligkeitsdatum der Kundenrechnungen
# --------------------------------------------------

soll_einzahlungen = (
    rechnungen
    .dropna(
        subset=[
            "faelligkeitsdatum",
            "rechnungsbetrag"
        ]
    )
    .set_index("faelligkeitsdatum")["rechnungsbetrag"]
    .resample("W-SUN")
    .sum()
)

soll_einzahlungen.name = "Soll_Einzahlungen"


# --------------------------------------------------
# SOLL-AUSZAHLUNGEN
# nach Fälligkeitsdatum der Bestellungen
# --------------------------------------------------

soll_auszahlungen = (
    bestellungen
    .dropna(
        subset=[
            "faelligkeitsdatum",
            "bestellwert"
        ]
    )
    .set_index("faelligkeitsdatum")["bestellwert"]
    .resample("W-SUN")
    .sum()
)

soll_auszahlungen.name = "Soll_Auszahlungen"


# ==================================================
# 15. SOLL-CASHFLOW
# ==================================================

soll_cashflow = pd.concat(
    [
        soll_einzahlungen,
        soll_auszahlungen
    ],
    axis=1
).fillna(0)


soll_cashflow["Soll_Netto_Cashflow"] = (
    soll_cashflow["Soll_Einzahlungen"]
    - soll_cashflow["Soll_Auszahlungen"]
)


# ==================================================
# 16. SOLL UND IST ZUSAMMENFÜHREN
# ==================================================

cashflow_vergleich = pd.concat(
    [
        soll_cashflow,
        cashflow[
            [
                "Einzahlungen",
                "Auszahlungen",
                "Netto_Cashflow"
            ]
        ]
    ],
    axis=1
).fillna(0)


# ==================================================
# 17. ABWEICHUNGEN
# ==================================================

cashflow_vergleich["Abweichung_Einzahlungen"] = (
    cashflow_vergleich["Einzahlungen"]
    - cashflow_vergleich["Soll_Einzahlungen"]
)


cashflow_vergleich["Abweichung_Auszahlungen"] = (
    cashflow_vergleich["Auszahlungen"]
    - cashflow_vergleich["Soll_Auszahlungen"]
)


cashflow_vergleich["Abweichung_Netto_Cashflow"] = (
    cashflow_vergleich["Netto_Cashflow"]
    - cashflow_vergleich["Soll_Netto_Cashflow"]
)


# ==================================================
# 18. JAHR UND KALENDERWOCHE
# ==================================================

cashflow_vergleich["Jahr"] = (
    cashflow_vergleich.index
    .isocalendar()
    .year
    .astype(int)
)

cashflow_vergleich["KW"] = (
    cashflow_vergleich.index
    .isocalendar()
    .week
    .astype(int)
)


# ==================================================
# 19. AUSGABE 2026
# ==================================================

vergleich_2026 = cashflow_vergleich[
    cashflow_vergleich["Jahr"] == 2026
]


print("\n======================================")
print("WÖCHENTLICHER SOLL-/IST-CASHFLOW 2026")
print("======================================")

print(
    vergleich_2026.tail(15).round(2)
)
# ==================================================
# 20. STICHTAG DER IST-DATEN
# ==================================================

letzte_einzahlung = (
    rechnungen["zahlungsdatum_ist"].max()
)

letzte_auszahlung = (
    bestellungen["zahlungsdatum_ist"].max()
)

# Gemeinsamer Stichtag:
# Nur bis zu diesem Datum liegen für beide Seiten
# tatsächliche Zahlungsinformationen vor.
stichtag = min(
    letzte_einzahlung,
    letzte_auszahlung
)

print("\n======================================")
print("CASHFLOW-STICHTAG")
print("======================================")

print("Letzte Einzahlung:", letzte_einzahlung)
print("Letzte Auszahlung:", letzte_auszahlung)
print("Gemeinsamer Stichtag:", stichtag)
# ==================================================
# 22. KUNDENSPEZIFISCHES ZAHLUNGSVERHALTEN
# ==================================================

kunden_median = (
    rechnungen
    .dropna(
        subset=[
            "kunden_id",
            "zahlungsabweichung_tage"
        ]
    )
    .groupby("kunden_id")[
        "zahlungsabweichung_tage"
    ]
    .median()
)

globaler_kunden_median = (
    rechnungen[
        "zahlungsabweichung_tage"
    ].median()
)

print("\n======================================")
print("KUNDENSPEZIFISCHES ZAHLUNGSVERHALTEN")
print("======================================")

print(
    "Globaler Kunden-Median:",
    globaler_kunden_median,
    "Tage"
)


# ==================================================
# 23. LIEFERANTENSPEZIFISCHES ZAHLUNGSVERHALTEN
# ==================================================

lieferanten_median = (
    bestellungen
    .dropna(
        subset=[
            "lieferanten_id",
            "tage_wareneingang_bis_zahlung"
        ]
    )
    .groupby("lieferanten_id")[
        "tage_wareneingang_bis_zahlung"
    ]
    .median()
)

globaler_lieferanten_median = (
    bestellungen[
        "tage_wareneingang_bis_zahlung"
    ].median()
)

print("\n======================================")
print("LIEFERANTENSPEZIFISCHES ZAHLUNGSVERHALTEN")
print("======================================")

print(
    "Globaler Lieferanten-Median:",
    globaler_lieferanten_median,
    "Tage"
)
# ==================================================
# 24. ERWARTETE ZAHLUNGSDATEN ERMITTELN
# ==================================================

# -------------------------------
# KUNDEN
# -------------------------------

rechnungen["kunden_median_tage"] = (
    rechnungen["kunden_id"]
    .map(kunden_median)
    .fillna(globaler_kunden_median)
)

rechnungen["erwartetes_zahlungsdatum"] = (
    rechnungen["faelligkeitsdatum"]
    + pd.to_timedelta(
        rechnungen["kunden_median_tage"],
        unit="D"
    )
)


# -------------------------------
# LIEFERANTEN
# -------------------------------

bestellungen["lieferanten_median_tage"] = (
    bestellungen["lieferanten_id"]
    .map(lieferanten_median)
    .fillna(globaler_lieferanten_median)
)

# tatsächlicher Wareneingang bevorzugt,
# sonst erwarteter Wareneingang
bestellungen["wareneingang_prognose"] = (
    bestellungen["tatsaechlicher_wareneingang"]
    .fillna(
        bestellungen["erwarteter_wareneingang"]
    )
)

bestellungen["erwartetes_zahlungsdatum"] = (
    bestellungen["wareneingang_prognose"]
    + pd.to_timedelta(
        bestellungen["lieferanten_median_tage"],
        unit="D"
    )
)


# ==================================================
# 25. ZUKÜNFTIGE / NOCH NICHT BEZAHLTE POSITIONEN
# ==================================================

# Nur Positionen berücksichtigen,
# deren tatsächliches Zahlungsdatum NICHT bekannt ist.

offene_rechnungen = rechnungen[
    rechnungen["zahlungsdatum_ist"].isna()
].copy()

offene_bestellungen = bestellungen[
    bestellungen["zahlungsdatum_ist"].isna()
].copy()


print("\n======================================")
print("OFFENE POSITIONEN")
print("======================================")

print(
    "Offene Rechnungen:",
    len(offene_rechnungen)
)

print(
    "Offene Bestellungen:",
    len(offene_bestellungen)
)


# ==================================================
# 26. ERWARTETE EINZAHLUNGEN PRO WOCHE
# ==================================================

forecast_einzahlungen = (
    offene_rechnungen
    .dropna(
        subset=[
            "erwartetes_zahlungsdatum",
            "rechnungsbetrag"
        ]
    )
    .set_index(
        "erwartetes_zahlungsdatum"
    )["rechnungsbetrag"]
    .resample("W-SUN")
    .sum()
)

forecast_einzahlungen.name = (
    "Erwartete_Einzahlungen"
)


# ==================================================
# 27. ERWARTETE AUSZAHLUNGEN PRO WOCHE
# ==================================================

forecast_auszahlungen = (
    offene_bestellungen
    .dropna(
        subset=[
            "erwartetes_zahlungsdatum",
            "bestellwert"
        ]
    )
    .set_index(
        "erwartetes_zahlungsdatum"
    )["bestellwert"]
    .resample("W-SUN")
    .sum()
)

forecast_auszahlungen.name = (
    "Erwartete_Auszahlungen"
)


# ==================================================
# 28. OPERATIVEN FORECAST ZUSAMMENFÜHREN
# ==================================================

operativer_forecast = pd.concat(
    [
        forecast_einzahlungen,
        forecast_auszahlungen
    ],
    axis=1
).fillna(0)


operativer_forecast[
    "Erwarteter_Netto_Cashflow"
] = (
    operativer_forecast[
        "Erwartete_Einzahlungen"
    ]
    - operativer_forecast[
        "Erwartete_Auszahlungen"
    ]
)


# Kalenderwoche
operativer_forecast["Jahr"] = (
    operativer_forecast.index
    .isocalendar()
    .year
    .astype(int)
)

operativer_forecast["KW"] = (
    operativer_forecast.index
    .isocalendar()
    .week
    .astype(int)
)


# ==================================================
# 29. NUR ZEITRAUM NACH STICHTAG
# ==================================================

operativer_forecast = (
    operativer_forecast[
        operativer_forecast.index
        > stichtag
    ]
)


print("\n======================================")
print("OPERATIVER WÖCHENTLICHER CASHFLOW-FORECAST")
print("======================================")

print(
    operativer_forecast
    .head(15)
    .round(2)
)
# ==================================================
# 30. OPERATIVEN FORECAST MONATLICH AGGREGIEREN
# ==================================================

operativer_monatsforecast = (
    operativer_forecast[
        [
            "Erwartete_Einzahlungen",
            "Erwartete_Auszahlungen"
        ]
    ]
    .resample("MS")
    .sum()
)

operativer_monatsforecast[
    "Erwarteter_Netto_Cashflow"
] = (
    operativer_monatsforecast[
        "Erwartete_Einzahlungen"
    ]
    - operativer_monatsforecast[
        "Erwartete_Auszahlungen"
    ]
)

print("\n======================================")
print("OPERATIVER MONATSFORECAST")
print("======================================")

print(
    operativer_monatsforecast.round(2)
)
# ==================================================
# 31. HISTORISCHE CASHFLOW-CONVERSION
# ==================================================

print("\n======================================")
print("HISTORISCHE CASHFLOW-CONVERSION")
print("======================================")


# ==================================================
# 31.1 RECHNUNGEN VS. EINZAHLUNGEN
# ==================================================

# Rechnungsvolumen nach Rechnungsmonat
historische_rechnungen = (
    rechnungen
    .dropna(
        subset=[
            "rechnungsdatum",
            "rechnungsbetrag"
        ]
    )
    .set_index("rechnungsdatum")["rechnungsbetrag"]
    .resample("MS")
    .sum()
)

historische_rechnungen.name = "Rechnungsvolumen"


# Tatsächliche Einzahlungen nach Zahlungsmonat
historische_einzahlungen = (
    rechnungen
    .dropna(
        subset=[
            "zahlungsdatum_ist",
            "bezahlter_betrag"
        ]
    )
    .set_index("zahlungsdatum_ist")["bezahlter_betrag"]
    .resample("MS")
    .sum()
)

historische_einzahlungen.name = "Einzahlungen"


# Zusammenführen
conversion_kunden = pd.concat(
    [
        historische_rechnungen,
        historische_einzahlungen
    ],
    axis=1
)


# Nur vollständige Monate bis Mai 2026
# Juni wird nicht verwendet, weil Zahlungen aus
# Juni-Rechnungen teilweise erst später eintreffen.
conversion_kunden = conversion_kunden[
    conversion_kunden.index
    < "2026-06-01"
].dropna()


# Verhältnis Einzahlung / Rechnungsvolumen
conversion_kunden["Cash_Conversion"] = (
    conversion_kunden["Einzahlungen"]
    / conversion_kunden["Rechnungsvolumen"]
)


print("\n--------------------------------------")
print("KUNDEN: RECHNUNGSVOLUMEN -> CASH-IN")
print("--------------------------------------")

print(
    conversion_kunden.tail(12).round(3)
)


# ==================================================
# 31.2 BESTELLUNGEN VS. AUSZAHLUNGEN
# ==================================================

historische_bestellungen = (
    bestellungen
    .dropna(
        subset=[
            "bestelldatum",
            "bestellwert"
        ]
    )
    .set_index("bestelldatum")["bestellwert"]
    .resample("MS")
    .sum()
)

historische_bestellungen.name = "Bestellvolumen"


historische_auszahlungen = (
    bestellungen
    .dropna(
        subset=[
            "zahlungsdatum_ist",
            "bezahlter_betrag"
        ]
    )
    .set_index("zahlungsdatum_ist")["bezahlter_betrag"]
    .resample("MS")
    .sum()
)

historische_auszahlungen.name = "Auszahlungen"


conversion_lieferanten = pd.concat(
    [
        historische_bestellungen,
        historische_auszahlungen
    ],
    axis=1
)


conversion_lieferanten = conversion_lieferanten[
    conversion_lieferanten.index
    < "2026-06-01"
].dropna()


conversion_lieferanten["Cash_Conversion"] = (
    conversion_lieferanten["Auszahlungen"]
    / conversion_lieferanten["Bestellvolumen"]
)


print("\n--------------------------------------")
print("LIEFERANTEN: BESTELLVOLUMEN -> CASH-OUT")
print("--------------------------------------")

print(
    conversion_lieferanten.tail(12).round(3)
)


# ==================================================
# 32. DURCHSCHNITT UND MEDIAN
# ==================================================

kunden_conversion_mean = (
    conversion_kunden["Cash_Conversion"].mean()
)

kunden_conversion_median = (
    conversion_kunden["Cash_Conversion"].median()
)


lieferanten_conversion_mean = (
    conversion_lieferanten["Cash_Conversion"].mean()
)

lieferanten_conversion_median = (
    conversion_lieferanten["Cash_Conversion"].median()
)


print("\n======================================")
print("CASH-CONVERSION KENNZAHLEN")
print("======================================")

print("\nKUNDEN:")

print(
    "Durchschnitt:",
    round(kunden_conversion_mean * 100, 2),
    "%"
)

print(
    "Median:",
    round(kunden_conversion_median * 100, 2),
    "%"
)


print("\nLIEFERANTEN:")

print(
    "Durchschnitt:",
    round(lieferanten_conversion_mean * 100, 2),
    "%"
)

print(
    "Median:",
    round(lieferanten_conversion_median * 100, 2),
    "%"
)
# ==================================================
# 33. PAYMENT-LAG-ANALYSE
# ==================================================

print("\n======================================")
print("PAYMENT-LAG-ANALYSE")
print("======================================")


# ==================================================
# 33.1 KUNDEN
# Rechnung -> Zahlung
# ==================================================

rechnungen["tage_rechnung_bis_zahlung"] = (
    rechnungen["zahlungsdatum_ist"]
    - rechnungen["rechnungsdatum"]
).dt.days


kunden_lags = rechnungen[
    "tage_rechnung_bis_zahlung"
].dropna()


print("\nKUNDEN: RECHNUNG -> ZAHLUNG")

print(
    "Durchschnitt:",
    round(kunden_lags.mean(), 2),
    "Tage"
)

print(
    "Median:",
    round(kunden_lags.median(), 2),
    "Tage"
)

print(
    "25%-Quantil:",
    round(kunden_lags.quantile(0.25), 2),
    "Tage"
)

print(
    "75%-Quantil:",
    round(kunden_lags.quantile(0.75), 2),
    "Tage"
)


# ==================================================
# 33.2 LIEFERANTEN
# Wareneingang -> Zahlung
# ==================================================

lieferanten_lags = bestellungen[
    "tage_wareneingang_bis_zahlung"
].dropna()


print("\nLIEFERANTEN: WARENEINGANG -> ZAHLUNG")

print(
    "Durchschnitt:",
    round(lieferanten_lags.mean(), 2),
    "Tage"
)

print(
    "Median:",
    round(lieferanten_lags.median(), 2),
    "Tage"
)

print(
    "25%-Quantil:",
    round(lieferanten_lags.quantile(0.25), 2),
    "Tage"
)

print(
    "75%-Quantil:",
    round(lieferanten_lags.quantile(0.75), 2),
    "Tage"
)


# ==================================================
# 34. PAYMENT LAGS IN WOCHEN EINTEILEN
# ==================================================

def lag_klasse(tage):

    if pd.isna(tage):
        return None

    elif tage <= 7:
        return "0-1 Woche"

    elif tage <= 14:
        return "1-2 Wochen"

    elif tage <= 21:
        return "2-3 Wochen"

    elif tage <= 28:
        return "3-4 Wochen"

    elif tage <= 35:
        return "4-5 Wochen"

    elif tage <= 42:
        return "5-6 Wochen"

    elif tage <= 56:
        return "6-8 Wochen"

    else:
        return ">8 Wochen"


rechnungen["lag_klasse"] = (
    rechnungen[
        "tage_rechnung_bis_zahlung"
    ].apply(lag_klasse)
)

bestellungen["lag_klasse"] = (
    bestellungen[
        "tage_wareneingang_bis_zahlung"
    ].apply(lag_klasse)
)


# ==================================================
# 35. VERTEILUNG
# ==================================================

kunden_lag_verteilung = (
    rechnungen[
        "lag_klasse"
    ]
    .value_counts(normalize=True)
    .mul(100)
)


lieferanten_lag_verteilung = (
    bestellungen[
        "lag_klasse"
    ]
    .value_counts(normalize=True)
    .mul(100)
)


print("\n======================================")
print("PAYMENT-LAG-VERTEILUNG KUNDEN")
print("======================================")

print(
    kunden_lag_verteilung.round(2)
)


print("\n======================================")
print("PAYMENT-LAG-VERTEILUNG LIEFERANTEN")
print("======================================")

print(
    lieferanten_lag_verteilung.round(2)
)
# ==================================================
# 36. BESTELLUNG -> WARENEINGANG ANALYSE
# ==================================================

bestellungen["tage_bestellung_bis_wareneingang"] = (
    bestellungen["tatsaechlicher_wareneingang"]
    - bestellungen["bestelldatum"]
).dt.days


wareneingang_lags = (
    bestellungen[
        "tage_bestellung_bis_wareneingang"
    ]
    .dropna()
)


print("\n======================================")
print("BESTELLUNG -> WARENEINGANG")
print("======================================")

print(
    "Durchschnitt:",
    round(wareneingang_lags.mean(), 2),
    "Tage"
)

print(
    "Median:",
    round(wareneingang_lags.median(), 2),
    "Tage"
)

print(
    "25%-Quantil:",
    round(wareneingang_lags.quantile(0.25), 2),
    "Tage"
)

print(
    "75%-Quantil:",
    round(wareneingang_lags.quantile(0.75), 2),
    "Tage"
)


# ==================================================
# 37. GESAMTER CASH-OUT-LAG
# Bestellung -> Zahlung
# ==================================================

bestellungen["tage_bestellung_bis_zahlung"] = (
    bestellungen["zahlungsdatum_ist"]
    - bestellungen["bestelldatum"]
).dt.days


gesamt_lag_lieferanten = (
    bestellungen[
        "tage_bestellung_bis_zahlung"
    ]
    .dropna()
)


print("\n======================================")
print("BESTELLUNG -> ZAHLUNG")
print("======================================")

print(
    "Durchschnitt:",
    round(gesamt_lag_lieferanten.mean(), 2),
    "Tage"
)

print(
    "Median:",
    round(gesamt_lag_lieferanten.median(), 2),
    "Tage"
)

print(
    "25%-Quantil:",
    round(
        gesamt_lag_lieferanten.quantile(0.25),
        2
    ),
    "Tage"
)

print(
    "75%-Quantil:",
    round(
        gesamt_lag_lieferanten.quantile(0.75),
        2
    ),
    "Tage"
)


# ==================================================
# 38. GESAMT-LAG IN WOCHEN
# ==================================================

bestellungen["gesamt_lag_klasse"] = (
    bestellungen[
        "tage_bestellung_bis_zahlung"
    ].apply(lag_klasse)
)


gesamt_lag_verteilung = (
    bestellungen[
        "gesamt_lag_klasse"
    ]
    .value_counts(normalize=True)
    .mul(100)
)


print("\n======================================")
print("PAYMENT-LAG BESTELLUNG -> ZAHLUNG")
print("======================================")

print(
    gesamt_lag_verteilung.round(2)
)
# ==================================================
# 39. SARIMA -> CASHFLOW TRANSFORMATION
# ==================================================

import calendar

print("\n======================================")
print("SARIMA -> CASHFLOW TRANSFORMATION")
print("======================================")


# --------------------------------------------------
# SARIMA FORECAST RECHNUNGEN
# = zukünftiges Rechnungsvolumen
# --------------------------------------------------

sarima_rechnungen = {
    "2026-07": 1706630.31,
    "2026-08": 1117942.69,
    "2026-09": 1075553.19,
    "2026-10": 1097650.56,
    "2026-11": 1653319.24,
    "2026-12": 1300711.55
}


# --------------------------------------------------
# SARIMA FORECAST BESTELLUNGEN
# --------------------------------------------------

sarima_bestellungen = {
    "2026-07": 491116.07,
    "2026-08": 766064.22,
    "2026-09": 841852.85,
    "2026-10": 1012184.20,
    "2026-11": 769306.13,
    "2026-12": 614205.15
}


# Historische Median-Lags
kunden_cash_lag = 24
lieferanten_cash_lag = 41


# ==================================================
# 40. MONATSFORECAST AUF TAGE VERTEILEN
# ==================================================

def monatsforecast_auf_tage(forecast_dict, wert_name):

    daten = []

    for monat, wert in forecast_dict.items():

        start = pd.Timestamp(monat + "-01")

        tage_im_monat = calendar.monthrange(
            start.year,
            start.month
        )[1]

        tageswert = wert / tage_im_monat

        for tag in pd.date_range(
            start=start,
            periods=tage_im_monat,
            freq="D"
        ):

            daten.append(
                {
                    "Datum": tag,
                    wert_name: tageswert
                }
            )

    return pd.DataFrame(daten)


sarima_rechnungen_tag = monatsforecast_auf_tage(
    sarima_rechnungen,
    "Forecast_Rechnungen"
)

sarima_bestellungen_tag = monatsforecast_auf_tage(
    sarima_bestellungen,
    "Forecast_Bestellungen"
)


# ==================================================
# 41. PAYMENT LAGS ANWENDEN
# ==================================================

sarima_rechnungen_tag[
    "Erwartetes_Zahlungsdatum"
] = (
    sarima_rechnungen_tag["Datum"]
    + pd.to_timedelta(
        kunden_cash_lag,
        unit="D"
    )
)


sarima_bestellungen_tag[
    "Erwartetes_Zahlungsdatum"
] = (
    sarima_bestellungen_tag["Datum"]
    + pd.to_timedelta(
        lieferanten_cash_lag,
        unit="D"
    )
)


# ==================================================
# 42. SARIMA CASH-IN WÖCHENTLICH
# ==================================================

sarima_cash_in_woche = (
    sarima_rechnungen_tag
    .set_index("Erwartetes_Zahlungsdatum")[
        "Forecast_Rechnungen"
    ]
    .resample("W-SUN")
    .sum()
)

sarima_cash_in_woche.name = "SARIMA_Cash_In"


# ==================================================
# 43. SARIMA CASH-OUT WÖCHENTLICH
# ==================================================

sarima_cash_out_woche = (
    sarima_bestellungen_tag
    .set_index("Erwartetes_Zahlungsdatum")[
        "Forecast_Bestellungen"
    ]
    .resample("W-SUN")
    .sum()
)

sarima_cash_out_woche.name = "SARIMA_Cash_Out"


# ==================================================
# 44. ZUSAMMENFÜHREN
# ==================================================

sarima_cashflow_woche = pd.concat(
    [
        sarima_cash_in_woche,
        sarima_cash_out_woche
    ],
    axis=1
).fillna(0)


sarima_cashflow_woche["SARIMA_Netto_Cashflow"] = (
    sarima_cashflow_woche["SARIMA_Cash_In"]
    - sarima_cashflow_woche["SARIMA_Cash_Out"]
)


sarima_cashflow_woche["Jahr"] = (
    sarima_cashflow_woche.index
    .isocalendar()
    .year
    .astype(int)
)

sarima_cashflow_woche["KW"] = (
    sarima_cashflow_woche.index
    .isocalendar()
    .week
    .astype(int)
)


print("\n======================================")
print("SARIMA WÖCHENTLICHER CASHFLOW")
print("======================================")

print(
    sarima_cashflow_woche
    .round(2)
)
# ==================================================
# 45. OPERATIVEN FORECAST + SARIMA ZUSAMMENFÜHREN
# ==================================================

gesamt_forecast = pd.concat(
    [
        operativer_forecast[
            [
                "Erwartete_Einzahlungen",
                "Erwartete_Auszahlungen"
            ]
        ],
        sarima_cashflow_woche[
            [
                "SARIMA_Cash_In",
                "SARIMA_Cash_Out"
            ]
        ]
    ],
    axis=1
).fillna(0)


# ==================================================
# 46. GESAMTE EIN- UND AUSZAHLUNGEN
# ==================================================

gesamt_forecast["Gesamt_Einzahlungen"] = (
    gesamt_forecast["Erwartete_Einzahlungen"]
    + gesamt_forecast["SARIMA_Cash_In"]
)

gesamt_forecast["Gesamt_Auszahlungen"] = (
    gesamt_forecast["Erwartete_Auszahlungen"]
    + gesamt_forecast["SARIMA_Cash_Out"]
)


# ==================================================
# 47. GESAMTER NETTO-CASHFLOW
# ==================================================

gesamt_forecast["Gesamt_Netto_Cashflow"] = (
    gesamt_forecast["Gesamt_Einzahlungen"]
    - gesamt_forecast["Gesamt_Auszahlungen"]
)


# ==================================================
# 48. KUMULIERTER FORECAST
# ==================================================

gesamt_forecast["Kumulierter_Forecast_Cashflow"] = (
    gesamt_forecast["Gesamt_Netto_Cashflow"]
    .cumsum()
)


# ==================================================
# 49. JAHR UND KW
# ==================================================

gesamt_forecast["Jahr"] = (
    gesamt_forecast.index
    .isocalendar()
    .year
    .astype(int)
)

gesamt_forecast["KW"] = (
    gesamt_forecast.index
    .isocalendar()
    .week
    .astype(int)
)


# ==================================================
# 50. NUR FORECAST AB JULI 2026
# ==================================================

gesamt_forecast_2026 = gesamt_forecast[
    (gesamt_forecast.index >= "2026-07-01") &
    (gesamt_forecast.index <= "2026-12-31")
]


print("\n======================================")
print("GESAMTER WÖCHENTLICHER CASHFLOW-FORECAST 2026")
print("======================================")

print(
    gesamt_forecast_2026.round(2)
)
# ==================================================
# 51. CASHFLOW-KENNZAHLEN
# ==================================================

print("\n======================================")
print("CASHFLOW-KENNZAHLEN 2026")
print("======================================")

durchschnitt_cashflow = (
    gesamt_forecast_2026["Gesamt_Netto_Cashflow"].mean()
)

negative_wochen = (
    gesamt_forecast_2026["Gesamt_Netto_Cashflow"] < 0
).sum()

positivste_woche = (
    gesamt_forecast_2026["Gesamt_Netto_Cashflow"].idxmax()
)

negativste_woche = (
    gesamt_forecast_2026["Gesamt_Netto_Cashflow"].idxmin()
)

print(
    "Durchschnittlicher Weekly Cashflow:",
    f"{durchschnitt_cashflow:,.2f} Euro"
)

print(
    "Anzahl negativer Wochen:",
    negative_wochen
)

print(
    "Positivste Woche:",
    positivste_woche.date(),
    f"{gesamt_forecast_2026.loc[positivste_woche, 'Gesamt_Netto_Cashflow']:,.2f} Euro"
)

print(
    "Negativste Woche:",
    negativste_woche.date(),
    f"{gesamt_forecast_2026.loc[negativste_woche, 'Gesamt_Netto_Cashflow']:,.2f} Euro"
)
# ==================================================
# 52. CASHFLOW-ERGEBNISSE EXPORTIEREN
# ==================================================

import os

print("\n======================================")
print("CASHFLOW-ERGEBNISSE EXPORTIEREN")
print("======================================")

# Ausgabeordner
output_dir = os.path.join(
    "Ergebnisse",
    "Cashflow"
)

os.makedirs(
    output_dir,
    exist_ok=True
)

# ==================================================
# 53. KENNZAHLEN-TABELLE
# ==================================================

kennzahlen_cashflow = pd.DataFrame({
    "Kennzahl": [
        "Durchschnittlicher Weekly Cashflow",
        "Anzahl negativer Wochen",
        "Positivste Woche",
        "Positivster Weekly Cashflow",
        "Negativste Woche",
        "Negativster Weekly Cashflow",
        "Kumulierter Netto-Cashflow bis Ende 2026"
    ],
    "Wert": [
        durchschnitt_cashflow,
        negative_wochen,
        positivste_woche.strftime("%Y-%m-%d"),
        gesamt_forecast_2026.loc[
            positivste_woche,
            "Gesamt_Netto_Cashflow"
        ],
        negativste_woche.strftime("%Y-%m-%d"),
        gesamt_forecast_2026.loc[
            negativste_woche,
            "Gesamt_Netto_Cashflow"
        ],
        gesamt_forecast_2026[
            "Gesamt_Netto_Cashflow"
        ].sum()
    ]
})

# ==================================================
# 54. CSV EXPORT
# ==================================================

csv_path = os.path.join(
    output_dir,
    "cashflow_forecast_2026.csv"
)

gesamt_forecast_2026.to_csv(
    csv_path,
    sep=";",
    decimal=",",
    encoding="utf-8-sig"
)

# ==================================================
# 55. EXCEL EXPORT
# ==================================================

excel_path = os.path.join(
    output_dir,
    "Cashflow_Analyse.xlsx"
)

with pd.ExcelWriter(
    excel_path,
    engine="openpyxl"
) as writer:

    cashflow.to_excel(
        writer,
        sheet_name="Ist_Cashflow"
    )

    operativer_forecast.to_excel(
        writer,
        sheet_name="Operativer_Forecast"
    )

    sarima_cashflow_woche.to_excel(
        writer,
        sheet_name="SARIMA_Cashflow"
    )

    gesamt_forecast_2026.to_excel(
        writer,
        sheet_name="Gesamtforecast_2026"
    )

    kennzahlen_cashflow.to_excel(
        writer,
        sheet_name="Kennzahlen",
        index=False
    )

print("Excel gespeichert:")
print(excel_path)

print("\nCSV gespeichert:")
print(csv_path)
# ==================================================
# 56. CASHFLOW-FORECAST GRAFIK
# ==================================================

import matplotlib.pyplot as plt

print("\n======================================")
print("CASHFLOW-FORECAST GRAFIK")
print("======================================")

plt.figure(figsize=(14, 7))

plt.plot(
    gesamt_forecast_2026.index,
    gesamt_forecast_2026["Gesamt_Einzahlungen"],
    marker="o",
    label="Einzahlungen"
)

plt.plot(
    gesamt_forecast_2026.index,
    gesamt_forecast_2026["Gesamt_Auszahlungen"],
    marker="o",
    label="Auszahlungen"
)

plt.plot(
    gesamt_forecast_2026.index,
    gesamt_forecast_2026["Gesamt_Netto_Cashflow"],
    marker="o",
    label="Netto-Cashflow"
)

plt.axhline(
    y=0,
    linewidth=1
)

plt.title(
    "Wöchentlicher Cashflow-Forecast 2026"
)

plt.xlabel(
    "Woche"
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


# Grafik speichern
grafik_path = os.path.join(
    output_dir,
    "Cashflow_Forecast_2026.png"
)

plt.savefig(
    grafik_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("\nGrafik gespeichert:")
print(grafik_path)
# ==================================================
# 57. CASHFLOW-BACKTEST JAN-JUN 2026
# ==================================================

from sklearn.metrics import mean_absolute_error, mean_squared_error

print("\n======================================")
print("CASHFLOW-BACKTEST JAN-JUN 2026")
print("======================================")

backtest_stichtag = pd.Timestamp("2025-12-31")


# ==================================================
# 58. ZAHLUNGSVERHALTEN NUR AUS VERGANGENHEIT
# ==================================================

# Kunden:
# Nur Zahlungen verwenden, die bis 31.12.2025 bekannt waren

rechnungen_train = rechnungen[
    rechnungen["zahlungsdatum_ist"] <= backtest_stichtag
].copy()

rechnungen_train["zahlungsabweichung_bt"] = (
    rechnungen_train["zahlungsdatum_ist"]
    - rechnungen_train["faelligkeitsdatum"]
).dt.days

kunden_median_bt = (
    rechnungen_train
    .dropna(
        subset=[
            "kunden_id",
            "zahlungsabweichung_bt"
        ]
    )
    .groupby("kunden_id")[
        "zahlungsabweichung_bt"
    ]
    .median()
)

globaler_kunden_median_bt = (
    rechnungen_train[
        "zahlungsabweichung_bt"
    ].median()
)


# Lieferanten:
# Nur Zahlungen verwenden, die bis 31.12.2025 bekannt waren

bestellungen_train = bestellungen[
    bestellungen["zahlungsdatum_ist"] <= backtest_stichtag
].copy()

bestellungen_train["bestellung_bis_zahlung_bt"] = (
    bestellungen_train["zahlungsdatum_ist"]
    - bestellungen_train["bestelldatum"]
).dt.days

lieferanten_median_bt = (
    bestellungen_train
    .dropna(
        subset=[
            "lieferanten_id",
            "bestellung_bis_zahlung_bt"
        ]
    )
    .groupby("lieferanten_id")[
        "bestellung_bis_zahlung_bt"
    ]
    .median()
)

globaler_lieferanten_median_bt = (
    bestellungen_train[
        "bestellung_bis_zahlung_bt"
    ].median()
)


print(
    "Kunden-Median Backtest:",
    globaler_kunden_median_bt,
    "Tage"
)

print(
    "Lieferanten-Median Backtest:",
    globaler_lieferanten_median_bt,
    "Tage"
)
# ==================================================
# 59. OFFENE POSITIONEN AM 31.12.2025
# ==================================================

# --------------------------------------------------
# OFFENE KUNDENRECHNUNGEN
# --------------------------------------------------

offene_rechnungen_bt = rechnungen[
    (rechnungen["rechnungsdatum"] <= backtest_stichtag)
    &
    (
        rechnungen["zahlungsdatum_ist"].isna()
        |
        (
            rechnungen["zahlungsdatum_ist"]
            > backtest_stichtag
        )
    )
].copy()


offene_rechnungen_bt[
    "kunden_median_bt"
] = (
    offene_rechnungen_bt["kunden_id"]
    .map(kunden_median_bt)
    .fillna(globaler_kunden_median_bt)
)


offene_rechnungen_bt[
    "forecast_zahlungsdatum"
] = (
    offene_rechnungen_bt["faelligkeitsdatum"]
    + pd.to_timedelta(
        offene_rechnungen_bt["kunden_median_bt"],
        unit="D"
    )
)


# --------------------------------------------------
# OFFENE BESTELLUNGEN
# --------------------------------------------------

offene_bestellungen_bt = bestellungen[
    (bestellungen["bestelldatum"] <= backtest_stichtag)
    &
    (
        bestellungen["zahlungsdatum_ist"].isna()
        |
        (
            bestellungen["zahlungsdatum_ist"]
            > backtest_stichtag
        )
    )
].copy()


offene_bestellungen_bt[
    "lieferanten_median_bt"
] = (
    offene_bestellungen_bt["lieferanten_id"]
    .map(lieferanten_median_bt)
    .fillna(globaler_lieferanten_median_bt)
)


# Für den Backtest verwenden wir das Bestelldatum
# + historisches Zahlungsverhalten.
# Dadurch verwenden wir keine später bekannten Ist-Zahlungsdaten.

offene_bestellungen_bt[
    "forecast_zahlungsdatum"
] = (
    offene_bestellungen_bt["bestelldatum"]
    + pd.to_timedelta(
        offene_bestellungen_bt[
            "lieferanten_median_bt"
        ],
        unit="D"
    )
)


print("\nOFFENE POSITIONEN ZUM BACKTEST-STICHTAG:")

print(
    "Offene Rechnungen:",
    len(offene_rechnungen_bt)
)

print(
    "Offene Bestellungen:",
    len(offene_bestellungen_bt)
)
# ==================================================
# 60. OPERATIVER BACKTEST-FORECAST
# ==================================================

operative_einzahlungen_bt = (
    offene_rechnungen_bt
    .dropna(
        subset=[
            "forecast_zahlungsdatum",
            "rechnungsbetrag"
        ]
    )
    .set_index(
        "forecast_zahlungsdatum"
    )["rechnungsbetrag"]
    .resample("W-SUN")
    .sum()
)

operative_einzahlungen_bt.name = (
    "Operative_Einzahlungen"
)


operative_auszahlungen_bt = (
    offene_bestellungen_bt
    .dropna(
        subset=[
            "forecast_zahlungsdatum",
            "bestellwert"
        ]
    )
    .set_index(
        "forecast_zahlungsdatum"
    )["bestellwert"]
    .resample("W-SUN")
    .sum()
)

operative_auszahlungen_bt.name = (
    "Operative_Auszahlungen"
)
# ==================================================
# 61. SARIMA JAN-JUN 2026 FÜR BACKTEST
# ==================================================

sarima_rechnungen_bt = {
    "2026-01": 752084.13,
    "2026-02": 852299.31,
    "2026-03": 756870.42,
    "2026-04": 765126.14,
    "2026-05": 1592633.87,
    "2026-06": 1550672.59
}


sarima_bestellungen_bt = {
    "2026-01": 545580.02,
    "2026-02": 466751.11,
    "2026-03": 481852.17,
    "2026-04": 1832508.02,
    "2026-05": 1603723.37,
    "2026-06": 1543635.99
}# ==================================================
# 62. SARIMA-BACKTEST IN CASHFLOW UMWANDELN
# ==================================================

sarima_rechnungen_bt_tag = monatsforecast_auf_tage(
    sarima_rechnungen_bt,
    "Forecast_Rechnungen"
)

sarima_bestellungen_bt_tag = monatsforecast_auf_tage(
    sarima_bestellungen_bt,
    "Forecast_Bestellungen"
)


# Historische Lags nur aus Trainingsdaten
kunden_lag_bt = (
    rechnungen_train["zahlungsdatum_ist"]
    - rechnungen_train["rechnungsdatum"]
).dt.days.median()


lieferanten_lag_bt = (
    bestellungen_train["zahlungsdatum_ist"]
    - bestellungen_train["bestelldatum"]
).dt.days.median()


print(
    "\nCash-In Lag Backtest:",
    kunden_lag_bt,
    "Tage"
)

print(
    "Cash-Out Lag Backtest:",
    lieferanten_lag_bt,
    "Tage"
)


sarima_rechnungen_bt_tag[
    "Zahlungsdatum"
] = (
    sarima_rechnungen_bt_tag["Datum"]
    + pd.to_timedelta(
        kunden_lag_bt,
        unit="D"
    )
)


sarima_bestellungen_bt_tag[
    "Zahlungsdatum"
] = (
    sarima_bestellungen_bt_tag["Datum"]
    + pd.to_timedelta(
        lieferanten_lag_bt,
        unit="D"
    )
)


sarima_cash_in_bt = (
    sarima_rechnungen_bt_tag
    .set_index("Zahlungsdatum")[
        "Forecast_Rechnungen"
    ]
    .resample("W-SUN")
    .sum()
)

sarima_cash_in_bt.name = "SARIMA_Cash_In"


sarima_cash_out_bt = (
    sarima_bestellungen_bt_tag
    .set_index("Zahlungsdatum")[
        "Forecast_Bestellungen"
    ]
    .resample("W-SUN")
    .sum()
)

sarima_cash_out_bt.name = "SARIMA_Cash_Out"
# ==================================================
# 63. BACKTEST-GESAMTFORECAST
# ==================================================

forecast_bt = pd.concat(
    [
        operative_einzahlungen_bt,
        operative_auszahlungen_bt,
        sarima_cash_in_bt,
        sarima_cash_out_bt
    ],
    axis=1
).fillna(0)


forecast_bt["Forecast_Cash_In"] = (
    forecast_bt["Operative_Einzahlungen"]
    + forecast_bt["SARIMA_Cash_In"]
)


forecast_bt["Forecast_Cash_Out"] = (
    forecast_bt["Operative_Auszahlungen"]
    + forecast_bt["SARIMA_Cash_Out"]
)


forecast_bt["Forecast_Netto"] = (
    forecast_bt["Forecast_Cash_In"]
    - forecast_bt["Forecast_Cash_Out"]
)
# ==================================================
# 64. IST JAN-JUN 2026
# ==================================================

ist_bt = cashflow[
    (cashflow.index >= "2026-01-01")
    &
    (cashflow.index <= "2026-06-28")
][
    [
        "Einzahlungen",
        "Auszahlungen",
        "Netto_Cashflow"
    ]
].copy()


backtest = pd.concat(
    [
        ist_bt,
        forecast_bt[
            [
                "Forecast_Cash_In",
                "Forecast_Cash_Out",
                "Forecast_Netto"
            ]
        ]
    ],
    axis=1
)


# Nur Wochen verwenden,
# für die echte Ist-Werte vorhanden sind
backtest = backtest.loc[
    ist_bt.index
].fillna(0)


print("\n======================================")
print("CASHFLOW-BACKTEST JAN-JUN 2026")
print("======================================")

print(
    backtest.round(2)
)
# ==================================================
# 65. CASHFLOW-FEHLERKENNZAHLEN
# ==================================================

def berechne_metriken(ist, forecast):

    mae = mean_absolute_error(
        ist,
        forecast
    )

    rmse = np.sqrt(
        mean_squared_error(
            ist,
            forecast
        )
    )

    # Nur Werte ungleich 0 für MAPE verwenden
    maske = ist != 0

    if maske.sum() > 0:

        mape = np.mean(
            np.abs(
                (
                    ist[maske]
                    - forecast[maske]
                )
                / ist[maske]
            )
        ) * 100

    else:
        mape = np.nan

    return mae, rmse, mape


# CASH-IN
cashin_mae, cashin_rmse, cashin_mape = (
    berechne_metriken(
        backtest["Einzahlungen"],
        backtest["Forecast_Cash_In"]
    )
)


# CASH-OUT
cashout_mae, cashout_rmse, cashout_mape = (
    berechne_metriken(
        backtest["Auszahlungen"],
        backtest["Forecast_Cash_Out"]
    )
)


# NETTO CASHFLOW
netto_mae, netto_rmse, netto_mape = (
    berechne_metriken(
        backtest["Netto_Cashflow"],
        backtest["Forecast_Netto"]
    )
)


print("\n======================================")
print("CASHFLOW-MODELLBEWERTUNG")
print("======================================")


print("\nCASH-IN")

print(
    f"MAE:  {cashin_mae:,.2f} Euro"
)

print(
    f"RMSE: {cashin_rmse:,.2f} Euro"
)

print(
    f"MAPE: {cashin_mape:.2f} %"
)


print("\nCASH-OUT")

print(
    f"MAE:  {cashout_mae:,.2f} Euro"
)

print(
    f"RMSE: {cashout_rmse:,.2f} Euro"
)

print(
    f"MAPE: {cashout_mape:.2f} %"
)


print("\nNETTO-CASHFLOW")

print(
    f"MAE:  {netto_mae:,.2f} Euro"
)

print(
    f"RMSE: {netto_rmse:,.2f} Euro"
)

print(
    f"MAPE: {netto_mape:.2f} %"
)
# ==================================================
# 66. WAPE BERECHNEN
# ==================================================

def berechne_wape(ist, forecast):

    nenner = np.abs(ist).sum()

    if nenner == 0:
        return np.nan

    return (
        np.abs(ist - forecast).sum()
        / nenner
        * 100
    )


cashin_wape = berechne_wape(
    backtest["Einzahlungen"],
    backtest["Forecast_Cash_In"]
)

cashout_wape = berechne_wape(
    backtest["Auszahlungen"],
    backtest["Forecast_Cash_Out"]
)

netto_wape = berechne_wape(
    backtest["Netto_Cashflow"],
    backtest["Forecast_Netto"]
)


print("\n======================================")
print("WAPE")
print("======================================")

print(f"Cash-In WAPE:  {cashin_wape:.2f} %")
print(f"Cash-Out WAPE: {cashout_wape:.2f} %")
print(f"Netto WAPE:    {netto_wape:.2f} %")


# ==================================================
# 67. BACKTEST-KENNZAHLEN ALS TABELLE
# ==================================================

backtest_kennzahlen = pd.DataFrame({

    "Bereich": [
        "Cash-In",
        "Cash-Out",
        "Netto-Cashflow"
    ],

    "MAE_Euro": [
        cashin_mae,
        cashout_mae,
        netto_mae
    ],

    "RMSE_Euro": [
        cashin_rmse,
        cashout_rmse,
        netto_rmse
    ],

    "MAPE_Prozent": [
        cashin_mape,
        cashout_mape,
        netto_mape
    ],

    "WAPE_Prozent": [
        cashin_wape,
        cashout_wape,
        netto_wape
    ]
})


print("\n======================================")
print("BACKTEST-KENNZAHLEN")
print("======================================")

print(
    backtest_kennzahlen.round(2)
)


# ==================================================
# 68. BESTEHENDE EXCEL-DATEI ERGÄNZEN
# ==================================================

excel_path = os.path.join(
    "Ergebnisse",
    "Cashflow",
    "Cashflow_Analyse.xlsx"
)


with pd.ExcelWriter(
    excel_path,
    engine="openpyxl",
    mode="a",
    if_sheet_exists="replace"
) as writer:

    backtest.to_excel(
        writer,
        sheet_name="Backtest_Jan_Jun_2026"
    )

    backtest_kennzahlen.to_excel(
        writer,
        sheet_name="Backtest_Kennzahlen",
        index=False
    )


print("\n======================================")
print("EXCEL-DATEI ERGÄNZT")
print("======================================")

print(excel_path)