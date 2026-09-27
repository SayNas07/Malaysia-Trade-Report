"""Build the quarterly VARX dataset for Malaysia's trade.

Downloads real quarterly series, seasonally adjusts only those that are
not already adjusted, and writes the merged estimation file plus the
stationarity, cointegration, and lag-length results.

    python src/var_data.py

Annual COMTRADE and WDI series are used only to build fixed export
weights. They are not model variables. The script does not fill gaps.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
LOG_PATH = RAW / "download_log.txt"

# The quarter containing this run is not a full quarter until it has ended.
# 2026Q3 is incomplete in the published TPU file (monthly data stop in August).
DROP_FROM = pd.Period("2026Q3", freq="Q-DEC")
WEIGHT_YEARS = range(2000, 2020)
MAX_LAGS = 8
ADF_ALPHA = 0.05
# A lag that leaves fewer residual degrees of freedom than this makes the
# residual covariance close to singular, so AIC/BIC/HQ keep falling.
MIN_RESIDUAL_DF = 10

EU27 = [
    "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA",
    "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "MLT", "NLD",
    "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE",
]
PARTNERS = [
    ("China", "CHN"),
    ("United States", "USA"),
    ("Singapore", "SGP"),
    ("Japan", "JPN"),
    ("EU27", "EU27"),
]

DOSM_URL = "https://storage.dosm.gov.my/gdp/gdp_qtr_real_sa_demand.csv"
BIS_URL = (
    "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_EER/1.0/M.R.B.MY"
    "?startPeriod=2000-01&format=csv"
)
OECD_BASE = (
    "https://sdmx.oecd.org/public/rest/data/OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA,1.0/"
)
OECD_US_JP = "Q.Y.USA+JPN.S1.S1.B1GQ._Z._Z._Z.XDC.L.N.T0102"
OECD_CN = "Q.N.CHN.S1.S1.B1GQ._Z._Z._Z.XDC.Q.N.T0102"
EUROSTAT_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/namq_10_gdp/"
    "Q.CLV15_MEUR.SCA.B1GQ.EU27_2020?format=SDMX-CSV&startPeriod=2000"
)
SINGSTAT_URL = "https://www.tablebuilder.singstat.gov.sg/api/table/tabledata/M015662"
IMF_PAGE = "https://www.imf.org/en/Research/commodity-prices"
COMMODITY_FILE = RAW / "imf_pallfnf_monthly.csv"

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "Mozilla/5.0"


class DownloadError(RuntimeError):
    pass


def main() -> int:
    for folder in (RAW, PROCESSED, TABLES, FIGURES):
        folder.mkdir(parents=True, exist_ok=True)

    exports, imports = load_malaysia_trade()
    reer = load_reer()
    gdp = load_partner_gdp()
    tpu = load_tpu()
    weights = export_weights()
    commodity = load_commodity()

    frame = build_frame(exports, imports, reer, gdp, tpu, weights, commodity)
    out = PROCESSED / "var_quarterly.csv"
    frame.to_csv(out, index=False)
    weights.to_csv(PROCESSED / "var_export_weights.csv", index=False)
    write_sources(frame)
    plot_series(frame)

    level_cols, diff_cols = model_columns(frame)
    stationarity = stationarity_table(frame, level_cols)
    stationarity.to_csv(TABLES / "var_stationarity.csv", index=False)

    endogenous = ["log_exports", "log_imports", "log_reer"]
    exog = baseline_exog(frame)
    y = frame.set_index("quarter")[endogenous].astype(float)
    y.index = pd.PeriodIndex(y.index, freq="Q-DEC")
    x = frame.set_index("quarter")[exog].astype(float)
    x.index = y.index

    level_lags = lag_table(y, x, "levels")
    differenced = y.diff().dropna()
    diff_lags = lag_table(differenced, x.loc[differenced.index], "differences")
    lags = pd.concat([level_lags, diff_lags], ignore_index=True)
    lags.to_csv(TABLES / "var_lag_selection.csv", index=False)

    orders = integration_orders(stationarity, endogenous)
    johansen = None
    if all(orders[name] == "I(1)" for name in endogenous):
        p_levels = chosen_lag(level_lags, "bic", MIN_RESIDUAL_DF)
        johansen = johansen_table(y, max(p_levels - 1, 0))
        johansen["lag_basis"] = "bic"
        p_hq = chosen_lag(level_lags, "hq", MIN_RESIDUAL_DF)
        if p_hq != p_levels:
            extra = johansen_table(y, max(p_hq - 1, 0))
            extra["lag_basis"] = "hq"
            johansen = pd.concat([johansen, extra], ignore_index=True)
        johansen.to_csv(TABLES / "var_johansen.csv", index=False)
    else:
        print("\nJohansen test skipped: the three endogenous series are not all I(1).")

    print_recommendation(frame, weights, orders, level_lags, diff_lags, johansen, commodity is not None)
    print_manual_gaps(frame)
    print(f"\nWrote {out.relative_to(ROOT).as_posix()} ({len(frame)} quarters).")
    return 0


def log_download(source: str, path: Path, date_range: str, rows: int, note: str = "") -> None:
    stamp = datetime.now().isoformat(timespec="seconds")
    rel = path.relative_to(ROOT).as_posix()
    line = f"[{stamp}] source={source} | file={rel} | date_range={date_range} | rows={rows}"
    if note:
        line += f" | {note}"
    print(line)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def fetch(url: str, dest: Path, source: str) -> bytes:
    try:
        response = SESSION.get(url, timeout=120)
    except requests.RequestException as exc:
        raise DownloadError(f"{source} request failed: {exc}\nURL: {url}") from exc
    if response.status_code != 200 or not response.content:
        raise DownloadError(
            f"{source} returned HTTP {response.status_code}.\nURL: {url}\n"
            f"Save the file manually as {dest.relative_to(ROOT).as_posix()}."
        )
    dest.write_bytes(response.content)
    return response.content


def period_from_label(label: str) -> pd.Period:
    text = str(label).strip().upper().replace("-", "")
    if "Q" not in text:
        raise ValueError(f"Unrecognised quarter label: {label}")
    year, quarter = text.split("Q")
    return pd.Period(f"{year}Q{quarter}", freq="Q-DEC")


def quarter_range(series: pd.Series) -> str:
    return f"{series.index.min()} to {series.index.max()}"


def require_regular(series: pd.Series, name: str) -> pd.Series:
    series = series.sort_index()
    series = series[~series.index.duplicated(keep="last")]
    full = pd.period_range(series.index.min(), series.index.max(), freq="Q-DEC")
    missing = full.difference(series.index)
    if len(missing):
        shown = ", ".join(str(period) for period in missing[:8])
        raise DownloadError(
            f"{name} has {len(missing)} missing quarter(s) ({shown}). "
            "Those quarters were not interpolated."
        )
    if (series <= 0).any():
        raise DownloadError(f"{name} has a non-positive value, so it cannot be logged.")
    return series.astype(float)


def stl_adjust(series: pd.Series, name: str) -> pd.Series:
    """Subtract the quarterly seasonal component. Trend and remainder stay."""
    series = require_regular(series, name)
    stamped = series.copy()
    stamped.index = series.index.to_timestamp(how="end")
    fitted = STL(stamped, period=4, robust=True).fit()
    adjusted = pd.Series(fitted.trend + fitted.resid, index=stamped.index)
    adjusted.index = series.index
    if (adjusted <= 0).any():
        raise DownloadError(f"STL produced a non-positive value for {name}.")
    return adjusted


def load_malaysia_trade() -> tuple[pd.Series, pd.Series]:
    dest = RAW / "dosm_gdp_qtr_real_sa_demand.csv"
    fetch(DOSM_URL, dest, "DOSM real SA GDP by expenditure")
    frame = pd.read_csv(dest)
    needed = {"series", "date", "type", "value"}
    if not needed.issubset(frame.columns):
        raise DownloadError(f"DOSM file is missing columns: {sorted(needed - set(frame.columns))}")
    levels = frame.loc[frame["series"] == "abs", ["date", "type", "value"]]
    wide = levels.pivot(index="date", columns="type", values="value")
    for code in ("e0", "e1", "e2", "e3", "e5", "e6"):
        if code not in wide.columns:
            raise DownloadError(f"DOSM file has no expenditure type {code}.")
    check = wide.loc["2015-01-01"]
    implied = check["e1"] + check["e2"] + check["e3"] + check["e5"] - check["e6"]
    gap = abs(implied - check["e0"]) / check["e0"]
    if gap > 0.03:
        raise DownloadError(
            "DOSM type codes no longer match the expenditure identity "
            f"(gap {gap:.1%} in 2015Q1). e5 and e6 were not labelled."
        )
    # e0 GDP, e1 private consumption, e2 government consumption, e3 GFCF,
    # e5 exports of goods and services, e6 imports. The SA file omits
    # inventories; the 2015Q1 residual is that omitted item.
    wide.index = pd.PeriodIndex(pd.to_datetime(wide.index), freq="Q-DEC")
    exports = require_regular(wide["e5"].dropna(), "Malaysia real exports")
    imports = require_regular(wide["e6"].dropna(), "Malaysia real imports")
    exports, imports = prepend_historical_trade(exports, imports)
    exports = require_regular(exports[exports.index < DROP_FROM], "Malaysia real exports")
    imports = require_regular(imports[imports.index < DROP_FROM], "Malaysia real imports")
    log_download(
        "DOSM gdp_qtr_real_sa_demand, constant 2015 prices, seasonally adjusted",
        dest,
        f"{exports.index.min()} to {exports.index.max()}",
        int(exports.shape[0]),
        "e5 exports and e6 imports of goods and services, RM million. "
        f"2015Q1 expenditure identity residual {gap:.2%}.",
    )
    return exports, imports


def prepend_historical_trade(
    exports: pd.Series, imports: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Use a constant-2015 SA file for 2000Q1-2014Q4 when the user has saved one."""
    path = RAW / "dosm_gdp_real_sa_2000_2014.csv"
    if not path.exists():
        return exports, imports
    frame = pd.read_csv(path)
    needed = {"date", "exports", "imports"}
    if not needed.issubset(frame.columns):
        raise DownloadError(
            "data/raw/dosm_gdp_real_sa_2000_2014.csv needs columns date, exports, imports."
        )
    frame["quarter"] = pd.PeriodIndex(pd.to_datetime(frame["date"]), freq="Q-DEC")
    earlier = frame.loc[frame["quarter"] < exports.index.min()].set_index("quarter")
    if earlier.empty:
        return exports, imports
    old_x = require_regular(earlier["exports"].astype(float), "historical real exports")
    old_m = require_regular(earlier["imports"].astype(float), "historical real imports")
    if not old_x.index.equals(old_m.index):
        raise DownloadError("Historical exports and imports do not cover the same quarters.")
    print(f"Prepended historical DOSM trade, {old_x.index.min()} to {old_x.index.max()}.")
    return pd.concat([old_x, exports]), pd.concat([old_m, imports])


def load_reer() -> pd.Series:
    dest = RAW / "bis_reer_mys_monthly.csv"
    fetch(BIS_URL, dest, "BIS REER")
    frame = pd.read_csv(dest)
    if "TIME_PERIOD" not in frame.columns or "OBS_VALUE" not in frame.columns:
        raise DownloadError("BIS file has no TIME_PERIOD or OBS_VALUE column.")
    title = str(frame["TITLE_TS"].iloc[0]) if "TITLE_TS" in frame.columns else "BIS REER"
    frame = frame.dropna(subset=["OBS_VALUE"]).copy()
    frame["month"] = pd.PeriodIndex(frame["TIME_PERIOD"], freq="M")
    frame["quarter"] = frame["month"].dt.asfreq("Q-DEC")
    grouped = frame.groupby("quarter")["OBS_VALUE"]
    quarterly = grouped.mean()
    counts = grouped.count()
    complete = counts[counts == 3].index
    dropped = quarterly.index.difference(complete)
    quarterly = quarterly.loc[complete]
    quarterly = quarterly[quarterly.index < DROP_FROM]
    adjusted = stl_adjust(quarterly, "BIS REER")
    note = f"{title}. Quarterly mean of the three months, then STL. "
    if len(dropped):
        note += "Dropped incomplete quarters: " + ", ".join(str(q) for q in dropped) + "."
    log_download("BIS WS_EER broad real effective exchange rate, index", dest, quarter_range(adjusted), len(adjusted), note)
    return adjusted


def oecd_frame(key: str, dest: Path, source: str) -> pd.DataFrame:
    url = (
        OECD_BASE + key
        + "?startPeriod=2000-Q1&dimensionAtObservation=AllDimensions&format=csvfile"
    )
    payload = fetch(url, dest, source)
    text = payload.decode("utf-8-sig")
    if "TIME_PERIOD" not in text:
        raise DownloadError(f"{source} did not return an SDMX-CSV table.\nURL: {url}\n{text[:240]}")
    frame = pd.read_csv(StringIO(text))
    log_download(source, dest, f"{frame['TIME_PERIOD'].min()} to {frame['TIME_PERIOD'].max()}", len(frame))
    return frame


def series_from_oecd(frame: pd.DataFrame, area: str, name: str) -> pd.Series:
    part = frame.loc[frame["REF_AREA"] == area, ["TIME_PERIOD", "OBS_VALUE"]].dropna()
    if part.empty:
        raise DownloadError(f"OECD QNA returned no {name} observations.")
    part = part.copy()
    part["quarter"] = part["TIME_PERIOD"].map(lambda value: pd.Period(str(value).replace("-", ""), freq="Q-DEC"))
    series = part.drop_duplicates("quarter").set_index("quarter")["OBS_VALUE"].astype(float)
    return series[series.index < DROP_FROM]


def load_partner_gdp() -> dict[str, pd.Series]:
    us_jp = oecd_frame(
        OECD_US_JP,
        RAW / "oecd_qna_usa_jpn_gdp.csv",
        "OECD QNA real SA GDP, United States and Japan, national currency, chain-linked volume",
    )
    china = oecd_frame(
        OECD_CN,
        RAW / "oecd_qna_chn_gdp_nsa.csv",
        "OECD QNA China real GDP, national currency, constant prices, not seasonally adjusted",
    )
    united_states = require_regular(series_from_oecd(us_jp, "USA", "US real GDP"), "US real GDP")
    japan = require_regular(series_from_oecd(us_jp, "JPN", "Japan real GDP"), "Japan real GDP")
    china_nsa = require_regular(series_from_oecd(china, "CHN", "China real GDP"), "China real GDP")
    if _looks_cumulative(china_nsa):
        raise DownloadError(
            "China's OECD constant-price series looks cumulative within the year. "
            "It was not differenced or seasonally adjusted."
        )
    china_sa = stl_adjust(china_nsa, "China real GDP")

    eurostat = fetch(EUROSTAT_URL, RAW / "eurostat_eu27_gdp_sa.csv", "Eurostat EU27 real SA GDP")
    euro_frame = pd.read_csv(StringIO(eurostat.decode("utf-8-sig")))
    if "TIME_PERIOD" not in euro_frame.columns or "OBS_VALUE" not in euro_frame.columns:
        raise DownloadError("Eurostat file has no TIME_PERIOD or OBS_VALUE column.")
    euro_frame["quarter"] = euro_frame["TIME_PERIOD"].map(period_from_label)
    europe = euro_frame.drop_duplicates("quarter").set_index("quarter")["OBS_VALUE"].astype(float)
    europe = require_regular(europe[europe.index < DROP_FROM], "EU27 real GDP")
    log_download(
        "Eurostat namq_10_gdp EU27_2020, chain-linked 2015 euros, seasonally and calendar adjusted",
        RAW / "eurostat_eu27_gdp_sa.csv",
        quarter_range(europe),
        len(europe),
    )

    singstat = fetch(SINGSTAT_URL, RAW / "singstat_m015662_gdp.json", "SingStat real SA GDP")
    payload = json.loads(singstat.decode("utf-8"))
    rows = payload["Data"]["row"]
    total = next(row for row in rows if row.get("rowText") == "GDP In Chained (2015) Dollars")
    records = []
    for column in total["columns"]:
        key = str(column["key"])
        year, quarter = key.split()
        records.append((pd.Period(f"{year}Q{quarter[0]}", freq="Q-DEC"), float(column["value"])))
    singapore = pd.Series({quarter: value for quarter, value in records}).sort_index()
    singapore = require_regular(singapore[singapore.index < DROP_FROM], "Singapore real GDP")
    updated = payload["Data"].get("dataLastUpdated", "")
    log_download(
        "SingStat table M015662, GDP in chained 2015 dollars, seasonally adjusted, million dollars",
        RAW / "singstat_m015662_gdp.json",
        quarter_range(singapore),
        len(singapore),
        f"dataLastUpdated={updated}",
    )
    return {
        "USA": united_states,
        "JPN": japan,
        "CHN": china_sa,
        "SGP": singapore,
        "EU27": europe,
    }


def _looks_cumulative(series: pd.Series) -> bool:
    """Year-to-date GDP has Q4 several times Q1. A seasonal quarterly level does not."""
    frame = series.rename("value").reset_index()
    frame.columns = ["quarter", "value"]
    frame["year"] = frame["quarter"].dt.year
    frame["q"] = frame["quarter"].dt.quarter
    ratios = []
    for _, part in frame.groupby("year"):
        ordered = part.sort_values("q")
        if list(ordered["q"]) != [1, 2, 3, 4]:
            continue
        values = ordered["value"].to_numpy()
        ratios.append(values[-1] / values[0])
    return bool(ratios) and float(np.median(ratios)) > 2.5


def load_tpu() -> pd.Series:
    path = RAW / "tpu_quarterly.csv"
    if not path.exists():
        raise DownloadError(
            "Missing data/raw/tpu_quarterly.csv. Run python src/fetch_data.py first."
        )
    frame = pd.read_csv(path)
    if "tpuq_published" not in frame.columns or "quarter_label" not in frame.columns:
        raise DownloadError("tpu_quarterly.csv has no tpuq_published column.")
    frame["quarter"] = frame["quarter_label"].map(period_from_label)
    published = frame.dropna(subset=["tpuq_published"]).set_index("quarter")["tpuq_published"].astype(float)
    published = published[published.index < DROP_FROM]
    if (published <= 0).any():
        raise DownloadError("Published TPU has a non-positive value.")
    return published.sort_index()


def export_weights() -> pd.DataFrame:
    path = RAW / "comtrade_mys_partner_total.csv"
    if not path.exists():
        raise DownloadError(
            "Missing data/raw/comtrade_mys_partner_total.csv. Run python src/fetch_data.py first."
        )
    frame = pd.read_csv(
        path,
        usecols=["refYear", "flowCode", "partnerISO", "primaryValue"],
    )
    exports = frame.loc[
        (frame["flowCode"] == "X") & (frame["refYear"].isin(WEIGHT_YEARS))
    ].copy()
    world = exports.loc[exports["partnerISO"] == "W00", ["refYear", "primaryValue"]]
    world = world.groupby("refYear")["primaryValue"].sum()
    if (world <= 0).any() or world.shape[0] < 20:
        raise DownloadError("COMTRADE world export totals are missing for part of 2000-2019.")

    def annual_share(iso_list: list[str]) -> pd.Series:
        part = exports.loc[exports["partnerISO"].isin(iso_list)]
        totals = part.groupby("refYear")["primaryValue"].sum()
        totals = totals.reindex(world.index, fill_value=0.0)
        return totals / world

    rows = []
    for name, code in PARTNERS:
        members = EU27 if code == "EU27" else [code]
        missing = [iso for iso in members if iso not in set(exports["partnerISO"])]
        if missing:
            raise DownloadError(f"COMTRADE export file has no partner {', '.join(missing)}.")
        share = annual_share(members)
        rows.append(
            {
                "partner": name,
                "code": code,
                "mean_export_share": float(share.mean()),
                "years": int(share.shape[0]),
            }
        )
    weights = pd.DataFrame(rows)
    weights["weight"] = weights["mean_export_share"] / weights["mean_export_share"].sum()
    return weights


def load_commodity() -> pd.Series | None:
    if not COMMODITY_FILE.exists():
        return None
    frame = pd.read_csv(COMMODITY_FILE)
    columns = {column.lower(): column for column in frame.columns}
    date_col = columns.get("date") or columns.get("time_period") or columns.get("month")
    value_col = columns.get("value") or columns.get("obs_value") or columns.get("pallfnf")
    if date_col is None or value_col is None:
        raise DownloadError(
            "data/raw/imf_pallfnf_monthly.csv needs columns date (YYYY-MM) and value."
        )
    frame = frame.dropna(subset=[date_col, value_col]).copy()
    frame["month"] = pd.PeriodIndex(frame[date_col].astype(str).str.slice(0, 7), freq="M")
    frame["quarter"] = frame["month"].dt.asfreq("Q-DEC")
    grouped = frame.groupby("quarter")[value_col]
    quarterly = grouped.mean().astype(float)
    quarterly = quarterly.loc[grouped.count() == 3]
    quarterly = quarterly[quarterly.index < DROP_FROM]
    return stl_adjust(quarterly, "IMF PALLFNF")


def build_frame(
    exports: pd.Series,
    imports: pd.Series,
    reer: pd.Series,
    gdp: dict[str, pd.Series],
    tpu: pd.Series,
    weights: pd.DataFrame,
    commodity: pd.Series | None,
) -> pd.DataFrame:
    data = {
        "exports_real_sa": exports,
        "imports_real_sa": imports,
        "reer_sa": reer,
        "gdp_us": gdp["USA"],
        "gdp_china_sa": gdp["CHN"],
        "gdp_singapore": gdp["SGP"],
        "gdp_japan": gdp["JPN"],
        "gdp_eu27": gdp["EU27"],
        "tpu": tpu,
    }
    if commodity is not None:
        data["commodity_sa"] = commodity
    frame = pd.DataFrame(data).sort_index()
    frame = frame.dropna()
    if frame.empty:
        raise DownloadError("The series have no common quarter. Nothing was written.")
    weight_map = dict(zip(weights["code"], weights["weight"]))
    logged = (
        weight_map["CHN"] * np.log(frame["gdp_china_sa"])
        + weight_map["USA"] * np.log(frame["gdp_us"])
        + weight_map["SGP"] * np.log(frame["gdp_singapore"])
        + weight_map["JPN"] * np.log(frame["gdp_japan"])
        + weight_map["EU27"] * np.log(frame["gdp_eu27"])
    )
    frame["log_foreign_gdp"] = logged
    frame["foreign_gdp_index"] = 100 * np.exp(logged - logged.iloc[0])
    frame["log_exports"] = np.log(frame["exports_real_sa"])
    frame["log_imports"] = np.log(frame["imports_real_sa"])
    frame["log_reer"] = np.log(frame["reer_sa"])
    frame["log_gdp_us"] = np.log(frame["gdp_us"])
    frame["log_gdp_china"] = np.log(frame["gdp_china_sa"])
    frame["log_gdp_singapore"] = np.log(frame["gdp_singapore"])
    frame["log_gdp_japan"] = np.log(frame["gdp_japan"])
    frame["log_gdp_eu27"] = np.log(frame["gdp_eu27"])
    if "commodity_sa" in frame.columns:
        frame["log_commodity"] = np.log(frame["commodity_sa"])
    quarters = frame.index
    gfc = (quarters >= pd.Period("2008Q4", freq="Q-DEC")) & (quarters <= pd.Period("2009Q2", freq="Q-DEC"))
    covid = (quarters >= pd.Period("2020Q1", freq="Q-DEC")) & (quarters <= pd.Period("2020Q3", freq="Q-DEC"))
    trade_war = quarters >= pd.Period("2018Q3", freq="Q-DEC")
    frame["dummy_gfc"] = gfc.astype(int)
    frame["dummy_covid"] = covid.astype(int)
    frame["dummy_trade_war"] = trade_war.astype(int)
    frame.insert(0, "quarter", quarters.astype(str))
    return frame.reset_index(drop=True)


def write_sources(frame: pd.DataFrame) -> None:
    start, end = frame["quarter"].iloc[0], frame["quarter"].iloc[-1]
    rows = [
        ("exports_real_sa", "DOSM gdp_qtr_real_sa_demand type e5", "RM million, constant 2015, already SA", "used as published"),
        ("imports_real_sa", "DOSM gdp_qtr_real_sa_demand type e6", "RM million, constant 2015, already SA", "used as published"),
        ("reer_sa", "BIS broad real effective exchange rate, Malaysia", "index, 2020=100 in the monthly source", "quarterly mean, then STL"),
        ("gdp_us", "OECD QNA, USA, B1GQ, XDC, chain-linked volume, SA", "national currency", "already SA"),
        ("gdp_japan", "OECD QNA, JPN, B1GQ, XDC, chain-linked volume, SA", "national currency", "already SA"),
        ("gdp_china_sa", "OECD QNA, CHN, B1GQ, XDC, constant prices, NSA", "million CNY", "STL"),
        ("gdp_singapore", "SingStat M015662 total GDP", "million chained 2015 SGD, already SA", "already SA"),
        ("gdp_eu27", "Eurostat namq_10_gdp EU27_2020 B1GQ CLV15_MEUR SCA", "million chain-linked 2015 euros, SA", "already SA"),
        ("foreign_gdp_index", "Fixed 2000-2019 COMTRADE export-share weights", "100 in the first estimation quarter", "weighted sum of log real GDP"),
        ("tpu", "Caldara-Iacoviello tpuq_published", "index", "official quarterly series, not re-averaged"),
        ("dummy_gfc", "2008Q4-2009Q2", "0/1", "zero throughout a sample that starts in 2015"),
        ("dummy_covid", "2020Q1-2020Q3", "0/1", ""),
        ("dummy_trade_war", "2018Q3 onward", "0/1 step", ""),
    ]
    table = pd.DataFrame(rows, columns=["variable", "source", "unit", "seasonal_adjustment"])
    table["sample"] = f"{start} to {end}"
    table.to_csv(TABLES / "var_sources.csv", index=False)


def model_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    levels = [
        "log_exports",
        "log_imports",
        "log_reer",
        "log_foreign_gdp",
        "log_gdp_us",
        "log_gdp_china",
        "tpu",
    ]
    if "log_commodity" in frame.columns:
        levels.append("log_commodity")
    return levels, levels


def plot_series(frame: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    panels = [
        ("log_exports", "Log real exports"),
        ("log_imports", "Log real imports"),
        ("log_reer", "Log real effective exchange rate"),
        ("log_foreign_gdp", "Log trade-weighted foreign GDP"),
        ("log_gdp_us", "Log US real GDP"),
        ("log_gdp_china", "Log China real GDP"),
        ("log_gdp_singapore", "Log Singapore real GDP"),
        ("log_gdp_japan", "Log Japan real GDP"),
        ("log_gdp_eu27", "Log EU27 real GDP"),
        ("tpu", "Trade policy uncertainty"),
        ("dummy_gfc", "GFC dummy"),
        ("dummy_covid", "COVID dummy"),
        ("dummy_trade_war", "US-China trade-war dummy"),
    ]
    if "log_commodity" in frame.columns:
        panels.append(("log_commodity", "Log commodity price index"))
    quarters = pd.PeriodIndex(frame["quarter"], freq="Q-DEC").to_timestamp(how="end")
    ncols = 3
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.1 * nrows), sharex=True)
    for ax, (column, title) in zip(axes.ravel(), panels):
        ax.plot(quarters, frame[column], color="#1f4e79", linewidth=1.6)
        if column.startswith("dummy_"):
            ax.set_ylim(-0.05, 1.05)
        ax.set_title(title, fontsize=12)
        ax.tick_params(axis="x", labelrotation=0, labelsize=8)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle(
        f"VARX series, {frame['quarter'].iloc[0]} to {frame['quarter'].iloc[-1]}",
        fontsize=14,
    )
    fig.tight_layout()
    path = FIGURES / "var_series.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)


def stationarity_table(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        level = frame[column].astype(float)
        rows.append(unit_root_row(column, "level", level, trend=column != "tpu"))
        rows.append(unit_root_row(column, "difference", level.diff().dropna(), trend=False))
    return pd.DataFrame(rows)


def unit_root_row(name: str, transform: str, series: pd.Series, trend: bool) -> dict:
    regression = "ct" if trend else "c"
    adf_stat, adf_p, *_ = adfuller(
        series, regression=regression, autolag="AIC", result_object=False
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The test statistic is outside")
        kpss_stat, kpss_p, *_ = kpss(
            series, regression=regression, nlags="auto", result_object=False
        )
    return {
        "variable": name,
        "transform": transform,
        "adf_stat": adf_stat,
        "adf_pvalue": adf_p,
        "adf_rejects_unit_root_5": adf_p < ADF_ALPHA,
        "kpss_stat": kpss_stat,
        "kpss_pvalue": kpss_p,
        "kpss_rejects_stationarity_5": kpss_p < ADF_ALPHA,
        "reading": read_tests(adf_p < ADF_ALPHA, kpss_p < ADF_ALPHA),
    }


def read_tests(adf_rejects: bool, kpss_rejects: bool) -> str:
    if adf_rejects and not kpss_rejects:
        return "stationary"
    if not adf_rejects and kpss_rejects:
        return "unit root"
    if adf_rejects and kpss_rejects:
        return "conflict: ADF stationary, KPSS unit root"
    return "conflict: ADF unit root, KPSS stationary"


def integration_orders(table: pd.DataFrame, names: list[str]) -> dict[str, str]:
    orders = {}
    for name in names:
        level = table.loc[(table["variable"] == name) & (table["transform"] == "level")].iloc[0]
        diff = table.loc[(table["variable"] == name) & (table["transform"] == "difference")].iloc[0]
        level_root = not bool(level["adf_rejects_unit_root_5"])
        diff_stationary = bool(diff["adf_rejects_unit_root_5"])
        if (not level_root) and diff_stationary:
            orders[name] = "I(0)"
        elif level_root and diff_stationary:
            orders[name] = "I(1)"
        else:
            orders[name] = "unresolved"
    return orders


def baseline_exog(frame: pd.DataFrame) -> list[str]:
    columns = ["log_foreign_gdp", "tpu", "dummy_covid", "dummy_trade_war"]
    if "log_commodity" in frame.columns:
        columns.insert(2, "log_commodity")
    varying = [column for column in columns if frame[column].nunique() > 1]
    dropped = [column for column in ("dummy_gfc",) if column in frame.columns and frame[column].nunique() < 2]
    if dropped:
        print(
            "GFC dummy is in the dataset and is zero in every estimation quarter, "
            "so it is left out of the lag-length regression."
        )
    return varying


def lag_table(y: pd.DataFrame, exog: pd.DataFrame, specification: str) -> pd.DataFrame:
    rows = []
    skipped = []
    for lag in range(1, MAX_LAGS + 1):
        criteria = information_criteria(y, exog, lag)
        if criteria is None:
            skipped.append(lag)
            continue
        criteria["specification"] = specification
        criteria["lag"] = lag
        rows.append(criteria)
    if not rows:
        raise DownloadError(f"No feasible lag for the {specification} VARX. The sample is too short.")
    table = pd.DataFrame(rows)
    if skipped:
        print(
            f"Lag search for {specification} stopped before lag(s) "
            + ", ".join(str(lag) for lag in skipped)
            + ": residual degrees of freedom would be zero."
        )
    return table


def information_criteria(y: pd.DataFrame, exog: pd.DataFrame, lag: int) -> dict | None:
    """VARX information criteria. Exogenous terms enter contemporaneously."""
    aligned = y.copy()
    regressors = [np.ones((len(y) - lag, 1))]
    for step in range(1, lag + 1):
        lagged = y.shift(step).iloc[lag:]
        regressors.append(lagged.to_numpy())
    regressors.append(exog.iloc[lag:].to_numpy())
    design = np.column_stack(regressors)
    target = aligned.iloc[lag:].to_numpy()
    nobs, n_eq = target.shape
    if nobs <= design.shape[1] + 1:
        return None
    beta, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ beta
    sigma = residual.T @ residual / nobs
    sign, logdet = np.linalg.slogdet(sigma)
    if sign <= 0:
        return None
    n_params = n_eq * design.shape[1]
    return {
        "nobs": nobs,
        "residual_df": nobs - design.shape[1],
        "aic": logdet + 2 * n_params / nobs,
        "bic": logdet + np.log(nobs) * n_params / nobs,
        "hq": logdet + 2 * np.log(np.log(nobs)) * n_params / nobs,
    }


def chosen_lag(table: pd.DataFrame, criterion: str = "bic", min_df: int | None = None) -> int:
    usable = table if min_df is None else table.loc[table["residual_df"] >= min_df]
    if usable.empty:
        usable = table
    return int(usable.loc[usable[criterion].idxmin(), "lag"])


def johansen_table(y: pd.DataFrame, k_ar_diff: int) -> pd.DataFrame:
    # det_order 0: a constant in the cointegrating relation. The series drift,
    # but a linear trend in the VECM would put a quadratic trend in the levels.
    result = coint_johansen(y, det_order=0, k_ar_diff=k_ar_diff)
    rows = []
    for rank in range(y.shape[1]):
        rows.append(
            {
                "null_rank": rank,
                "trace": result.lr1[rank],
                "trace_crit_5": result.cvt[rank, 1],
                "trace_rejects_5": result.lr1[rank] > result.cvt[rank, 1],
                "max_eigen": result.lr2[rank],
                "max_eigen_crit_5": result.cvm[rank, 1],
                "max_eigen_rejects_5": result.lr2[rank] > result.cvm[rank, 1],
                "k_ar_diff": k_ar_diff,
                "det_order": 0,
            }
        )
    return pd.DataFrame(rows)


def cointegrating_rank(table: pd.DataFrame, column: str) -> int:
    rank = 0
    for _, row in table.iterrows():
        if bool(row[column]):
            rank = int(row["null_rank"]) + 1
        else:
            break
    return rank


def print_recommendation(
    frame: pd.DataFrame,
    weights: pd.DataFrame,
    orders: dict[str, str],
    level_lags: pd.DataFrame,
    diff_lags: pd.DataFrame,
    johansen: pd.DataFrame | None,
    has_commodity: bool,
) -> None:
    start, end = frame["quarter"].iloc[0], frame["quarter"].iloc[-1]
    print()
    print("=" * 72)
    print(f"Estimation sample: {start} to {end} ({len(frame)} quarters).")
    print("Malaysia trade: DOSM seasonally adjusted GDP by expenditure,")
    print("  constant 2015 prices, exports (e5) and imports (e6), RM million.")
    print("  Already real and already seasonally adjusted, so they were not deflated")
    print("  and STL was not applied again.")
    print("REER: BIS broad real index, quarterly average of the monthly series, then STL.")
    print("Foreign GDP: US and Japan from OECD QNA (already SA); EU27 from Eurostat")
    print("  (already SA); Singapore from SingStat M015662 (already SA); China from")
    print("  OECD constant-price NSA levels, then STL. No pre-2011 backcast.")
    print("TPU: tpuq_published. The recomputed monthly mean is not in the model.")
    print("Weights (mean export share of world exports, 2000-2019, then rescaled to 1):")
    for _, row in weights.iterrows():
        print(
            f"  {row['partner']}: share {row['mean_export_share']:.3f}, "
            f"weight {row['weight']:.3f}"
        )
    if not has_commodity:
        print("Commodity price index: not in the file. See the manual-download note.")
    print()
    print("Integration order from the ADF at 5% (constant and trend in levels,")
    print("  constant only in differences; TPU has a constant and no trend):")
    for name, order in orders.items():
        print(f"  {name}: {order}")
    print("KPSS does not reject stationarity for exports, imports, or the REER in")
    print("  levels. Its p-values sit on the 0.10 censoring point, so the unit-root")
    print("  reading is the ADF result and the two tests are not unanimous.")
    raw_l = {name: chosen_lag(level_lags, name) for name in ("aic", "bic", "hq")}
    raw_d = {name: chosen_lag(diff_lags, name) for name in ("aic", "bic", "hq")}
    bic_l = chosen_lag(level_lags, "bic", MIN_RESIDUAL_DF)
    aic_l = chosen_lag(level_lags, "aic", MIN_RESIDUAL_DF)
    hq_l = chosen_lag(level_lags, "hq", MIN_RESIDUAL_DF)
    bic_d = chosen_lag(diff_lags, "bic", MIN_RESIDUAL_DF)
    aic_d = chosen_lag(diff_lags, "aic", MIN_RESIDUAL_DF)
    hq_d = chosen_lag(diff_lags, "hq", MIN_RESIDUAL_DF)
    print(
        f"Unrestricted lags, up to the longest lag with any residual degrees of freedom: "
        f"levels AIC {raw_l['aic']}, BIC {raw_l['bic']}, HQ {raw_l['hq']}; "
        f"differences AIC {raw_d['aic']}, BIC {raw_d['bic']}, HQ {raw_d['hq']}."
    )
    print(
        "Where that minimum is the longest computed lag, the residual covariance is "
        "nearly singular and the information criterion is not a usable lag choice."
    )
    print(
        f"Lags that leave at least {MIN_RESIDUAL_DF} residual degrees of freedom: "
        f"levels AIC {aic_l}, BIC {bic_l}, HQ {hq_l}; "
        f"differences AIC {aic_d}, BIC {bic_d}, HQ {hq_d}."
    )
    print(
        f"The lag used below is the BIC lag from that restricted set "
        f"({len(frame)} quarters)."
    )
    print()
    if johansen is None:
        unresolved = [name for name, order in orders.items() if order != "I(1)"]
        print(
            "Recommendation: do not estimate a VECM on these three endogenous series. "
            + ", ".join(unresolved)
            + " is not I(1) on the ADF. Difference only the I(1) variables."
        )
        print(f"A levels VAR is appropriate for any series read as I(0). Chosen lag by BIC: {bic_l}.")
        return
    bic_rows = johansen.loc[johansen["lag_basis"] == "bic"]
    trace_rank = cointegrating_rank(bic_rows, "trace_rejects_5")
    eigen_rank = cointegrating_rank(bic_rows, "max_eigen_rejects_5")
    k_diff = int(bic_rows["k_ar_diff"].iloc[0])
    print(
        f"Johansen on log exports, log imports, and log REER, "
        f"{k_diff} lagged difference(s), constant in the cointegrating relation."
    )
    print(f"  Trace rank at 5%: {trace_rank}. Max-eigenvalue rank at 5%: {eigen_rank}.")
    hq_rows = johansen.loc[johansen["lag_basis"] == "hq"]
    if not hq_rows.empty:
        hq_trace = cointegrating_rank(hq_rows, "trace_rejects_5")
        hq_lag = int(hq_rows["k_ar_diff"].iloc[0]) + 1
        print(
            f"  At the HQ lag of {hq_lag}, the trace rank at 5% is {hq_trace}."
        )
    if trace_rank >= 1:
        print(
            f"Recommendation: VECM. The trace test finds {trace_rank} cointegrating "
            f"relation(s), so differencing all three series would drop that long-run link."
        )
        print(
            f"Chosen lag: {bic_l} in the levels VAR, which is {k_diff} lagged "
            "difference(s) in the VECM. Foreign GDP, TPU, and the COVID and trade-war "
            "dummies stay outside the cointegrating relation as exogenous regressors."
        )
    else:
        print(
            "Recommendation: VAR in differences. The three endogenous series are I(1) "
            "and the trace test does not reject zero cointegrating relations."
        )
        print(
            f"Chosen lag: {bic_d} on the differenced endogenous variables. "
            "The same lag is the starting point for the alternative that replaces "
            "trade-weighted foreign GDP with log US and log China GDP."
        )
    if trace_rank != eigen_rank:
        print("The max-eigenvalue test does not agree with the trace test. The trace result is the one used above.")


def print_manual_gaps(frame: pd.DataFrame) -> None:
    start = frame["quarter"].iloc[0]
    end = frame["quarter"].iloc[-1]
    print()
    print("=" * 72)
    print("MANUAL DOWNLOAD - sample does not start in 2000Q1")
    print("What is missing: Malaysia real exports and imports of goods and services,")
    print("  quarterly, constant 2015 prices, seasonally adjusted, 2000Q1 through 2014Q4.")
    print("OpenDOSM's constant-2015 SA expenditure file starts in 2015Q1.")
    print("Download from: DOSM quarterly GDP by expenditure, historical constant-price")
    print("  series (the time-series table in the quarterly GDP release), or request it")
    print("  from data@dosm.gov.my. Do not use the older 2005-price or 2010-price vintages")
    print("  in the same file as the 2015-price series.")
    print("Save as: data/raw/dosm_gdp_real_sa_2000_2014.csv")
    print("Columns: date (YYYY-MM-DD, first month of the quarter), exports, imports.")
    print("Units: RM million, constant 2015 prices, seasonally adjusted.")
    print(f"This run estimated {start} to {end} and did not backcast the missing years.")
    print("Re-run python src/var_data.py after saving the file. Quarters before 2015Q1")
    print("  are prepended only if exports and imports are strictly positive and the")
    print("  quarters are continuous.")
    print()
    if not COMMODITY_FILE.exists():
        print("MANUAL DOWNLOAD - optional, left out of this file")
        print("What is missing: IMF Primary Commodity Price index PALLFNF")
        print("  (all commodities, fuel and non-fuel), monthly, index 2016=100.")
        print(f"Download from: {IMF_PAGE}")
        print("  The workbook is External_Data.xls / External_Data.csv on that page.")
        print("  The IMF data services host did not resolve from this machine.")
        print("Save as: data/raw/imf_pallfnf_monthly.csv")
        print("Columns: date (YYYY-MM), value.")
        print("Re-run python src/var_data.py after saving it. It will be averaged to")
        print("  complete quarters and seasonally adjusted with STL.")
        print("=" * 72)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DownloadError as exc:
        print()
        print("=" * 72)
        print("STOPPED - a required series was not available.")
        print(exc)
        print("No observation was filled in.")
        raise SystemExit(1)
