"""Build the quarterly VARX dataset for Malaysia's trade.

Downloads real quarterly series, seasonally adjusts only those that are
not already adjusted, and writes the merged estimation file plus the
stationarity, cointegration, and lag-length results.

    python src/var_data.py

Annual COMTRADE and WDI series are used only to build fixed export
weights. They are not model variables. The script does not fill gaps.
"""

from __future__ import annotations

import csv
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
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

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
MAX_LAGS = 12
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
# Seasonally adjusted chain-linked volume. GY is year-on-year percent; G1 is
# quarter-on-quarter percent. Both are percentage changes, not levels.
OECD_CN_GY = "Q.Y.CHN.S1.S1.B1GQ._Z._Z._Z.PC.L.GY.T0102"
OECD_CN_G1 = "Q.Y.CHN.S1.S1.B1GQ._Z._Z._Z.PC.L.G1.T0102"
BEA_URL = "https://apps.bea.gov/national/Release/XLS/Survey/Section1All_xls.xlsx"
# Cabinet Office, Apr-Jun 2026 first preliminary (17 Aug 2026), benchmark year 2020.
JAPAN_URL = (
    "https://www.esri.cao.go.jp/jp/sna/data/data_list/sokuhou/files/2026/qe262/"
    "tables/gaku-jk2621.csv"
)
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

    exports, imports, exports_usd, imports_usd = load_malaysia_trade()
    reer = load_reer()
    gdp = load_partner_gdp()
    tpu = load_tpu()
    weights = export_weights()
    commodity = load_commodity()
    print_source_spans(exports, imports, reer, gdp, tpu)

    frame = build_frame(exports, imports, reer, gdp, tpu, weights, commodity)
    frame = attach_usd(frame, exports_usd, imports_usd)
    out = PROCESSED / "var_quarterly.csv"
    frame.to_csv(out, index=False)
    weights.to_csv(PROCESSED / "var_export_weights.csv", index=False)
    write_sources(frame)
    plot_series(frame)

    level_cols, diff_cols = model_columns(frame)
    stationarity = stationarity_table(frame, level_cols)
    stationarity.to_csv(TABLES / "var_stationarity.csv", index=False)

    endogenous = ["log_reer", "log_exports", "log_imports"]
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
    spec = None
    if all(orders[name] == "I(1)" for name in endogenous):
        spec, johansen = choose_specification(y, x, level_lags)
        johansen.to_csv(TABLES / "var_johansen.csv", index=False)
        pd.DataFrame([spec]).to_csv(TABLES / "var_specification.csv", index=False)
    else:
        print("\nJohansen test skipped: the three endogenous series are not all I(1).")

    print_recommendation(
        frame, weights, orders, level_lags, diff_lags, johansen, commodity is not None, spec
    )
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
    try:
        dest.write_bytes(response.content)
    except OSError as exc:
        print(
            f"Could not replace {dest.name} ({exc}). The new download is used from memory "
            "and the file already on disk was left unchanged."
        )
    return response.content


def period_from_label(label: str) -> pd.Period:
    text = str(label).strip().upper().replace("-", "")
    if "Q" not in text:
        raise ValueError(f"Unrecognised quarter label: {label}")
    year, quarter = text.split("Q")
    return pd.Period(f"{year}Q{quarter}", freq="Q-DEC")


def quarter_range(series: pd.Series) -> str:
    return f"{series.index.min()} to {series.index.max()}"


def require_regular(series: pd.Series, name: str, positive: bool = True) -> pd.Series:
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
    if positive and (series <= 0).any():
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


def load_malaysia_trade() -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """CPI-deflated nominal goods trade. DOSM monthly trade starts in 2000, so no splice."""
    trade_path = RAW / "mys_trade_monthly_dosm.csv"
    cpi_path = RAW / "mys_cpi_monthly.csv"
    fx_path = RAW / "myr_usd_fred.csv"
    us_cpi_path = RAW / "us_cpi_fred.csv"
    for path in (trade_path, cpi_path, fx_path, us_cpi_path):
        if not path.exists():
            raise DownloadError(f"Missing {path.relative_to(ROOT).as_posix()}.")

    trade = pd.read_csv(trade_path)
    needed = {"series", "date", "exports", "imports"}
    if not needed.issubset(trade.columns):
        raise DownloadError(
            "mys_trade_monthly_dosm.csv is missing columns: "
            + ", ".join(sorted(needed - set(trade.columns)))
        )
    levels = trade.loc[trade["series"] == "abs", ["date", "exports", "imports"]].copy()
    if levels.empty:
        raise DownloadError("mys_trade_monthly_dosm.csv has no series=='abs' rows.")
    levels["month"] = pd.PeriodIndex(pd.to_datetime(levels["date"]), freq="M")
    levels = levels.drop_duplicates("month").set_index("month").sort_index()
    if levels.index.min() > pd.Period("2000-01", freq="M"):
        raise DownloadError(
            f"DOSM monthly goods trade starts in {levels.index.min()}, after 2000-01. "
            "A BNM splice was not applied because that case was not reached."
        )
    print(
        f"DOSM nominal goods trade starts in {levels.index.min()}, so it is used alone. "
        "BNM nominal trade was not spliced."
    )

    cpi = pd.read_csv(cpi_path)
    if not {"date", "division", "index"}.issubset(cpi.columns):
        raise DownloadError("mys_cpi_monthly.csv needs columns date, division, index.")
    overall = cpi.loc[cpi["division"] == "overall", ["date", "index"]].copy()
    if overall.empty:
        raise DownloadError("mys_cpi_monthly.csv has no division=='overall' rows.")
    overall["month"] = pd.PeriodIndex(pd.to_datetime(overall["date"]), freq="M")
    overall = overall.drop_duplicates("month").set_index("month")["index"].astype(float).sort_index()

    fx = _fred_monthly(fx_path, "EXMAUS", "myr_usd_fred.csv")
    us_cpi = _fred_monthly(us_cpi_path, "CPIAUCSL", "us_cpi_fred.csv")

    exports = _deflate_and_adjust(levels["exports"], overall, "CPI-deflated exports")
    imports = _deflate_and_adjust(levels["imports"], overall, "CPI-deflated imports")
    exports_usd = _deflate_and_adjust(
        levels["exports"] / fx, us_cpi, "USD CPI-deflated exports", allow_gap=True
    )
    imports_usd = _deflate_and_adjust(
        levels["imports"] / fx, us_cpi, "USD CPI-deflated imports", allow_gap=True
    )
    compare_with_national_accounts(exports, imports)
    log_download(
        "OpenDOSM monthly goods trade, deflated by headline CPI, summed to quarters, then STL",
        trade_path,
        quarter_range(exports),
        len(exports),
        "Nominal RM goods exports and imports divided by CPI division 'overall'. "
        "Not national-accounts goods and services, and the CPI is not a trade price index.",
    )
    return exports, imports, exports_usd, imports_usd


def _fred_monthly(path: Path, column: str, label: str) -> pd.Series:
    frame = pd.read_csv(path)
    if "observation_date" not in frame.columns or column not in frame.columns:
        raise DownloadError(f"{label} needs columns observation_date and {column}.")
    frame = frame.copy()
    frame["month"] = pd.PeriodIndex(pd.to_datetime(frame["observation_date"]), freq="M")
    series = frame.drop_duplicates("month").set_index("month")[column].astype(float).sort_index()
    return series


def _deflate_and_adjust(
    nominal: pd.Series, price: pd.Series, name: str, allow_gap: bool = False
) -> pd.Series:
    aligned = pd.concat(
        [nominal.rename("nominal"), price.rename("price")], axis=1, join="inner"
    ).dropna()
    if (aligned["price"] <= 0).any() or (aligned["nominal"] <= 0).any():
        raise DownloadError(f"{name} has a non-positive nominal value or price.")
    real = aligned["nominal"] / aligned["price"]
    quarterly = _quarterly_sum(real, name)
    if allow_gap:
        quarterly = _longest_prefix(quarterly, name)
    return stl_adjust(quarterly, name)


def _quarterly_sum(monthly: pd.Series, name: str) -> pd.Series:
    frame = monthly.rename("value").to_frame()
    frame["quarter"] = pd.PeriodIndex(frame.index.to_timestamp(), freq="Q-DEC")
    grouped = frame.groupby("quarter")["value"]
    total = grouped.sum()
    complete = grouped.count() == 3
    dropped = [str(quarter) for quarter in total.index[~complete] if quarter < DROP_FROM and quarter >= pd.Period("2000Q1", freq="Q-DEC")]
    total = total.loc[complete]
    total = total[(total.index >= pd.Period("2000Q1", freq="Q-DEC")) & (total.index < DROP_FROM)]
    if dropped:
        print(f"{name}: incomplete quarters left out ({', '.join(dropped)}). No month was filled in.")
    return total.sort_index()


def _longest_prefix(series: pd.Series, name: str) -> pd.Series:
    """Keep the continuous run from the first quarter. Later pieces after a gap are left out."""
    full = pd.period_range(series.index.min(), series.index.max(), freq="Q-DEC")
    missing = full.difference(series.index)
    if len(missing) == 0:
        return series
    first_gap = missing.min()
    kept = series[series.index < first_gap]
    print(
        f"{name}: stopped before {first_gap} because that quarter is incomplete. "
        f"Kept {kept.index.min()} to {kept.index.max()}. Later quarters were not joined across the gap."
    )
    return kept


def compare_with_national_accounts(exports: pd.Series, imports: pd.Series) -> None:
    path = RAW / "dosm_gdp_qtr_real_sa_demand.csv"
    if not path.exists():
        print("National-accounts comparison skipped: dosm_gdp_qtr_real_sa_demand.csv is missing.")
        return
    frame = pd.read_csv(path)
    levels = frame.loc[frame["series"] == "abs", ["date", "type", "value"]]
    wide = levels.pivot(index="date", columns="type", values="value")
    if "e5" not in wide.columns or "e6" not in wide.columns:
        raise DownloadError("DOSM national-accounts file has no e5 or e6 column for the growth check.")
    wide.index = pd.PeriodIndex(pd.to_datetime(wide.index), freq="Q-DEC")
    rows = []
    for label, built, code in (
        ("exports", exports, "e5"),
        ("imports", imports, "e6"),
    ):
        official = wide[code].dropna().astype(float)
        official = official[(official.index >= pd.Period("2015Q1", freq="Q-DEC")) & (official.index < DROP_FROM)]
        both = pd.concat(
            [built.rename("built"), official.rename("official")], axis=1, join="inner"
        ).dropna()
        growth = np.log(both).diff().dropna()
        corr = float(growth["built"].corr(growth["official"]))
        rows.append(
            {
                "flow": label,
                "start": str(growth.index.min()),
                "end": str(growth.index.max()),
                "quarters": int(len(growth)),
                "growth_correlation": corr,
            }
        )
        print(
            f"Quarterly growth correlation, CPI-deflated goods {label} vs DOSM constant-2015 "
            f"SA goods and services, {growth.index.min()} to {growth.index.max()}: {corr:.3f}."
        )
    pd.DataFrame(rows).to_csv(TABLES / "var_trade_deflator_check.csv", index=False)


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
    united_states = load_us_gdp()
    japan = load_japan_gdp()
    china_sa = load_china_gdp()

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


def load_us_gdp() -> pd.Series:
    dest = RAW / "bea_section1.xlsx"
    fetch(BEA_URL, dest, "BEA NIPA Section 1")
    sheet = pd.read_excel(dest, sheet_name="T10106-Q", header=None)
    header = None
    values = None
    for row in sheet.itertuples(index=False):
        cells = list(row)
        if header is None and any(cell == "1947Q1" for cell in cells):
            header = cells
        if any(cell == "A191RX" for cell in cells):
            values = cells
            break
    if header is None or values is None:
        raise DownloadError("BEA Table 1.1.6 has no A191RX real GDP row.")
    records = []
    for label, value in zip(header, values):
        text = str(label)
        if len(text) == 6 and text[4] == "Q" and text[:4].isdigit():
            records.append((pd.Period(text, freq="Q-DEC"), float(value)))
    series = pd.Series({quarter: value for quarter, value in records}).sort_index()
    series = require_regular(series[series.index < DROP_FROM], "US real GDP")
    log_download(
        "BEA NIPA Table 1.1.6 line 1, real GDP, millions of chained 2017 dollars, SAAR",
        dest,
        quarter_range(series),
        len(series),
        "Code A191RX. Already seasonally adjusted.",
    )
    return series


def load_japan_gdp() -> pd.Series:
    dest = RAW / "esri_japan_real_sa.csv"
    payload = fetch(JAPAN_URL, dest, "Cabinet Office Japan real SA GDP")
    text = payload.decode("cp932")
    year = None
    records = []
    for cells in csv.reader(StringIO(text)):
        if not cells or not str(cells[0]).strip():
            continue
        label = str(cells[0]).strip()
        if "/" in label and label[:4].isdigit():
            year = int(label[:4])
            quarter = _japan_quarter(label.split("/", 1)[1])
        elif year is not None and label[0].isdigit():
            quarter = _japan_quarter(label)
        else:
            continue
        raw_value = str(cells[1]).replace(",", "").strip() if len(cells) > 1 else ""
        if quarter is None or not raw_value:
            continue
        records.append((pd.Period(f"{year}Q{quarter}", freq="Q-DEC"), float(raw_value)))
    if not records:
        raise DownloadError("Cabinet Office real SA GDP file has no quarterly GDP column.")
    series = pd.Series({quarter: value for quarter, value in records}).sort_index()
    series = require_regular(series[series.index < DROP_FROM], "Japan real GDP")
    log_download(
        "Cabinet Office ESRI gaku-jk2621, real seasonally adjusted GDP, billions of chained 2020 yen",
        dest,
        quarter_range(series),
        len(series),
        "Apr-Jun 2026 first preliminary, published 17 Aug 2026. Already seasonally adjusted.",
    )
    return series


def _japan_quarter(label: str) -> int | None:
    part = label.replace(".", "").strip()
    if part.startswith("10"):
        return 4
    if part.startswith("7"):
        return 3
    if part.startswith("4"):
        return 2
    if part.startswith("1"):
        return 1
    return None


def load_china_gdp() -> pd.Series:
    """Index of official seasonally adjusted real GDP growth, 2011Q1 = 100.

    Quarter-on-quarter growth is published from 2011Q1. Year-on-year growth
    is published from 1993Q1. The index uses the quarter-on-quarter rates
    from 2011Q2 onward, then steps back four quarters at a time with the
    year-on-year rates. No annual figure is split into quarters.
    """
    yoy_frame = oecd_frame(
        OECD_CN_GY,
        RAW / "oecd_qna_chn_gdp_gy.csv",
        "OECD QNA China real GDP, seasonally adjusted chain-linked volume, year-on-year percent",
    )
    qoq_frame = oecd_frame(
        OECD_CN_G1,
        RAW / "oecd_qna_chn_gdp_g1.csv",
        "OECD QNA China real GDP, seasonally adjusted chain-linked volume, quarter-on-quarter percent",
    )
    yoy = series_from_oecd(yoy_frame, "CHN", "China real GDP year-on-year growth")
    qoq = series_from_oecd(qoq_frame, "CHN", "China real GDP quarter-on-quarter growth")
    yoy = require_regular(yoy, "China year-on-year real GDP growth", positive=False)
    qoq = require_regular(qoq, "China quarter-on-quarter real GDP growth", positive=False)
    if (1 + yoy / 100 <= 0).any() or (1 + qoq / 100 <= 0).any():
        raise DownloadError("A China growth rate is at or below -100%. The index was not built.")
    anchor = qoq.index.min()
    index = pd.Series(index=pd.period_range(yoy.index.min(), qoq.index.max(), freq="Q-DEC"), dtype=float)
    index.loc[anchor] = 100.0
    for quarter in index.loc[anchor + 1 :].index:
        index.loc[quarter] = index.loc[quarter - 1] * (1 + qoq.loc[quarter] / 100)
    quarter = anchor - 1
    while quarter >= index.index.min():
        ahead = quarter + 4
        if ahead not in yoy.index:
            raise DownloadError(
                f"China year-on-year growth is missing in {ahead}, so the index "
                f"was not built back through {quarter}."
            )
        index.loc[quarter] = index.loc[ahead] / (1 + yoy.loc[ahead] / 100)
        quarter -= 1
    if index.isna().any():
        raise DownloadError("The China real GDP index has a gap. It was not interpolated.")
    overlap = []
    for quarter in qoq.index:
        previous = quarter - 4
        if previous >= anchor and previous in index.index:
            implied = (index.loc[quarter] / index.loc[previous] - 1) * 100
            overlap.append(implied - yoy.loc[quarter])
    gap = float(np.max(np.abs(overlap))) if overlap else float("nan")
    nbs = load_nbs_yoy()
    overlap_rows = []
    for quarter in index.index:
        previous = quarter - 4
        if previous not in index.index or quarter not in nbs.index:
            continue
        if quarter > pd.Period("2024Q1", freq="Q-DEC"):
            continue
        implied = 100 * index.loc[quarter] / index.loc[previous]
        overlap_rows.append(
            {
                "quarter": str(quarter),
                "index_yoy": implied,
                "nbs_preceding_year": float(nbs.loc[quarter]),
                "difference_points": implied - float(nbs.loc[quarter]),
            }
        )
    overlap_table = pd.DataFrame(overlap_rows)
    if overlap_table.empty:
        raise DownloadError("The China index and the NBS growth file have no overlapping year-on-year quarter.")
    overlap_table.to_csv(TABLES / "var_china_overlap.csv", index=False)
    max_row = overlap_table.loc[overlap_table["difference_points"].abs().idxmax()]
    print(
        "China overlap, existing index year-on-year versus NBS preceding-year index, "
        f"{overlap_table['quarter'].iloc[0]} to {overlap_table['quarter'].iloc[-1]}: "
        f"mean absolute difference {overlap_table['difference_points'].abs().mean():.2f} points, "
        f"largest {max_row['difference_points']:+.2f} in {max_row['quarter']}."
    )
    last = index.index.max()
    for quarter in pd.period_range(last + 1, nbs.index.max(), freq="Q-DEC"):
        if quarter >= DROP_FROM:
            break
        base = quarter - 4
        if base not in index.index or quarter not in nbs.index:
            raise DownloadError(
                f"Cannot extend the China index to {quarter}: the base quarter or the NBS rate is missing."
            )
        index.loc[quarter] = index.loc[base] * (nbs.loc[quarter] / 100.0)
    index = require_regular(index[index.index >= pd.Period("2000Q1", freq="Q-DEC")], "China real GDP index")
    log_download(
        "China real GDP index from OECD QNA SA chain-linked growth, 2011Q1=100",
        RAW / "oecd_qna_chn_gdp_g1.csv",
        quarter_range(index),
        len(index),
        "Chained on OECD quarter-on-quarter growth from 2011Q2, extended back to 2000Q1 "
        f"with OECD year-on-year growth, then extended past 2024Q1 with NBS current-quarter "
        f"preceding-year indices. Largest OECD quarter-on-quarter versus year-on-year gap "
        f"after 2011 is {gap:.2f} percentage points. "
        f"Largest NBS overlap gap is {float(max_row['difference_points']):+.2f} points "
        f"in {max_row['quarter']}. Not a yuan level, and not an interpolation of annual GDP.",
    )
    return index


def load_nbs_yoy() -> pd.Series:
    """NBS current-quarter real GDP index, preceding year = 100. 104.3 means 4.3% y/y."""
    xlsx = RAW / "china_gdp_growth_nbs.xlsx"
    csv_path = RAW / "china_gdp_growth_nbs.csv"
    if xlsx.exists():
        frame = pd.read_excel(xlsx, header=None)
        header = None
        values = None
        for row in frame.itertuples(index=False):
            cells = ["" if pd.isna(cell) else str(cell).strip() for cell in row]
            joined = " ".join(cells)
            if header is None and any(cell.startswith("2Q") or cell.startswith("1Q") for cell in cells):
                header = cells
            if "Gross Domestic Product" in joined and "Current Quarter" in joined and "Accumulated" not in joined:
                values = cells
        if header is None or values is None:
            raise DownloadError(
                "china_gdp_growth_nbs.xlsx has no 'Indices of Gross Domestic Product, Current Quarter' row."
            )
        pairs = list(zip(header, values))
    elif csv_path.exists():
        lines = csv_path.read_text(encoding="utf-8-sig").splitlines()
        header = None
        values = None
        for line in lines:
            parts = [part.strip() for part in line.split("\t,")]
            if parts and parts[0] == "Indicators":
                header = parts
            name = parts[0] if parts else ""
            if (
                name.startswith("Indices of Gross Domestic Product")
                and "Current Quarter" in name
                and "Accumulated" not in name
            ):
                values = parts
        if header is None or values is None:
            raise DownloadError(
                "china_gdp_growth_nbs.csv has no GDP current-quarter row. "
                "Expected a line starting 'Indices of Gross Domestic Product (preceding year=100) , Current Quarter'."
            )
        print("China growth file used: data/raw/china_gdp_growth_nbs.csv (the xlsx name was not in data/raw).")
        pairs = list(zip(header[1:], values[1:]))
    else:
        raise DownloadError("Missing data/raw/china_gdp_growth_nbs.xlsx.")

    records = []
    for label, value in pairs:
        label = str(label).strip().strip(",")
        value = str(value).strip().strip(",")
        if not label or not value or value.lower() == "nan":
            continue
        pieces = label.replace("Q", " Q").split()
        if len(pieces) < 2 or not pieces[-1].isdigit():
            continue
        quarter_number = pieces[0].replace("Q", "")
        if quarter_number not in {"1", "2", "3", "4"}:
            raise DownloadError(f"Unrecognised NBS quarter label: {label}")
        if value == "":
            raise DownloadError(f"NBS GDP growth is blank in {label}. It was not interpolated.")
        records.append((pd.Period(f"{pieces[-1]}Q{quarter_number}", freq="Q-DEC"), float(value)))
    if not records:
        raise DownloadError("The NBS GDP current-quarter row did not parse into quarters.")
    series = pd.Series({quarter: value for quarter, value in records}).sort_index()
    series = series[series.index < DROP_FROM]
    series = require_regular(series, "NBS China real GDP year-on-year index", positive=False)
    if (series <= 0).any():
        raise DownloadError("An NBS preceding-year index is not positive, so it cannot be chained.")
    log_download(
        "NBS indices of GDP, preceding year=100, current quarter",
        csv_path if not xlsx.exists() else xlsx,
        quarter_range(series),
        len(series),
        "104.3 means real GDP is 4.3% above the same quarter a year earlier.",
    )
    return series


def attach_usd(
    frame: pd.DataFrame, exports_usd: pd.Series, imports_usd: pd.Series
) -> pd.DataFrame:
    quarters = pd.PeriodIndex(frame["quarter"], freq="Q-DEC")
    out = frame.copy()
    out["exports_real_usd_sa"] = exports_usd.reindex(quarters).to_numpy()
    out["imports_real_usd_sa"] = imports_usd.reindex(quarters).to_numpy()
    out["log_exports_usd"] = np.log(out["exports_real_usd_sa"])
    out["log_imports_usd"] = np.log(out["imports_real_usd_sa"])
    missing = int(out["log_exports_usd"].isna().sum())
    if missing:
        print(
            f"USD CPI-deflated trade is missing in {missing} estimation quarter(s). "
            "Those quarters stay in the main file and are left out of that robustness check."
        )
    return out


def print_source_spans(
    exports: pd.Series,
    imports: pd.Series,
    reer: pd.Series,
    gdp: dict[str, pd.Series],
    tpu: pd.Series,
) -> None:
    spans = {
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
    print()
    print("Source spans, first to last non-missing quarter:")
    for name, series in spans.items():
        print(f"  {name}: {series.index.min()} to {series.index.max()} ({len(series)} quarters)")
    start = max(series.index.min() for series in spans.values())
    end = min(series.index.max() for series in spans.values())
    starters = [name for name, series in spans.items() if series.index.min() == start]
    enders = [name for name, series in spans.items() if series.index.max() == end]
    print(f"Common sample: {start} to {end}.")
    print("Truncates the start: " + ", ".join(starters) + ".")
    print("Truncates the end: " + ", ".join(enders) + ".")


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
    fred = RAW / "commodity_prices_fred.csv"
    if fred.exists():
        frame = pd.read_csv(fred)
        if "observation_date" not in frame.columns or "PALLFNFINDEXQ" not in frame.columns:
            raise DownloadError(
                "commodity_prices_fred.csv needs columns observation_date and PALLFNFINDEXQ."
            )
        frame = frame.dropna(subset=["observation_date", "PALLFNFINDEXQ"]).copy()
        frame["quarter"] = pd.PeriodIndex(pd.to_datetime(frame["observation_date"]), freq="Q-DEC")
        if frame["quarter"].duplicated().any():
            raise DownloadError("commodity_prices_fred.csv has a duplicated quarter. It was not averaged.")
        series = frame.set_index("quarter")["PALLFNFINDEXQ"].astype(float).sort_index()
        series = require_regular(series[series.index < DROP_FROM], "IMF PALLFNF quarterly index")
        log_download(
            "FRED PALLFNFINDEXQ, IMF all-commodity price index, quarterly",
            fred,
            quarter_range(series),
            len(series),
            "Published quarterly index. Not seasonally adjusted and not interpolated.",
        )
        return series
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
        ("exports_real_sa", "OpenDOSM monthly goods exports, deflated by headline CPI, then STL", "nominal RM / CPI index, quarterly sum", "STL"),
        ("imports_real_sa", "OpenDOSM monthly goods imports, deflated by headline CPI, then STL", "nominal RM / CPI index, quarterly sum", "STL"),
        ("exports_real_usd_sa", "Same goods trade in USD (EXMAUS), deflated by US CPI, then STL", "USD / CPIAUCSL, quarterly sum", "STL; 2025Q4 left out where US CPI is missing"),
        ("reer_sa", "BIS broad real effective exchange rate, Malaysia", "index, 2020=100 in the monthly source", "quarterly mean, then STL"),
        ("gdp_us", "BEA NIPA Table 1.1.6 line 1, A191RX", "millions of chained 2017 dollars, SAAR", "already SA"),
        ("gdp_japan", "Cabinet Office ESRI gaku-jk2621, Apr-Jun 2026 first preliminary", "billions of chained 2020 yen", "already SA"),
        ("gdp_china_sa", "OECD QNA SA growth index through 2024Q1, then NBS y/y indices", "index, 2011Q1=100, not yuan", "official growth rates, not STL"),
        ("gdp_singapore", "SingStat M015662 total GDP", "million chained 2015 SGD, already SA", "already SA"),
        ("gdp_eu27", "Eurostat namq_10_gdp EU27_2020 B1GQ CLV15_MEUR SCA", "million chain-linked 2015 euros, SA", "already SA"),
        ("foreign_gdp_index", "Fixed 2000-2019 COMTRADE export-share weights", "100 in the first estimation quarter", "weighted sum of log real GDP"),
        ("tpu", "Caldara-Iacoviello tpuq_published", "index", "official quarterly series, not re-averaged"),
        (
            "dummy_gfc",
            "2008Q4-2009Q2",
            "0/1",
            "zero throughout this sample" if frame["dummy_gfc"].nunique() < 2 else "varies in this sample",
        ),
        ("dummy_covid", "2020Q1-2020Q3", "0/1", ""),
        ("dummy_trade_war", "2018Q3 onward", "0/1 step", ""),
    ]
    if "log_commodity" in frame.columns:
        rows.insert(
            -3,
            (
                "commodity_sa",
                "FRED PALLFNFINDEXQ, IMF all-commodity price index",
                "index, quarterly",
                "published quarterly index, not seasonally adjusted",
            ),
        )
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
    columns = ["log_foreign_gdp", "tpu", "dummy_gfc", "dummy_covid", "dummy_trade_war"]
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


def choose_specification(
    y: pd.DataFrame, exog: pd.DataFrame, level_lags: pd.DataFrame
) -> tuple[dict, pd.DataFrame]:
    """Smallest lag that clears residual serial correlation, not the BIC lag."""
    from var_model import breusch_godfrey

    usable = level_lags.loc[level_lags["residual_df"] >= MIN_RESIDUAL_DF]
    if usable.empty:
        usable = level_lags
    bic_lag = int(usable.loc[usable["bic"].idxmin(), "lag"])
    rows = []
    johansen_parts = []
    frame = y.copy()
    for column in exog.columns:
        frame[column] = exog[column]
    for lag in usable["lag"].astype(int):
        johansen = johansen_table(y, lag - 1)
        rank = cointegrating_rank(johansen, "trace_rejects_5")
        johansen["lag_basis"] = "serial" if lag != bic_lag else "bic"
        if lag == bic_lag:
            johansen_parts.append(johansen.assign(lag_basis="bic"))
        equations = y.shape[1]
        test_lags = 4
        if equations**2 * (test_lags - lag + 1) - equations * int(rank) <= 0:
            test_lags = lag
        deterministic = "ci" if int(rank) >= 1 else "co"
        try:
            fitted = VECM(
                y,
                exog=exog.to_numpy(dtype=float),
                k_ar_diff=lag - 1,
                coint_rank=int(rank),
                deterministic=deterministic,
            ).fit()
            white = fitted.test_whiteness(nlags=test_lags, signif=0.05, adjusted=True)
            _, _, bg_p = breusch_godfrey(frame, list(exog.columns), fitted, test_lags)
            port_p = float(white.pvalue)
        except (np.linalg.LinAlgError, ValueError) as exc:
            print(f"Lag {lag} did not estimate: {exc}")
            continue
        rows.append(
            {
                "lag": lag,
                "k_ar_diff": lag - 1,
                "coint_rank": rank,
                "portmanteau_lags": test_lags,
                "portmanteau_p": port_p,
                "bg_lags": test_lags,
                "bg_p": bg_p,
                "clears_5": port_p >= 0.05 and bg_p >= 0.05,
            }
        )
        if lag != bic_lag:
            johansen_parts.append(johansen.assign(lag_basis="serial"))
    if not rows:
        raise DownloadError("No lag produced a VECM for the serial-correlation search.")
    tested = pd.DataFrame(rows)
    tested.to_csv(TABLES / "var_serial_correlation.csv", index=False)
    clear = tested.loc[tested["clears_5"]]
    if not clear.empty:
        chosen = clear.sort_values("lag").iloc[0]
        rule = "smallest lag at which the adjusted Portmanteau and Breusch-Godfrey tests both fail to reject at 5%"
    else:
        tested["min_p"] = tested[["portmanteau_p", "bg_p"]].min(axis=1)
        chosen = tested.sort_values(["min_p", "lag"], ascending=[False, True]).iloc[0]
        rule = "no lag cleared both tests at 5%; this is the lag with the larger minimum p-value"
    # Recompute rank at the chosen lag from the trace test, including rank 0.
    chosen_lag = int(chosen["lag"])
    chosen_johansen = johansen_table(y, chosen_lag - 1)
    rank = cointegrating_rank(chosen_johansen, "trace_rejects_5")
    spec = {
        "lag": chosen_lag,
        "k_ar_diff": chosen_lag - 1,
        "coint_rank": rank,
        "deterministic": "ci" if rank >= 1 else "co",
        "bic_lag": bic_lag,
        "portmanteau_lags": int(chosen["portmanteau_lags"]),
        "portmanteau_p": float(chosen["portmanteau_p"]),
        "bg_lags": int(chosen["bg_lags"]),
        "bg_p": float(chosen["bg_p"]),
        "rule": rule,
    }
    if not johansen_parts:
        johansen_parts.append(chosen_johansen.assign(lag_basis="serial"))
    print(
        f"Serial-correlation lag: {chosen_lag} (BIC lag {bic_lag}). "
        f"Portmanteau p={spec['portmanteau_p']:.3f}, Breusch-Godfrey p={spec['bg_p']:.3f}. "
        f"Trace rank {rank}. {rule}."
    )
    return spec, pd.concat(johansen_parts, ignore_index=True)


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
    spec: dict | None = None,
) -> None:
    start, end = frame["quarter"].iloc[0], frame["quarter"].iloc[-1]
    print()
    print("=" * 72)
    print(f"Estimation sample: {start} to {end} ({len(frame)} quarters).")
    print("Malaysia trade: OpenDOSM monthly exports and imports of goods, nominal RM,")
    print("  divided by the headline CPI (division 'overall'), summed to quarters, then STL.")
    print("  The CPI is a consumer price index, not a trade price index. The series is")
    print("  goods only, not national-accounts goods and services. DOSM monthly trade")
    print("  starts in 2000, so no BNM splice was used.")
    print("REER: BIS broad real index, quarterly average of the monthly series, then STL.")
    print("Foreign GDP: US from BEA Table 1.1.6 (already SA, through 2026Q2); Japan from")
    print("  the Cabinet Office real SA series (through 2026Q2); EU27 from Eurostat;")
    print("  Singapore from SingStat M015662. China is an index of official real")
    print("  GDP growth: OECD seasonally adjusted rates through 2024Q1, then NBS")
    print("  current-quarter preceding-year indices through the latest NBS quarter.")
    print("No partner was dropped from the trade-weighted index. Weights are unchanged.")
    print("TPU: tpuq_published. The recomputed monthly mean is not in the model.")
    print("Weights (mean export share of world exports, 2000-2019, then rescaled to 1):")
    for _, row in weights.iterrows():
        print(
            f"  {row['partner']}: share {row['mean_export_share']:.3f}, "
            f"weight {row['weight']:.3f}"
        )
    if not has_commodity:
        print("Commodity price index: not in the file. See the manual-download note.")
    else:
        print("Commodity prices: FRED PALLFNFINDEXQ, the IMF all-commodity index,")
        print("  quarterly as published. Entered in logs. Not seasonally adjusted.")
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
    longest = int(level_lags["lag"].max())
    if any(raw_l[name] == longest for name in raw_l):
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
        f"The BIC lag, among lags with at least {MIN_RESIDUAL_DF} residual degrees of freedom, "
        f"is {bic_l}. The lag used in the VECM is the one that clears residual serial "
        "correlation when such a lag exists."
    )
    if spec is not None:
        print(
            f"Chosen lag: {int(spec['lag'])} in levels ({int(spec['k_ar_diff'])} lagged "
            f"difference(s)), trace rank {int(spec['coint_rank'])}. "
            f"At that lag the adjusted Portmanteau p-value is {float(spec['portmanteau_p']):.3f} "
            f"and the Breusch-Godfrey p-value is {float(spec['bg_p']):.3f}."
        )
        print(spec["rule"])
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
    bic_rows = johansen.loc[johansen["lag_basis"] == "bic"] if johansen is not None else pd.DataFrame()
    if spec is not None and not bic_rows.empty:
        k_diff = int(spec["k_ar_diff"])
        trace_rank = int(spec["coint_rank"])
        eigen_rows = johansen.loc[johansen["k_ar_diff"] == k_diff]
        eigen_rank = cointegrating_rank(eigen_rows, "max_eigen_rejects_5") if not eigen_rows.empty else trace_rank
    elif not bic_rows.empty:
        trace_rank = cointegrating_rank(bic_rows, "trace_rejects_5")
        eigen_rank = cointegrating_rank(bic_rows, "max_eigen_rejects_5")
        k_diff = int(bic_rows["k_ar_diff"].iloc[0])
    else:
        return
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
            f"Chosen lag: {int(spec['lag']) if spec is not None else bic_l} in the levels VAR, which is {k_diff} lagged "
            "difference(s) in the VECM. Foreign GDP, TPU, and the GFC, COVID and trade-war "
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
    if pd.Period(start, freq="Q-DEC") <= pd.Period("2000Q1", freq="Q-DEC"):
        print(f"The estimation sample is {start} to {end}.")
        print("Malaysian trade is CPI-deflated nominal goods trade from 2000, so no")
        print("  constant-2015 historical file was required.")
        print()
    else:
        print("MANUAL DOWNLOAD - sample does not start in 2000Q1")
        print("What is missing: Malaysia real exports and imports of goods and services,")
        print("  quarterly, before the first quarter now in the file.")
        print(f"This run estimated {start} to {end}.")
        print()
    if "log_commodity" not in frame.columns:
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
