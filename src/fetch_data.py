"""Download Malaysia trade, World Bank WDI, and Caldara et al. TPU data.

Run from any directory:

    python src/fetch_data.py

COMTRADE needs COMTRADE_API_KEY in the project .env file. If a source
fails, the script prints the exact file to download, the URL, and the
filename to save under data/raw/. It does not fill gaps with simulated data.
"""

from __future__ import annotations

import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
LOG_PATH = RAW / "download_log.txt"

START_YEAR = 2000
REPORTER = "458"
REPORTER_NAME = "Malaysia"
# Stay under the free-subscription record cap (100,000). The API has no
# offset parameter, so a slice larger than this is split and requested again.
MAX_RECORDS = 100_000
REQUEST_PAUSE = 1.25
MAX_ATTEMPTS = 5
TOP_N_PARTNERS = 10

WDI_DB = 2
WDI_INDICATORS = [
    "NY.GDP.MKTP.KD",
    "NE.GDI.FTOT.KD",
    "NE.EXP.GNFS.KD",
    "PA.NUS.FCRF",
    "PX.REX.REER",
]
WDI_ECONOMIES = ["MYS", "CHN", "USA", "SGP", "JPN", "EUU"]

TPU_PAGE = "https://www.matteoiacoviello.com/tpu.htm"
TPU_URL = "https://www.matteoiacoviello.com/tpu_files/tpu_web_latest.xlsx"
TPU_XLSX_NAME = "tpu_web_latest.xlsx"

COMTRADE_PORTAL = "https://comtradeplus.un.org/"
COMTRADE_KEYS = "https://comtradedeveloper.un.org/"
WDI_PORTAL = "https://databank.worldbank.org/source/world-development-indicators"

RESIDUAL_DESC = re.compile(r"\bnes\b|not elsewhere|bunkers|free zones|special categor", re.I)


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    load_dotenv(ROOT / ".env")

    failures = 0
    failures += fetch_comtrade()
    failures += fetch_wdi()
    failures += fetch_tpu()
    if failures:
        print(f"\nFinished with {failures} source(s) that need attention. See the MANUAL DOWNLOAD blocks above and {LOG_PATH.relative_to(ROOT)}.")
        return 1
    print(f"\nAll downloads finished. Log: {LOG_PATH.relative_to(ROOT)}")
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


def print_manual(source: str, url: str, filename: str, details: list[str]) -> None:
    print()
    print("=" * 72)
    print("MANUAL DOWNLOAD REQUIRED")
    print(f"Source: {source}")
    print(f"Download from: {url}")
    print(f"Save as: {filename}")
    for line in details:
        print(line)
    print("=" * 72)
    print()


def _pause(seconds: float) -> None:
    time.sleep(seconds)


def call_with_retry(func, description: str, **kwargs):
    """Call a COMTRADE helper. The library returns None on HTTP errors."""
    delay = 10.0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            result = func(**kwargs)
        except Exception as exc:
            result = None
            print(f"{description} raised {type(exc).__name__}: {exc}")
        if result is not None:
            _pause(REQUEST_PAUSE)
            return result
        if attempt == MAX_ATTEMPTS:
            break
        print(f"{description} failed (attempt {attempt}/{MAX_ATTEMPTS}). Sleeping {delay:.0f}s before retry.")
        _pause(delay)
        delay = min(delay * 2, 120)
    print(f"{description} failed after {MAX_ATTEMPTS} attempts.")
    return None


def extract_count(frame: pd.DataFrame | None) -> int | None:
    if frame is None or frame.empty or "count" not in frame.columns:
        return None
    value = frame.iloc[0]["count"]
    if pd.isna(value):
        return None
    return int(value)


def hs2_codes() -> list[str]:
    import comtradeapicall

    reference = call_with_retry(comtradeapicall.getReference, "HS reference", category="cmd:HS")
    if reference is None or reference.empty:
        return [f"{code:02d}" for code in range(1, 98) if code != 77]
    level = reference["aggrLevel"].astype(int)
    codes = reference.loc[level == 2, "id"].astype(str).str.zfill(2)
    codes = sorted(code for code in codes.unique() if re.fullmatch(r"\d{2}", code))
    return codes


def available_years(subscription_key: str) -> list[int] | None:
    import comtradeapicall

    frame = call_with_retry(
        comtradeapicall.getFinalDataAvailability,
        "COMTRADE data availability",
        subscription_key=subscription_key,
        typeCode="C",
        freqCode="A",
        clCode="HS",
        period=None,
        reporterCode=REPORTER,
    )
    if frame is None or frame.empty or "period" not in frame.columns:
        return None
    years = sorted({int(value) for value in frame["period"] if int(value) >= START_YEAR})
    return years


def _split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parts = [part for part in str(value).split(",") if part != ""]
    return parts or None


def fetch_slice(subscription_key: str, period: str, cmd: str, partner: str | None, flow: str, depth: int = 0):
    """Return one complete slice, splitting the query when the API truncates it."""
    import comtradeapicall

    if depth > 8:
        print(f"Stopped splitting {period} cmd={cmd} partner={partner} flow={flow}: recursion limit.")
        return None

    count_frame = call_with_retry(
        comtradeapicall.getCountFinalData,
        f"COMTRADE count {period} cmd={cmd} partner={partner} flow={flow}",
        subscription_key=subscription_key,
        typeCode="C",
        freqCode="A",
        clCode="HS",
        period=period,
        reporterCode=REPORTER,
        cmdCode=cmd,
        flowCode=flow,
        partnerCode=partner,
        partner2Code=None,
        customsCode=None,
        motCode=None,
        aggregateBy=None,
        breakdownMode="classic",
    )
    count = extract_count(count_frame)
    if count == 0:
        return pd.DataFrame()
    if count is not None and count > MAX_RECORDS:
        return _split_and_fetch(subscription_key, period, cmd, partner, flow, depth, count)

    frame = call_with_retry(
        comtradeapicall.getFinalData,
        f"COMTRADE data {period} cmd={cmd} partner={partner} flow={flow}",
        subscription_key=subscription_key,
        typeCode="C",
        freqCode="A",
        clCode="HS",
        period=period,
        reporterCode=REPORTER,
        cmdCode=cmd,
        flowCode=flow,
        partnerCode=partner,
        partner2Code=None,
        customsCode=None,
        motCode=None,
        maxRecords=MAX_RECORDS,
        format_output="JSON",
        aggregateBy=None,
        breakdownMode="classic",
        countOnly=None,
        includeDesc=True,
    )
    if frame is None:
        return None
    if count is not None and len(frame) != count:
        print(f"Truncated response for {period}: received {len(frame)} rows, count={count}. Splitting the query.")
        return _split_and_fetch(subscription_key, period, cmd, partner, flow, depth, count)
    if count is None:
        frame.attrs["count_unverified"] = True
    return frame


def _split_and_fetch(subscription_key: str, period: str, cmd: str, partner: str | None, flow: str, depth: int, count: int):
    periods = _split_csv(period)
    flows = _split_csv(flow)
    partners = _split_csv(partner)
    commands = _split_csv(cmd)

    if periods and len(periods) > 1:
        groups = _halve(periods)
        pieces = [( ",".join(group), cmd, partner, flow) for group in groups]
    elif flows and len(flows) > 1:
        pieces = [(period, cmd, partner, item) for item in flows]
    elif partners and len(partners) > 1:
        pieces = [(period, cmd, ",".join(group), flow) for group in _halve(partners)]
    elif cmd == "AG2" or (commands and len(commands) > 1):
        codes = commands if commands and cmd != "AG2" else hs2_codes()
        if len(codes) <= 1:
            print(f"Cannot paginate further (count={count}) for period={period} cmd={cmd} partner={partner} flow={flow}.")
            return None
        pieces = [(period, ",".join(group), partner, flow) for group in _halve(codes)]
    else:
        print(f"Cannot paginate further (count={count}) for period={period} cmd={cmd} partner={partner} flow={flow}.")
        return None

    frames = []
    unverified = False
    for item_period, item_cmd, item_partner, item_flow in pieces:
        piece = fetch_slice(subscription_key, item_period, item_cmd, item_partner, item_flow, depth + 1)
        if piece is None:
            return None
        if piece.attrs.get("count_unverified"):
            unverified = True
        if not piece.empty:
            frames.append(piece)
    if not frames:
        empty = pd.DataFrame()
        if unverified:
            empty.attrs["count_unverified"] = True
        return empty
    combined = pd.concat(frames, ignore_index=True)
    if unverified:
        combined.attrs["count_unverified"] = True
    return combined


def _halve(values: list[str]) -> list[list[str]]:
    if len(values) < 2:
        raise ValueError("Cannot split a one-item query")
    mid = len(values) // 2
    return [values[:mid], values[mid:]]


def fetch_years(subscription_key: str, years: list[int], cmd: str, partner: str | None, flow: str = "X,M"):
    frames = []
    failed_years = []
    unverified = False
    for year in years:
        print(f"  {year}: cmd={cmd} partner={partner or 'all'} flow={flow}")
        piece = fetch_slice(subscription_key, str(year), cmd, partner, flow)
        if piece is None:
            failed_years.append(year)
            continue
        if piece.attrs.get("count_unverified"):
            unverified = True
        if not piece.empty:
            frames.append(piece)
    if not frames:
        return None, failed_years, unverified
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates().reset_index(drop=True)
    return combined, failed_years, unverified


def comtrade_manual(filename: str, commodity: str, partner: str, years: str) -> None:
    print_manual(
        "UN Comtrade",
        COMTRADE_PORTAL,
        filename,
        [
            f"API keys: {COMTRADE_KEYS}",
            f"Reporter: {REPORTER_NAME} (reporter code {REPORTER})",
            "Frequency: Annual (A)",
            "Classification: HS (as reported)",
            "Breakdown: classic",
            "Flows: Exports (X) and Imports (M)",
            f"Commodity: {commodity}",
            f"Partner: {partner}",
            f"Years: {years}",
            "Save the extract as a CSV at the path above, inside this project.",
        ],
    )


def save_comtrade(frame: pd.DataFrame, path: Path, note: str, failed_years: list[int], unverified: bool) -> bool:
    if "refYear" in frame.columns:
        years = pd.to_numeric(frame["refYear"], errors="coerce").dropna()
        date_range = f"{int(years.min())}-{int(years.max())}" if not years.empty else "unknown"
    else:
        date_range = "unknown"
    extra = note
    if failed_years:
        extra += f"; failed years not included: {','.join(str(year) for year in failed_years)}"
    if unverified:
        extra += "; completeness of at least one slice could not be checked against the API count"
    frame.to_csv(path, index=False)
    log_download("UN Comtrade", path, date_range, len(frame), extra)
    return not failed_years and not unverified


def is_residual_partner(code, description, iso) -> bool:
    try:
        numeric = int(code)
    except (TypeError, ValueError):
        return True
    if numeric == 0:
        return True
    text = "" if pd.isna(description) else str(description)
    if RESIDUAL_DESC.search(text):
        return True
    iso_text = "" if pd.isna(iso) else str(iso).strip()
    return re.fullmatch(r"[A-Z]{3}", iso_text) is None


def rank_top_partners(partner_frame: pd.DataFrame) -> pd.DataFrame | None:
    required = {"partnerCode", "partnerDesc", "partnerISO", "flowCode", "primaryValue"}
    missing = required.difference(partner_frame.columns)
    if missing:
        print("Partner file is missing columns needed to rank partners: " + ", ".join(sorted(missing)))
        return None
    frame = partner_frame.copy()
    frame = frame[frame["flowCode"].isin(["X", "M"])]
    frame["primaryValue"] = pd.to_numeric(frame["primaryValue"], errors="coerce")
    keep = ~frame.apply(lambda row: is_residual_partner(row["partnerCode"], row["partnerDesc"], row["partnerISO"]), axis=1)
    frame = frame.loc[keep]
    if frame.empty:
        print("No individual partner countries remained after dropping World and residual partners.")
        return None
    exports = frame.loc[frame["flowCode"] == "X"].groupby("partnerCode")["primaryValue"].sum()
    imports = frame.loc[frame["flowCode"] == "M"].groupby("partnerCode")["primaryValue"].sum()
    names = frame.groupby("partnerCode").agg(partnerDesc=("partnerDesc", "first"), partnerISO=("partnerISO", "first"))
    ranked = names.copy()
    ranked["export_primary_value_usd"] = exports
    ranked["import_primary_value_usd"] = imports
    ranked["total_primary_value_usd"] = ranked["export_primary_value_usd"].fillna(0) + ranked["import_primary_value_usd"].fillna(0)
    ranked = ranked.sort_values(["total_primary_value_usd", "partnerCode"], ascending=[False, True])
    ranked = ranked.head(TOP_N_PARTNERS).reset_index()
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def fetch_comtrade() -> int:
    key = os.getenv("COMTRADE_API_KEY", "").strip()
    year_label = f"{START_YEAR} to latest available annual year"
    hs2_file = "data/raw/comtrade_mys_hs2_world.csv"
    partner_file = "data/raw/comtrade_mys_partner_total.csv"
    top_file = "data/raw/comtrade_mys_hs2_top10.csv"
    if not key:
        print("COMTRADE_API_KEY is empty. Set it in .env and run this script again.")
        comtrade_manual(hs2_file, "AG2 (HS 2-digit)", "World (partner code 0)", year_label)
        comtrade_manual(partner_file, "TOTAL (all products)", "All partner countries", year_label)
        comtrade_manual(
            top_file,
            "AG2 (HS 2-digit)",
            "Top 10 partner countries by the sum of export and import primaryValue, excluding World (0) and residual 'nes' partners. Rank them from the partner TOTAL extract first; do not guess the list.",
            year_label,
        )
        log_download("UN Comtrade", RAW / "comtrade_mys_hs2_world.csv", year_label, 0, "FAILED: COMTRADE_API_KEY is not set")
        return 1

    try:
        import comtradeapicall
    except ImportError as exc:
        print(f"Could not import comtradeapicall: {exc}")
        comtrade_manual(hs2_file, "AG2 (HS 2-digit)", "World (partner code 0)", year_label)
        return 1

    years = available_years(key)
    if not years:
        print("Could not read Malaysia's annual HS data-availability list, so the latest year is unknown.")
        comtrade_manual(hs2_file, "AG2 (HS 2-digit)", "World (partner code 0)", year_label)
        comtrade_manual(partner_file, "TOTAL (all products)", "All partner countries", year_label)
        comtrade_manual(top_file, "AG2 (HS 2-digit)", "Top 10 partner countries from the partner TOTAL extract", year_label)
        log_download("UN Comtrade", RAW / "comtrade_mys_hs2_world.csv", year_label, 0, "FAILED: data availability request failed")
        return 1

    print(f"COMTRADE annual HS years for Malaysia: {years[0]}-{years[-1]} ({len(years)} years)")
    year_span = f"{years[0]}-{years[-1]}"
    failures = 0

    print("Downloading HS 2-digit exports and imports, partner = World")
    hs2, failed, unverified = fetch_years(key, years, "AG2", "0")
    if hs2 is None:
        comtrade_manual(hs2_file, "AG2 (HS 2-digit)", "World (partner code 0)", year_span)
        log_download("UN Comtrade", RAW / "comtrade_mys_hs2_world.csv", year_span, 0, "FAILED: no rows saved")
        failures += 1
    else:
        ok = save_comtrade(
            hs2,
            RAW / "comtrade_mys_hs2_world.csv",
            "reporter=458; cmd=AG2; partner=0 (World); flows=X,M; breakdown=classic",
            failed,
            unverified,
        )
        if not ok:
            comtrade_manual(
                hs2_file,
                "AG2 (HS 2-digit)",
                "World (partner code 0)",
                "Missing years: " + ",".join(str(year) for year in failed) if failed else year_span,
            )
            failures += 1

    print("Downloading partner-country exports and imports, commodity = TOTAL")
    partners, failed, unverified = fetch_years(key, years, "TOTAL", None)
    if partners is None:
        comtrade_manual(partner_file, "TOTAL (all products)", "All partner countries", year_span)
        comtrade_manual(top_file, "AG2 (HS 2-digit)", "Top 10 partner countries from the partner TOTAL extract", year_span)
        log_download("UN Comtrade", RAW / "comtrade_mys_partner_total.csv", year_span, 0, "FAILED: no rows saved")
        return failures + 1

    ok = save_comtrade(
        partners,
        RAW / "comtrade_mys_partner_total.csv",
        "reporter=458; cmd=TOTAL; partner=all; flows=X,M; breakdown=classic",
        failed,
        unverified,
    )
    if not ok:
        comtrade_manual(
            partner_file,
            "TOTAL (all products)",
            "All partner countries",
            "Missing years: " + ",".join(str(year) for year in failed) if failed else year_span,
        )
        failures += 1

    ranked = rank_top_partners(partners)
    if ranked is None or ranked.empty:
        print("Top 10 partners could not be ranked from the partner extract. The HS 2-digit by partner file was not requested.")
        comtrade_manual(top_file, "AG2 (HS 2-digit)", "Top 10 partner countries from the partner TOTAL extract", year_span)
        return failures + 1

    ranked_path = PROCESSED / "comtrade_mys_top10_partners.csv"
    ranked.to_csv(ranked_path, index=False)
    log_download(
        "UN Comtrade (derived ranking, not a separate download)",
        ranked_path,
        year_span,
        len(ranked),
        "top partners by sum of X and M primaryValue; World and residual nes partners excluded",
    )
    print("Top partners:")
    print(ranked[["rank", "partnerCode", "partnerISO", "partnerDesc", "total_primary_value_usd"]].to_string(index=False))

    partner_codes = ",".join(str(int(code)) for code in ranked["partnerCode"])
    partner_label = "; ".join(f"{int(row.partnerCode)} {row.partnerDesc}" for row in ranked.itertuples(index=False))
    print("Downloading HS 2-digit exports and imports for the top partners")
    bilateral, failed, unverified = fetch_years(key, years, "AG2", partner_codes)
    if bilateral is None:
        comtrade_manual(top_file, "AG2 (HS 2-digit)", partner_label, year_span)
        log_download("UN Comtrade", RAW / "comtrade_mys_hs2_top10.csv", year_span, 0, "FAILED: no rows saved")
        return failures + 1
    ok = save_comtrade(
        bilateral,
        RAW / "comtrade_mys_hs2_top10.csv",
        f"reporter=458; cmd=AG2; partners={partner_codes}; flows=X,M; breakdown=classic",
        failed,
        unverified,
    )
    if not ok:
        comtrade_manual(
            top_file,
            "AG2 (HS 2-digit)",
            partner_label,
            "Missing years: " + ",".join(str(year) for year in failed) if failed else year_span,
        )
        failures += 1
    return failures


def lookup_name(feature_module, code: str) -> str | None:
    try:
        info = feature_module.info(code)
    except Exception as exc:
        print(f"Lookup failed for {code}: {type(exc).__name__}: {exc}")
        return None
    for row in info.items:
        if row.get("id") == code:
            return row.get("value")
    return None


def fetch_wdi() -> int:
    try:
        import wbgapi as wb
    except ImportError as exc:
        print(f"Could not import wbgapi: {exc}")
        print_manual(
            "World Bank World Development Indicators",
            WDI_PORTAL,
            "data/raw/wdi_annual.csv",
            ["Install wbgapi and rerun this script, or download the indicators listed in src/fetch_data.py."],
        )
        return 1

    wb.db = WDI_DB
    print("Checking WDI indicator codes in database 2 before downloading")
    names = {}
    missing_codes = []
    for code in WDI_INDICATORS:
        official = lookup_name(wb.series, code)
        if official is None:
            missing_codes.append(code)
            print(f"WDI indicator code not found: {code}. No substitute will be used.")
        else:
            names[code] = official
            print(f"  {code}: {official}")
    if missing_codes:
        print_manual(
            "World Bank World Development Indicators",
            "https://data.worldbank.org/indicator",
            "data/raw/wdi_annual.csv",
            [
                "These indicator codes were not in WDI database 2 and were not downloaded:",
                ", ".join(missing_codes),
                "Do not replace them with a different code unless you change the request.",
            ],
        )

    economy_names = {}
    missing_economies = []
    for code in WDI_ECONOMIES:
        official = lookup_name(wb.economy, code)
        if official is None:
            missing_economies.append(code)
            print(f"WDI economy code not found: {code}. No substitute will be used.")
        else:
            economy_names[code] = official
            print(f"  {code}: {official}")
    if missing_economies:
        print_manual(
            "World Bank World Development Indicators",
            WDI_PORTAL,
            "data/raw/wdi_annual.csv",
            [
                "These economy codes were not found and were not downloaded:",
                ", ".join(missing_economies),
                "EU in this project is the WDI aggregate EUU (European Union), not EMU (Euro area).",
            ],
        )

    if not names or not economy_names:
        log_download("World Bank WDI", RAW / "wdi_annual.csv", f"{START_YEAR}-latest", 0, "FAILED: indicator or economy check failed")
        return 1

    end_year = datetime.now().year
    last_error = None
    records = None
    delay = 10.0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            records = list(
                wb.data.fetch(
                    list(names),
                    list(economy_names),
                    time=range(START_YEAR, end_year + 1),
                    skipBlanks=False,
                    labels=True,
                    skipAggs=False,
                    db=WDI_DB,
                )
            )
            break
        except Exception as exc:
            last_error = exc
            print(f"WDI download failed (attempt {attempt}/{MAX_ATTEMPTS}): {exc}")
            if attempt < MAX_ATTEMPTS:
                _pause(delay)
                delay = min(delay * 2, 120)
    if records is None:
        print_manual(
            "World Bank World Development Indicators",
            WDI_PORTAL,
            "data/raw/wdi_annual.csv",
            [
                f"Database: WDI (db={WDI_DB})",
                "Economies: " + ", ".join(f"{code} ({economy_names[code]})" for code in economy_names),
                "Indicators:",
                *[f"  {code}: {official}" for code, official in names.items()],
                f"Frequency: annual, {START_YEAR} through the latest year the DataBank returns",
                "Columns to keep: economy_id, economy_name, series_id, series_name, year, value",
                f"Last error: {last_error}",
            ],
        )
        log_download("World Bank WDI", RAW / "wdi_annual.csv", f"{START_YEAR}-{end_year}", 0, f"FAILED: {last_error}")
        return 1

    rows = [_flatten_wdi(record) for record in records]
    frame = pd.DataFrame(rows)
    if frame.empty:
        print("WDI returned no rows.")
        print_manual(
            "World Bank World Development Indicators",
            WDI_PORTAL,
            "data/raw/wdi_annual.csv",
            ["The API returned an empty result for the verified codes. Download those series from the DataBank."],
        )
        log_download("World Bank WDI", RAW / "wdi_annual.csv", f"{START_YEAR}-{end_year}", 0, "FAILED: empty response")
        return 1

    # Drop any row whose code is not one we verified, in case the API expands the request.
    frame = frame[frame["series_id"].isin(names) & frame["economy_id"].isin(economy_names)].copy()
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame = frame[frame["year"] >= START_YEAR]
    frame = frame.sort_values(["economy_id", "series_id", "year"]).reset_index(drop=True)
    path = RAW / "wdi_annual.csv"
    frame.to_csv(path, index=False)
    observed = frame.loc[frame["value"].notna(), "year"]
    if observed.empty:
        date_range = f"{START_YEAR}-{end_year} (no non-missing values)"
    else:
        date_range = f"{int(observed.min())}-{int(observed.max())}"
    log_download(
        "World Bank WDI",
        path,
        date_range,
        len(frame),
        "db=2; economies=" + ",".join(economy_names) + "; series=" + ",".join(names),
    )
    _report_wdi_gaps(frame, names)
    return 1 if missing_codes or missing_economies else 0


def _report_wdi_gaps(frame: pd.DataFrame, names: dict[str, str]) -> None:
    thin = []
    for (economy, series), part in frame.groupby(["economy_id", "series_id"]):
        observed = part.loc[part["value"].notna(), "year"]
        if len(observed) >= 5:
            continue
        years = ", ".join(str(int(year)) for year in sorted(observed)) or "none"
        thin.append(f"  {economy} / {series} ({names.get(series, series)}): {len(observed)} observations ({years})")
    if not thin:
        return
    print("Thin or empty WDI coverage. The requested codes were kept; nothing was substituted:")
    print("\n".join(thin))


def _flatten_wdi(record: dict) -> dict:
    def part(value, field: str):
        if isinstance(value, dict):
            return value.get(field)
        return value

    time_value = record.get("time")
    time_id = part(time_value, "id")
    time_label = part(time_value, "value")
    year_text = time_label if time_label not in (None, "") else time_id
    year_text = "" if year_text is None else str(year_text)
    year_digits = re.search(r"\d{4}", year_text)
    return {
        "economy_id": part(record.get("economy"), "id"),
        "economy_name": part(record.get("economy"), "value") if isinstance(record.get("economy"), dict) else None,
        "series_id": part(record.get("series"), "id"),
        "series_name": part(record.get("series"), "value") if isinstance(record.get("series"), dict) else None,
        "time": time_id if time_id is not None else time_label,
        "year": int(year_digits.group(0)) if year_digits else pd.NA,
        "value": record.get("value"),
    }


def fetch_tpu() -> int:
    destination = RAW / TPU_XLSX_NAME
    delay = 10.0
    payload = None
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.get(
                TPU_URL,
                timeout=120,
                headers={"User-Agent": "Mozilla/5.0 (research; Malaysia trade sensitivity project)"},
            )
            response.raise_for_status()
            payload = response.content
            break
        except requests.RequestException as exc:
            last_error = exc
            print(f"TPU download failed (attempt {attempt}/{MAX_ATTEMPTS}): {exc}")
            if attempt < MAX_ATTEMPTS:
                _pause(delay)
                delay = min(delay * 2, 120)
    if payload is None or not payload.startswith(b"PK"):
        print("The TPU response was not an Excel workbook.")
        _tpu_manual(str(last_error) if payload is None else "response was not an xlsx file")
        log_download("Caldara et al. TPU", destination, "unknown", 0, "FAILED: workbook not saved")
        return 1

    destination.write_bytes(payload)
    try:
        monthly = pd.read_excel(destination, sheet_name="TPU_MONTHLY", engine="openpyxl")
    except Exception as exc:
        print(f"Saved the workbook but could not read sheet TPU_MONTHLY: {exc}")
        _tpu_manual(str(exc))
        log_download("Caldara et al. TPU", destination, "unknown", 0, "FAILED: TPU_MONTHLY sheet unreadable")
        return 1

    monthly = monthly.loc[:, ~monthly.columns.astype(str).str.startswith("Unnamed")]
    monthly = monthly.dropna(axis=1, how="all")
    if "DATE" not in monthly.columns or "TPU" not in monthly.columns:
        print(f"TPU_MONTHLY columns were {list(monthly.columns)}. Expected DATE and TPU.")
        _tpu_manual("TPU_MONTHLY sheet does not have DATE and TPU columns")
        log_download("Caldara et al. TPU", destination, "unknown", len(monthly), "FAILED: unexpected monthly columns")
        return 1

    monthly["DATE"] = pd.to_datetime(monthly["DATE"], errors="coerce")
    monthly = monthly.dropna(subset=["DATE"]).sort_values("DATE")
    monthly["year"] = monthly["DATE"].dt.year.astype(int)
    monthly["month"] = monthly["DATE"].dt.month.astype(int)
    monthly_path = RAW / "tpu_monthly.csv"
    monthly.to_csv(monthly_path, index=False)
    monthly_range = f"{monthly['DATE'].min():%Y-%m} to {monthly['DATE'].max():%Y-%m}"
    log_download(
        "Caldara, Iacoviello, Molligo, Prestipino and Raffo TPU",
        destination,
        monthly_range,
        len(monthly),
        f"original workbook from {TPU_URL}; sheet TPU_MONTHLY",
    )
    log_download(
        "Caldara, Iacoviello, Molligo, Prestipino and Raffo TPU",
        monthly_path,
        monthly_range,
        len(monthly),
        "sheet TPU_MONTHLY kept as downloaded, with year and month taken from DATE",
    )

    quarterly = _aggregate_tpu(monthly)
    published_note = "quarterly TPU is the mean of monthly TPU"
    try:
        published = pd.read_excel(destination, sheet_name="TPU_QUARTERLY", engine="openpyxl")
    except Exception as exc:
        print(f"Published TPU_QUARTERLY sheet was not read ({exc}). Quarterly file uses the monthly mean only.")
        published = None
    if published is not None:
        published = published.loc[:, ~published.columns.astype(str).str.startswith("Unnamed")].dropna(axis=1, how="all")
        parsed = _parse_published_quarterly(published)
        if parsed is None:
            print(f"Published quarterly columns were {list(published.columns)}. They were not merged.")
            published_note += "; published TPU_QUARTERLY sheet was not merged"
        else:
            quarterly = quarterly.merge(parsed, on=["year", "quarter"], how="outer")
            both = quarterly["tpu"].notna() & quarterly["tpuq_published"].notna()
            if both.any():
                gap = (quarterly.loc[both, "tpu"] - quarterly.loc[both, "tpuq_published"]).abs()
                idx = gap.idxmax()
                row = quarterly.loc[idx]
                published_note += (
                    f"; largest gap versus published TPUQ is {row['quarter_label']}: "
                    f"monthly mean {row['tpu']:.4f} vs published {row['tpuq_published']:.4f}"
                )
            else:
                published_note += "; published TPUQ did not overlap the monthly aggregate"

    quarterly["quarter_label"] = quarterly["year"].astype(int).astype(str) + "Q" + quarterly["quarter"].astype(int).astype(str)
    quarterly = quarterly.sort_values(["year", "quarter"]).reset_index(drop=True)
    quarterly_path = RAW / "tpu_quarterly.csv"
    quarterly.to_csv(quarterly_path, index=False)
    first = quarterly.iloc[0]
    last = quarterly.iloc[-1]
    q_range = f"{int(first['year'])}Q{int(first['quarter'])} to {int(last['year'])}Q{int(last['quarter'])}"
    log_download(
        "Caldara, Iacoviello, Molligo, Prestipino and Raffo TPU",
        quarterly_path,
        q_range,
        len(quarterly),
        published_note,
    )
    return 0


def _aggregate_tpu(monthly: pd.DataFrame) -> pd.DataFrame:
    frame = monthly.copy()
    frame["quarter"] = frame["DATE"].dt.quarter.astype(int)
    frame["TPU"] = pd.to_numeric(frame["TPU"], errors="coerce")
    aggregations = {"tpu": ("TPU", "mean"), "n_months": ("TPU", "count")}
    if "TPU_SHARE" in frame.columns:
        frame["TPU_SHARE"] = pd.to_numeric(frame["TPU_SHARE"], errors="coerce")
        aggregations["tpu_share_mean"] = ("TPU_SHARE", "mean")
    if "TPU_RAW" in frame.columns:
        frame["TPU_RAW"] = pd.to_numeric(frame["TPU_RAW"], errors="coerce")
        aggregations["tpu_raw_sum"] = ("TPU_RAW", "sum")
    if "N7" in frame.columns:
        frame["N7"] = pd.to_numeric(frame["N7"], errors="coerce")
        aggregations["n7_sum"] = ("N7", "sum")
    grouped = frame.groupby(["year", "quarter"], as_index=False).agg(**aggregations)
    grouped.insert(2, "quarter_label", grouped["year"].astype(str) + "Q" + grouped["quarter"].astype(str))
    return grouped


def _parse_published_quarterly(published: pd.DataFrame) -> pd.DataFrame | None:
    if "DATEQ" not in published.columns:
        return None
    frame = published.copy()
    labels = frame["DATEQ"].astype(str).str.strip()
    match = labels.str.extract(r"(?P<year>\d{4})\s*Q\s*(?P<quarter>[1-4])", expand=True)
    frame["year"] = pd.to_numeric(match["year"], errors="coerce")
    frame["quarter"] = pd.to_numeric(match["quarter"], errors="coerce")
    frame = frame.dropna(subset=["year", "quarter"])
    frame["year"] = frame["year"].astype(int)
    frame["quarter"] = frame["quarter"].astype(int)
    rename = {}
    if "TPUQ" in frame.columns:
        rename["TPUQ"] = "tpuq_published"
    if "TARIFFVOL" in frame.columns:
        rename["TARIFFVOL"] = "tariffvol_published"
    if not rename:
        return None
    out = frame.rename(columns=rename)
    columns = ["year", "quarter", *rename.values()]
    out = out[columns].drop_duplicates(["year", "quarter"])
    return out


def _tpu_manual(reason: str) -> None:
    print_manual(
        "Trade Policy Uncertainty index (Caldara, Iacoviello, Molligo, Prestipino and Raffo)",
        TPU_PAGE,
        "data/raw/tpu_web_latest.xlsx",
        [
            f"Direct file, if the link on that page still points here: {TPU_URL}",
            "Use the aggregate TPU workbook linked as 'Download our aggregate TPU data here'.",
            "Do not substitute the Baker-Bloom-Davis categorical trade index or the firm-level tpuashare file.",
            f"Reason this run did not produce the monthly and quarterly CSVs: {reason}",
        ],
    )


if __name__ == "__main__":
    sys.exit(main())
