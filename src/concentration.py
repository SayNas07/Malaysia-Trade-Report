"""Product and partner concentration of Malaysia's merchandise trade.

Reads the UN Comtrade extracts in data/raw/ and writes HHI series, ranked
exposures, and figures. Shares are computed within each year and flow.
HHI is the sum of squared shares, scaled from 0 to 10,000.

    python src/concentration.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"

BASE_YEAR = 2000
FLOWS = {"X": "Exports", "M": "Imports"}
BANDS = [
    (2007.5, 2009.5, "2008–09", "#9aa0a6"),
    (2017.5, 2019.5, "2018–19", "#c4a574"),
    (2019.5, 2021.5, "2020–21", "#d27b73"),
]
LINE_COLORS = {
    ("product", "X"): "#1f4e79",
    ("product", "M"): "#b85c38",
    ("partner", "X"): "#5b9bd5",
    ("partner", "M"): "#e0a15a",
}
EXPORT_STACK = ["#1f4e79", "#2e75b6", "#5b9bd5", "#8fb4d6", "#c5d8eb", "#d9d9d9"]
IMPORT_STACK = ["#8c3a24", "#b85c38", "#d4896a", "#e4b39a", "#f0d6c8", "#d9d9d9"]
PARTNER_SHORT = {
    "China, Hong Kong SAR": "Hong Kong SAR",
    "Rep. of Korea": "Korea",
    "United States of America": "United States",
}

HS2_FILE = RAW / "comtrade_mys_hs2_world.csv"
PARTNER_FILE = RAW / "comtrade_mys_partner_total.csv"
BILATERAL_FILE = RAW / "comtrade_mys_hs2_top10.csv"
USE_COLS = [
    "refYear",
    "flowCode",
    "partnerCode",
    "partnerISO",
    "partnerDesc",
    "cmdCode",
    "cmdDesc",
    "primaryValue",
]


def main() -> int:
    missing = [path.name for path in (HS2_FILE, PARTNER_FILE, BILATERAL_FILE) if not path.exists()]
    if missing:
        print("Missing COMTRADE extracts in data/raw/: " + ", ".join(missing))
        print("Run python src/fetch_data.py after setting COMTRADE_API_KEY.")
        return 1

    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["axes.unicode_minus"] = False

    products = load_trade(HS2_FILE)
    partners = load_trade(PARTNER_FILE)
    partners = partners.loc[~partners.apply(partner_excluded, axis=1)].copy()
    bilateral = load_trade(BILATERAL_FILE)
    bilateral = bilateral.loc[~bilateral.apply(partner_excluded, axis=1)].copy()

    latest = int(products["year"].max())
    if BASE_YEAR not in set(products["year"]) or BASE_YEAR not in set(partners["year"]):
        print(f"{BASE_YEAR} is not in the COMTRADE extracts, so the comparison year is unavailable.")
        return 1

    descriptions = (
        products.sort_values("year")
        .groupby("hs_code")["cmd_desc"]
        .last()
        .to_dict()
    )
    partner_names = (
        partners.sort_values("year")
        .groupby("partner_code")
        .agg(partner_iso=("partner_iso", "last"), partner=("partner", "last"))
    )

    product_hhi = metrics_by_year(products, ["hs_code"], "product")
    partner_hhi = metrics_by_year(partners, ["partner_code"], "partner")
    cell_hhi = metrics_by_year(bilateral, ["hs_code", "partner_code"], "product_partner")
    hhi = pd.concat([product_hhi, partner_hhi, cell_hhi], ignore_index=True)
    hhi["concentration"] = hhi["hhi"].map(classify_hhi)
    hhi = hhi.sort_values(["dimension", "flow", "year"]).reset_index(drop=True)

    totals = products.groupby(["year", "flow"], as_index=False)["value_usd"].sum()
    totals = totals.rename(columns={"value_usd": "total_usd"})

    top_products = ranked_groups(products, ["hs_code"], descriptions, latest)
    top_partners = ranked_partners(partners, partner_names, latest)
    exposures = largest_exposures(bilateral, descriptions, totals, latest)

    hhi_path = TABLES / "hhi_timeseries.csv"
    products_path = TABLES / "top10_products_2000_latest.csv"
    partners_path = TABLES / "top10_partners_2000_latest.csv"
    exposure_path = TABLES / "top10_product_partner_exposures.csv"
    hhi.to_csv(hhi_path, index=False)
    top_products.to_csv(products_path, index=False)
    top_partners.to_csv(partners_path, index=False)
    exposures.to_csv(exposure_path, index=False)

    plot_hhi(hhi, latest)
    plot_product_shares(products, descriptions, latest)
    plot_partners(partners, latest)

    print_summary(hhi, top_products, top_partners, exposures, latest)
    print()
    print("Tables:")
    for path in (hhi_path, products_path, partners_path, exposure_path):
        print(f"  {path.relative_to(ROOT).as_posix()}")
    print("Figures:")
    for name in (
        "hhi_over_time.png",
        "top5_product_shares.png",
        "top_partners_2000_vs_latest.png",
    ):
        print(f"  {(FIGURES / name).relative_to(ROOT).as_posix()}")
    return 0


def load_trade(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=USE_COLS)
    frame = frame[frame["flowCode"].isin(FLOWS)].copy()
    frame["year"] = frame["refYear"].astype(int)
    frame["flow"] = frame["flowCode"]
    frame["partner_code"] = frame["partnerCode"].astype(int)
    frame["partner_iso"] = frame["partnerISO"].astype(str)
    frame["partner"] = frame["partnerDesc"].astype(str)
    frame["hs_code"] = frame["cmdCode"].map(format_hs)
    frame["cmd_desc"] = frame["cmdDesc"].astype(str)
    frame["value_usd"] = pd.to_numeric(frame["primaryValue"], errors="coerce")
    frame = frame.dropna(subset=["value_usd"])
    frame = frame[frame["value_usd"] > 0]
    keep = ["year", "flow", "partner_code", "partner_iso", "partner", "hs_code", "cmd_desc", "value_usd"]
    return frame[keep]


def format_hs(code) -> str:
    text = str(code).strip()
    if re.fullmatch(r"\d+", text):
        return f"{int(text):02d}"
    return text


def partner_excluded(row: pd.Series) -> bool:
    if int(row["partner_code"]) == 0:
        return True
    text = "" if pd.isna(row["partner"]) else str(row["partner"])
    if text.strip().lower() == "world":
        return True
    if re.search(r"\bnes\b|not elsewhere|bunkers|free zones|special categor", text, re.I):
        return True
    iso = "" if pd.isna(row["partner_iso"]) else str(row["partner_iso"]).strip()
    return re.fullmatch(r"[A-Z]{3}", iso) is None


def classify_hhi(hhi: float) -> str:
    if hhi < 1500:
        return "unconcentrated"
    if hhi <= 2500:
        return "moderate"
    return "high"


def concentration_row(values: pd.Series) -> dict | None:
    values = values[values > 0]
    total = float(values.sum())
    if total <= 0 or values.empty:
        return None
    shares = (values / total).sort_values(ascending=False)
    cumulative = shares.cumsum()
    n_to_80 = int((cumulative < 0.80).sum() + 1)
    n_to_80 = min(n_to_80, int(len(shares)))
    return {
        "hhi": float((shares.pow(2).sum()) * 10_000),
        "top5_share": float(shares.head(5).sum()),
        "n_to_80": n_to_80,
        "n_units": int(len(shares)),
    }


def metrics_by_year(frame: pd.DataFrame, keys: list[str], dimension: str) -> pd.DataFrame:
    rows = []
    grouped = frame.groupby(["year", "flow"] + keys, as_index=False)["value_usd"].sum()
    for (year, flow), part in grouped.groupby(["year", "flow"]):
        stats = concentration_row(part.set_index(keys)["value_usd"])
        if stats is None:
            continue
        rows.append({"year": int(year), "flow": flow, "dimension": dimension, **stats})
    return pd.DataFrame(rows)


def hs_label(code: str, description: str, width: int = 46) -> str:
    text = str(description).split(";")[0].strip()
    text = re.sub(r"\s+", " ", text)
    label = f"{code} {text}"
    if len(label) <= width:
        return label
    clipped = label[: width - 1].rsplit(" ", 1)[0]
    return clipped + "..."


def partner_label(name: str) -> str:
    return PARTNER_SHORT.get(name, name)


def ranked_groups(products: pd.DataFrame, keys: list[str], descriptions: dict, latest: int) -> pd.DataFrame:
    rows = []
    collapsed = products.groupby(["year", "flow", "hs_code"], as_index=False)["value_usd"].sum()
    for year in (BASE_YEAR, latest):
        for flow in FLOWS:
            part = collapsed[(collapsed["year"] == year) & (collapsed["flow"] == flow)].copy()
            total = part["value_usd"].sum()
            part = part.sort_values(["value_usd", "hs_code"], ascending=[False, True]).head(10)
            part["share"] = part["value_usd"] / total
            part["rank"] = range(1, len(part) + 1)
            part["hs_description"] = part["hs_code"].map(descriptions)
            part["flow_label"] = FLOWS[flow]
            rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    return out[
        ["flow", "flow_label", "year", "rank", "hs_code", "hs_description", "value_usd", "share"]
    ]


def ranked_partners(partners: pd.DataFrame, names: pd.DataFrame, latest: int) -> pd.DataFrame:
    rows = []
    collapsed = partners.groupby(["year", "flow", "partner_code"], as_index=False)["value_usd"].sum()
    for year in (BASE_YEAR, latest):
        for flow in FLOWS:
            part = collapsed[(collapsed["year"] == year) & (collapsed["flow"] == flow)].copy()
            total = part["value_usd"].sum()
            part = part.sort_values(["value_usd", "partner_code"], ascending=[False, True]).head(10)
            part = part.join(names, on="partner_code")
            part["share"] = part["value_usd"] / total
            part["rank"] = range(1, len(part) + 1)
            part["flow_label"] = FLOWS[flow]
            rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    return out[
        ["flow", "flow_label", "year", "rank", "partner_code", "partner_iso", "partner", "value_usd", "share"]
    ]


def largest_exposures(
    bilateral: pd.DataFrame,
    descriptions: dict,
    totals: pd.DataFrame,
    latest: int,
) -> pd.DataFrame:
    cells = bilateral.groupby(
        ["year", "flow", "hs_code", "partner_code", "partner_iso", "partner"],
        as_index=False,
    )["value_usd"].sum()
    latest_cells = cells[cells["year"] == latest].copy()
    rows = []
    for flow in FLOWS:
        part = latest_cells[latest_cells["flow"] == flow].copy()
        top10_total = float(part["value_usd"].sum())
        world_total = float(totals.loc[(totals["year"] == latest) & (totals["flow"] == flow), "total_usd"].iloc[0])
        part = part.sort_values(["value_usd", "hs_code", "partner_code"], ascending=[False, True, True]).head(10)
        part["share_within_top10"] = part["value_usd"] / top10_total
        part["share_of_total"] = part["value_usd"] / world_total
        part["rank"] = range(1, len(part) + 1)
        part["hs_description"] = part["hs_code"].map(descriptions)
        part["flow_label"] = FLOWS[flow]
        part["exposure"] = part["hs_code"] + " to " + part["partner"]
        rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    return out[
        [
            "flow",
            "flow_label",
            "year",
            "rank",
            "hs_code",
            "hs_description",
            "partner_code",
            "partner_iso",
            "partner",
            "exposure",
            "value_usd",
            "share_of_total",
            "share_within_top10",
        ]
    ]


def shade_bands(ax) -> None:
    for start, end, label, color in BANDS:
        ax.axvspan(start, end, color=color, alpha=0.28, label=label, zorder=0)


def plot_hhi(hhi: pd.DataFrame, latest: int) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    shade_bands(ax)
    shown = hhi[hhi["dimension"].isin(["product", "partner"])]
    for (dimension, flow), part in shown.groupby(["dimension", "flow"]):
        part = part.sort_values("year")
        ax.plot(
            part["year"],
            part["hhi"],
            color=LINE_COLORS[(dimension, flow)],
            linewidth=2.1,
            label=f"{FLOWS[flow]}, {dimension}",
        )
    ax.axhline(1500, color="#666666", linewidth=0.8, linestyle="--")
    ax.axhline(2500, color="#666666", linewidth=0.8, linestyle=":")
    ax.text(2013, 1565, "1,500 moderate", fontsize=8, color="#555555")
    ax.text(2013, 2585, "2,500 high", fontsize=8, color="#555555")
    ax.set_xlim(BASE_YEAR, latest)
    ax.set_ylim(0, 3400)
    ax.set_xlabel("Year")
    ax.set_ylabel("HHI (0-10,000)")
    ax.set_title("Malaysia merchandise trade concentration")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False, ncol=2, loc="upper left")
    fig.text(
        0.01,
        0.012,
        "Source: UN Comtrade, Malaysia, annual HS. HHI = sum of squared shares x 10,000.\n"
        "Products are HS 2-digit versus the world. Partners exclude World and Areas, nes. "
        "Bands mark 2008-09, 2018-19, and 2020-21.",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIGURES / "hhi_over_time.png", dpi=160)
    plt.close(fig)


def plot_product_shares(products: pd.DataFrame, descriptions: dict, latest: int) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.4), sharex=True)
    collapsed = products.groupby(["year", "flow", "hs_code"], as_index=False)["value_usd"].sum()
    for ax, flow, colors in zip(axes, ("X", "M"), (EXPORT_STACK, IMPORT_STACK)):
        part = collapsed[collapsed["flow"] == flow]
        latest_part = part[part["year"] == latest].sort_values("value_usd", ascending=False)
        codes = latest_part["hs_code"].head(5).tolist()
        years = sorted(part["year"].unique())
        series = {code: [] for code in codes}
        other = []
        for year in years:
            year_part = part[part["year"] == year]
            total = float(year_part["value_usd"].sum())
            shares = year_part.set_index("hs_code")["value_usd"] / total
            used = 0.0
            for code in codes:
                share = float(shares.get(code, 0.0))
                series[code].append(share * 100)
                used += share
            other.append(max(0.0, (1 - used) * 100))
        labels = [hs_label(code, descriptions.get(code, code), 42) for code in codes] + ["Other HS chapters"]
        ax.stackplot(years, *[series[code] for code in codes], other, colors=colors, labels=labels)
        ax.set_ylim(0, 100)
        ax.set_ylabel(f"Share of {FLOWS[flow].lower()} (%)")
        ax.set_title(f"{FLOWS[flow]}: shares of the five largest HS chapters in {latest}")
        ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    axes[-1].set_xlabel("Year")
    axes[-1].set_xlim(BASE_YEAR, latest)
    fig.text(
        0.01,
        0.01,
        "Source: UN Comtrade. Chapters are the top five in the latest year, tracked back to 2000. "
        "Other is the rest of HS 2-digit trade with the world.",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 0.78, 1))
    fig.savefig(FIGURES / "top5_product_shares.png", dpi=160)
    plt.close(fig)


def plot_partners(partners: pd.DataFrame, latest: int) -> None:
    collapsed = partners.groupby(["year", "flow", "partner_code"], as_index=False)["value_usd"].sum()
    names = partners.sort_values("year").groupby("partner_code")["partner"].last()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 6.4), sharex=False)
    for ax, flow in zip(axes, ("X", "M")):
        latest_rows = collapsed[(collapsed["flow"] == flow) & (collapsed["year"] == latest)].copy()
        latest_total = float(latest_rows["value_usd"].sum())
        latest_rows = latest_rows.sort_values(["value_usd", "partner_code"], ascending=[False, True]).head(10)
        codes = latest_rows["partner_code"].tolist()
        base_rows = collapsed[(collapsed["flow"] == flow) & (collapsed["year"] == BASE_YEAR)]
        base_total = float(base_rows["value_usd"].sum())
        base_share = base_rows.set_index("partner_code")["value_usd"] / base_total
        labels = [partner_label(names.get(code, str(code))) for code in codes]
        y = np.arange(len(codes))
        share_2000 = [float(base_share.get(code, 0.0)) * 100 for code in codes]
        share_latest = (latest_rows["value_usd"] / latest_total * 100).tolist()
        ax.barh(y - 0.18, share_2000, height=0.36, color="#9aa0a6", label=str(BASE_YEAR))
        ax.barh(y + 0.18, share_latest, height=0.36, color="#1f4e79", label=str(latest))
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(f"Share of {FLOWS[flow].lower()} (%)")
        ax.set_title(FLOWS[flow])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Largest partner countries in {latest}, compared with {BASE_YEAR}", y=1.08)
    fig.text(
        0.01,
        -0.02,
        "Source: UN Comtrade, commodity TOTAL. Shares are among partner countries, after dropping World and Areas, nes. "
        f"Partners are the ten largest in {latest}, with their {BASE_YEAR} shares shown even when they were outside that year's top ten.",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "top_partners_2000_vs_latest.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def metric(hhi: pd.DataFrame, dimension: str, flow: str, year: int) -> pd.Series:
    row = hhi[(hhi["dimension"] == dimension) & (hhi["flow"] == flow) & (hhi["year"] == year)]
    if row.empty:
        raise ValueError(f"No {dimension} {flow} row for {year}")
    return row.iloc[0]


def print_summary(
    hhi: pd.DataFrame,
    top_products: pd.DataFrame,
    top_partners: pd.DataFrame,
    exposures: pd.DataFrame,
    latest: int,
) -> None:
    print()
    print(f"Malaysia trade concentration, {BASE_YEAR}-{latest}")
    print("HHI below 1,500 is unconcentrated, 1,500 to 2,500 is moderate, and above 2,500 is high.")
    print()
    for flow, label in FLOWS.items():
        product_then = metric(hhi, "product", flow, BASE_YEAR)
        product_now = metric(hhi, "product", flow, latest)
        partner_then = metric(hhi, "partner", flow, BASE_YEAR)
        partner_now = metric(hhi, "partner", flow, latest)
        leaders = top_products[(top_products["flow"] == flow) & (top_products["year"] == latest)].head(3)
        names = [
            f"{hs_label(row.hs_code, row.hs_description, 70)} ({row.share * 100:.0f}%)"
            for row in leaders.itertuples(index=False)
        ]
        partner_rows = top_partners[(top_partners["flow"] == flow) & (top_partners["year"] == latest)].head(3)
        partner_text = [
            f"{partner_label(row.partner)} ({row.share * 100:.0f}%)"
            for row in partner_rows.itertuples(index=False)
        ]
        print(
            f"{label} by product {path_clause(hhi, 'product', flow)} "
            f"The top five chapters were {product_then.top5_share * 100:.0f}% of {label.lower()} in {BASE_YEAR} "
            f"and {product_now.top5_share * 100:.0f}% in {latest}. "
            f"It took {int(product_then.n_to_80)} chapters to reach 80% in {BASE_YEAR} "
            f"and {int(product_now.n_to_80)} in {latest}. "
            f"The largest chapters in {latest} are {join_names(names)}."
        )
        print(
            f"{label} by partner {path_clause(hhi, 'partner', flow)} "
            f"The top five partners were {partner_then.top5_share * 100:.0f}% of identified country {label.lower()} "
            f"in {BASE_YEAR} and {partner_now.top5_share * 100:.0f}% in {latest}. "
            f"The largest partners in {latest} are {join_names(partner_text)}."
        )
        print()

    cell_x = metric(hhi, "product_partner", "X", latest)
    cell_m = metric(hhi, "product_partner", "M", latest)
    export_top = exposures[exposures["flow"] == "X"].head(3)
    import_top = exposures[exposures["flow"] == "M"].head(3)
    print(
        f"Across HS chapter and top-10 partner cells, export HHI is {cell_x.hhi:,.0f} ({cell_x.concentration}) "
        f"and import HHI is {cell_m.hhi:,.0f} ({cell_m.concentration}) in {latest}. "
        f"The largest export exposures are {exposure_phrase(export_top)}. "
        f"The largest import exposures are {exposure_phrase(import_top)}."
    )


def path_clause(hhi: pd.DataFrame, dimension: str, flow: str) -> str:
    part = hhi[(hhi["dimension"] == dimension) & (hhi["flow"] == flow)]
    then = part.loc[part["year"].idxmin()]
    now = part.loc[part["year"].idxmax()]
    low = part.loc[part["hhi"].idxmin()]
    fell_then_rose = (
        int(low["year"]) not in (int(then["year"]), int(now["year"]))
        and then["hhi"] - low["hhi"] > 150
        and now["hhi"] - low["hhi"] > 150
    )
    if fell_then_rose:
        return (
            f"fell from {then['hhi']:,.0f} ({then['concentration']}) in {int(then['year'])} "
            f"to {low['hhi']:,.0f} in {int(low['year'])}, then rose to "
            f"{now['hhi']:,.0f} ({now['concentration']}) in {int(now['year'])}."
        )
    return (
        f"moved from {then['hhi']:,.0f} ({then['concentration']}) in {int(then['year'])} "
        f"to {now['hhi']:,.0f} ({now['concentration']}) in {int(now['year'])}."
    )


def join_names(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + ", and " + names[-1]


def exposure_phrase(frame: pd.DataFrame) -> str:
    parts = [
        f"HS {row.hs_code} to {partner_label(row.partner)} ({row.share_of_total * 100:.0f}% of the flow)"
        for row in frame.itertuples(index=False)
    ]
    return join_names(parts)


if __name__ == "__main__":
    sys.exit(main())
