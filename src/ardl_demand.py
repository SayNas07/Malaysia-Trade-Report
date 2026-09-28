"""Single-equation ARDL/ECM demand models with Pesaran bounds tests.

Export demand excludes imports, so the foreign-GDP elasticity is not absorbed
by the processing-trade link between exports and imports. Commodity prices
enter the long run because CPI deflation turns price swings into fake volume.
Episode dummies are short-run only and are not part of the bounds test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.ardl import UECM

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "var_quarterly.csv"
TABLES = ROOT / "outputs" / "tables"
BOUNDS_CASE = 3
MAX_LAG = 6
BG_LAGS = 4
DUMMIES = ["dummy_gfc", "dummy_covid", "dummy_trade_war"]

EQUATIONS = {
    "export_demand": {
        "endog": "log_exports",
        "exog": ["log_foreign_gdp", "log_reer", "log_commodity"],
    },
    "import_demand": {
        "endog": "log_imports",
        "exog": ["log_exports", "log_reer", "log_commodity"],
    },
}


def load_sample() -> pd.DataFrame:
    frame = pd.read_csv(PROCESSED)
    needed = ["log_exports", "log_imports", "log_foreign_gdp", "log_reer", "log_commodity", *DUMMIES]
    missing = [name for name in needed if name not in frame.columns]
    if missing:
        raise SystemExit(f"var_quarterly.csv is missing {missing}. Re-run src/var_data.py.")
    sample = frame.dropna(subset=needed).copy()
    sample.index = pd.PeriodIndex(sample["quarter"], freq="Q-DEC")
    return sample


def breusch_godfrey(result) -> tuple[float, float]:
    """LM test of residual correlation. UECM results do not expose k_constant."""
    resid = np.asarray(result.resid, dtype=float).reshape(-1)
    exog = np.asarray(result.model.exog, dtype=float)[-resid.shape[0] :]
    nobs = resid.shape[0]
    lagged = np.column_stack(
        [np.concatenate([np.zeros(lag), resid[: nobs - lag]]) for lag in range(1, BG_LAGS + 1)]
    )
    fitted = sm.OLS(resid, np.column_stack([exog, lagged])).fit()
    stat = float(nobs * fitted.rsquared)
    return stat, float(stats.chi2.sf(stat, BG_LAGS))


def choose_lag(endog: pd.Series, exog: pd.DataFrame, fixed: pd.DataFrame) -> tuple[object, int, list[dict]]:
    """Smallest common lag that clears residual correlation. Otherwise the best p-value."""
    tried = []
    best = None
    for lag in range(1, MAX_LAG + 1):
        model = UECM(
            endog,
            lags=lag,
            exog=exog,
            order=lag,
            trend="c",
            fixed=fixed,
        )
        result = model.fit()
        stat, pvalue = breusch_godfrey(result)
        row = {"lag": lag, "bic": float(result.bic), "bg_stat": stat, "bg_pvalue": pvalue}
        tried.append(row)
        if best is None or pvalue > best[2]:
            best = (result, lag, pvalue)
        if pvalue >= 0.05:
            return result, lag, tried
    return best[0], best[1], tried


def five_percent_bounds(critical: pd.DataFrame) -> tuple[float, float]:
    if 95 in critical.index:
        row = critical.loc[95]
    elif 0.05 in critical.index:
        row = critical.loc[0.05]
    else:
        numeric = pd.to_numeric(pd.Index(critical.index), errors="coerce")
        target = 0.05 if np.nanmax(numeric) <= 1 else 95.0
        row = critical.iloc[int(np.nanargmin(np.abs(numeric - target)))]
    return float(row["lower"]), float(row["upper"])


def bounds_row(result, equation: str, lag: int) -> dict:
    test = result.bounds_test(case=BOUNDS_CASE)
    critical = test.critical_values
    lower, upper = five_percent_bounds(critical)
    fstat = float(np.asarray(test.statistic).reshape(-1)[0])
    if fstat > upper:
        decision = "cointegration"
    elif fstat < lower:
        decision = "no cointegration"
    else:
        decision = "inconclusive"
    pvalues = test.pvalue
    return {
        "equation": equation,
        "lag": lag,
        "case": BOUNDS_CASE,
        "f_stat": fstat,
        "lower_05": lower,
        "upper_05": upper,
        "p_lower": float(pvalues["lower"]) if "lower" in pvalues.index else np.nan,
        "p_upper": float(pvalues["upper"]) if "upper" in pvalues.index else np.nan,
        "decision": decision,
        "nobs": int(result.nobs),
    }


def error_correction_name(result) -> str:
    dependent = next(name for name, value in result.ci_params.items() if np.isclose(value, 1.0))
    return f"{dependent}.L1"


def long_run_rows(result, equation: str) -> list[dict]:
    elasticities = -result.ci_params
    errors = result.ci_bse
    rows = []
    for name in elasticities.index:
        if np.isclose(result.ci_params[name], 1.0) or pd.isna(errors[name]):
            continue
        estimate = float(elasticities[name])
        se = float(errors[name])
        tstat = estimate / se if se else np.nan
        rows.append(
            {
                "equation": equation,
                "variable": name,
                "elasticity": estimate,
                "std_error": se,
                "t": tstat,
                "p_value": float(2 * stats.norm.sf(abs(tstat))),
            }
        )
    ect_name = error_correction_name(result)
    ect = float(result.params[ect_name])
    ect_se = float(result.bse[ect_name])
    rows.append(
        {
            "equation": equation,
            "variable": "ect",
            "elasticity": ect,
            "std_error": ect_se,
            "t": float(result.tvalues[ect_name]),
            "p_value": float(result.pvalues[ect_name]),
        }
    )
    return rows


def diagnostic_row(result, equation: str, lag: int, search: list[dict]) -> dict:
    bg_stat, bg_p = breusch_godfrey(result)
    jb_stat, jb_p = stats.jarque_bera(np.asarray(result.resid))
    ect_name = error_correction_name(result)
    chosen = next(row for row in search if row["lag"] == lag)
    return {
        "equation": equation,
        "lag": lag,
        "bic": chosen["bic"],
        "bg_lags": BG_LAGS,
        "bg_stat": bg_stat,
        "bg_pvalue": bg_p,
        "jarque_bera": float(jb_stat),
        "jarque_bera_pvalue": float(jb_p),
        "ect": float(result.params[ect_name]),
        "ect_pvalue": float(result.pvalues[ect_name]),
        "lags_searched": ",".join(str(row["lag"]) for row in search),
        "bg_by_lag": ",".join(f"{row['lag']}:{row['bg_pvalue']:.3f}" for row in search),
    }


def estimate(frame: pd.DataFrame | None = None) -> dict[str, object]:
    sample = frame if frame is not None else load_sample()
    if "quarter" in sample.columns and not isinstance(sample.index, pd.PeriodIndex):
        sample = sample.dropna(subset=["log_commodity"]).copy()
        sample.index = pd.PeriodIndex(sample["quarter"], freq="Q-DEC")
    bounds_rows = []
    elasticity_rows = []
    diagnostic_rows = []
    fitted = {}
    for equation, spec in EQUATIONS.items():
        endog = sample[spec["endog"]]
        exog = sample[spec["exog"]]
        fixed = sample[DUMMIES]
        result, lag, search = choose_lag(endog, exog, fixed)
        bounds_rows.append(bounds_row(result, equation, lag))
        elasticity_rows.extend(long_run_rows(result, equation))
        diagnostic_rows.append(diagnostic_row(result, equation, lag, search))
        fitted[equation] = result
    TABLES.mkdir(parents=True, exist_ok=True)
    bounds = pd.DataFrame(bounds_rows)
    long_run = pd.DataFrame(elasticity_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    bounds.to_csv(TABLES / "ardl_bounds.csv", index=False)
    long_run.to_csv(TABLES / "ardl_long_run.csv", index=False)
    diagnostics.to_csv(TABLES / "ardl_diagnostics.csv", index=False)
    return {"bounds": bounds, "long_run": long_run, "diagnostics": diagnostics, "fitted": fitted}


def main() -> None:
    out = estimate()
    print(out["bounds"].to_string(index=False))
    print(out["long_run"].to_string(index=False))
    print(out["diagnostics"].to_string(index=False))


if __name__ == "__main__":
    main()
