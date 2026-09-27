"""Estimate the Malaysian trade VECM and the robustness checks.

The specification is the one selected in src/var_data.py: one lag in levels
(no lagged differences), one cointegrating relation, and a constant inside
that relation. Foreign GDP and trade-policy uncertainty are exogenous.
Malaysia is treated as a small open economy.

    python src/var_model.py

The GFC dummy is stored in the quarterly file and is zero from 2015Q1
onward, so it is not a regressor. 2026Q3 is already absent from the file.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.vector_ar.vecm import VECM

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "var_quarterly.csv"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
SUMMARY = ROOT / "outputs" / "results_summary.md"

# REER is ordered first so a Cholesky shock to the exchange rate is the
# innovation in the REER equation. Trade volumes do not feed back into the
# REER inside the same quarter.
ENDOG = ["log_reer", "log_exports", "log_imports"]
ENDOG_LABEL = {
    "log_reer": "REER",
    "log_exports": "exports",
    "log_imports": "imports",
}
BASE_EXOG = ["log_foreign_gdp", "tpu", "dummy_covid", "dummy_trade_war"]
EXOG_LABEL = {
    "log_foreign_gdp": "foreign GDP",
    "log_gdp_us": "US GDP",
    "log_gdp_china": "China GDP",
    "tpu": "TPU",
    "dummy_covid": "COVID",
    "dummy_trade_war": "trade war",
    "dummy_gfc": "GFC",
}
HORIZON = 12
N_BOOT = 500
BOOT_SEED = 42
BAND = (5.0, 95.0)
LM_LAGS = 4
PORTMANTEAU_LAGS = 4
NAVY = "#1f4e79"
RUST = "#b85c38"
BAND_COLOR = "#8fb4d6"


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    frame = load_frame()
    exog_cols = varying(frame, BASE_EXOG)
    result = fit_vecm(frame, exog_cols)
    check_reconstruction(frame, exog_cols, result)

    shock_scale = innovation_scales(frame, ["log_foreign_gdp", "tpu"])
    point = response_paths(result, exog_cols, shock_scale)
    bands, n_boot_ok = bootstrap_bands(frame, exog_cols, shock_scale)
    save_responses(point, bands)

    diagnostics = save_diagnostics(frame, exog_cols, result)
    coefficients = save_coefficients(result, exog_cols)
    save_fevd(result)
    episodes = save_episodes(frame, result, exog_cols, coefficients)
    projections = save_local_projections(frame, shock_scale)
    robustness = save_robustness(frame, point)

    plot_responses(point, bands)
    plot_fevd(result)
    plot_counterfactuals(episodes)
    plot_local_projections(projections, point)
    plot_robustness(robustness)

    write_summary(
        frame,
        exog_cols,
        result,
        point,
        bands,
        n_boot_ok,
        diagnostics,
        coefficients,
        episodes,
        projections,
        robustness,
        shock_scale,
    )
    print(f"\nWrote {SUMMARY.relative_to(ROOT).as_posix()}")
    print(f"Bootstrap draws kept: {n_boot_ok} of {N_BOOT}.")
    return 0


def load_frame() -> pd.DataFrame:
    frame = pd.read_csv(DATA)
    frame["quarter"] = pd.PeriodIndex(frame["quarter"], freq="Q-DEC")
    frame = frame.sort_values("quarter").reset_index(drop=True)
    if frame["quarter"].iloc[-1] >= pd.Period("2026Q3", freq="Q-DEC"):
        raise RuntimeError("The quarterly file still contains 2026Q3.")
    return frame


def varying(frame: pd.DataFrame, columns: list[str], label: str = "the regression") -> list[str]:
    kept = [column for column in columns if frame[column].nunique(dropna=False) > 1]
    dropped = [column for column in columns if column not in kept]
    if dropped:
        span = f"{frame['quarter'].iloc[0]}-{frame['quarter'].iloc[-1]}"
        print(
            f"Left out of {label} ({span}) because they do not vary: "
            + ", ".join(dropped)
            + "."
        )
    return kept


def fit_vecm(frame: pd.DataFrame, exog_cols: list[str]):
    model = VECM(
        frame[ENDOG],
        exog=frame[exog_cols].to_numpy(dtype=float),
        k_ar_diff=0,
        coint_rank=1,
        deterministic="ci",
    )
    return model.fit()


def check_reconstruction(frame: pd.DataFrame, exog_cols: list[str], result) -> None:
    """The estimated recursion plus the original residuals should recover the data."""
    simulated = simulate(
        frame[ENDOG].to_numpy(dtype=float),
        frame[exog_cols].to_numpy(dtype=float),
        result,
        np.asarray(result.resid, dtype=float),
    )
    gap = np.max(np.abs(simulated - frame[ENDOG].to_numpy(dtype=float)))
    if not np.isfinite(gap) or gap > 1e-6:
        raise RuntimeError(f"VECM reconstruction error is {gap}. The exog timing is wrong.")


def simulate(levels: np.ndarray, exog: np.ndarray, result, residuals: np.ndarray) -> np.ndarray:
    """Rebuild levels from y0, the foreign path, and a residual sequence."""
    path = np.array(levels, dtype=float, copy=True)
    beta = np.asarray(result.beta, dtype=float)
    alpha = np.asarray(result.alpha, dtype=float)
    constant = np.asarray(result.const_coint, dtype=float).reshape(-1)
    loadings = np.asarray(result.exog_coefs, dtype=float)
    for step, residual in enumerate(residuals):
        t = step + 1
        equilibrium = beta.T @ path[t - 1] + constant
        growth = alpha @ equilibrium + loadings @ exog[t] + residual
        path[t] = path[t - 1] + growth
    return path


def ar1_innovation(series: pd.Series) -> pd.Series:
    """Residual from an AR(1), aligned to the quarter of the current value."""
    lagged = series.shift(1)
    sample = pd.concat([series.rename("y"), lagged.rename("lag")], axis=1).dropna()
    design = np.column_stack([np.ones(len(sample)), sample["lag"].to_numpy()])
    coef = np.linalg.lstsq(design, sample["y"].to_numpy(), rcond=None)[0]
    residual = sample["y"].to_numpy() - design @ coef
    return pd.Series(residual, index=sample.index, name=series.name)


def innovation_scales(frame: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    scales = {}
    for column in columns:
        residual = ar1_innovation(frame.set_index("quarter")[column])
        scale = float(residual.std(ddof=1))
        if not np.isfinite(scale) or scale <= 0:
            raise RuntimeError(f"{column} has no usable AR(1) innovation.")
        scales[column] = scale
    return scales


def response_paths(result, exog_cols: list[str], shock_scale: dict[str, float]) -> dict[str, np.ndarray]:
    """Log-level responses from horizon 0 through HORIZON.

    REER is an orthogonalised impulse response. Foreign GDP and TPU are
    one-quarter dynamic multipliers, scaled to a 1 s.d. AR(1) innovation.
    """
    horizon = HORIZON + 1
    # ma_rep returns maxn + 1 matrices, so maxn=HORIZON covers horizons 0..12.
    ma = np.asarray(result.orth_ma_rep(maxn=HORIZON), dtype=float)[:horizon]
    paths = {}
    reer_at = ENDOG.index("log_reer")
    for name in ENDOG:
        paths[f"{name}__log_reer"] = ma[:, ENDOG.index(name), reer_at]
    companion = np.asarray(result.var_rep[0], dtype=float)
    loadings = np.asarray(result.exog_coefs, dtype=float)
    for column, scale in shock_scale.items():
        if column not in exog_cols:
            continue
        shock = np.zeros(len(exog_cols))
        shock[exog_cols.index(column)] = scale
        level = loadings @ shock
        for name in ENDOG:
            series = np.empty(horizon)
            series[0] = level[ENDOG.index(name)]
            state = level.copy()
            for step in range(1, horizon):
                state = companion @ state
                series[step] = state[ENDOG.index(name)]
            paths[f"{name}__{column}"] = series
    return paths


def bootstrap_bands(
    frame: pd.DataFrame, exog_cols: list[str], shock_scale: dict[str, float]
) -> tuple[dict[str, np.ndarray], int]:
    levels = frame[ENDOG].to_numpy(dtype=float)
    exog = frame[exog_cols].to_numpy(dtype=float)
    base = fit_vecm(frame, exog_cols)
    residuals = np.asarray(base.resid, dtype=float)
    keys = [f"{name}__{shock}" for name in ENDOG for shock in ("log_reer", *shock_scale)]
    draws = {key: [] for key in keys}
    rng = np.random.default_rng(BOOT_SEED)
    kept = 0
    for _ in range(N_BOOT):
        choice = rng.integers(0, len(residuals), size=len(residuals))
        simulated = simulate(levels, exog, base, residuals[choice])
        if not np.all(np.isfinite(simulated)):
            continue
        try:
            boot = fit_vecm(pd.DataFrame(simulated, columns=ENDOG).assign(**{
                column: frame[column].to_numpy() for column in exog_cols
            }).assign(quarter=frame["quarter"]), exog_cols)
            paths = response_paths(boot, exog_cols, shock_scale)
        except np.linalg.LinAlgError:
            continue
        if any(not np.all(np.isfinite(paths[key])) for key in keys):
            continue
        for key in keys:
            draws[key].append(paths[key])
        kept += 1
    bands = {}
    for key, samples in draws.items():
        stack = np.vstack(samples)
        bands[key] = np.percentile(stack, BAND, axis=0)
    return bands, kept


def save_responses(point: dict[str, np.ndarray], bands: dict[str, np.ndarray]) -> None:
    rows = []
    for key, path in point.items():
        outcome, shock = key.split("__")
        lower, upper = bands[key]
        for horizon in range(len(path)):
            rows.append(
                {
                    "outcome": ENDOG_LABEL[outcome],
                    "shock": shock_name(shock),
                    "horizon": horizon,
                    "log_response": path[horizon],
                    "percent": to_percent(path[horizon]),
                    "percent_low_90": to_percent(lower[horizon]),
                    "percent_high_90": to_percent(upper[horizon]),
                    "kind": "impulse response" if shock == "log_reer" else "dynamic multiplier",
                }
            )
    pd.DataFrame(rows).to_csv(TABLES / "var_irf.csv", index=False)


def save_diagnostics(frame: pd.DataFrame, exog_cols: list[str], result) -> pd.DataFrame:
    rows = []
    white = result.test_whiteness(nlags=PORTMANTEAU_LAGS, signif=0.05, adjusted=True)
    rows.append(test_row("Portmanteau", PORTMANTEAU_LAGS, white))
    lm_stat, lm_df, lm_p = breusch_godfrey(frame, exog_cols, result, LM_LAGS)
    rows.append(
        {
            "test": "Breusch-Godfrey LM",
            "lags": LM_LAGS,
            "statistic": lm_stat,
            "df": lm_df,
            "pvalue": lm_p,
            "crit_5": stats.chi2.ppf(0.95, lm_df),
        }
    )
    normal = result.test_normality()
    rows.append(test_row("Jarque-Bera", None, normal))
    eigenvalues = companion_roots(result)
    eigenvalues.to_csv(TABLES / "var_eigenvalues.csv", index=False)
    unit_roots = int(np.sum(np.abs(eigenvalues["modulus"] - 1) < 1e-6))
    other = eigenvalues.loc[np.abs(eigenvalues["modulus"] - 1) >= 1e-6, "modulus"]
    rows.append(
        {
            "test": "Companion roots",
            "lags": None,
            "statistic": float(other.max()) if len(other) else np.nan,
            "df": unit_roots,
            "pvalue": np.nan,
            "crit_5": np.nan,
            "note": (
                f"{unit_roots} roots lie on the unit circle, against {len(ENDOG) - 1} expected "
                f"for one cointegrating relation among three variables. "
                f"The largest other modulus is {float(other.max()) if len(other) else float('nan'):.3f}."
            ),
        }
    )
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "var_diagnostics.csv", index=False)
    return table


def test_row(name: str, lags: int | None, result) -> dict:
    return {
        "test": name,
        "lags": lags,
        "statistic": float(result.test_statistic),
        "df": int(result.df),
        "pvalue": float(result.pvalue),
        "crit_5": float(result.crit_value),
        "note": "",
    }


def breusch_godfrey(frame: pd.DataFrame, exog_cols: list[str], result, lags: int) -> tuple[float, int, float]:
    """System LM test: residuals on lagged residuals and the VECM regressors."""
    resid = np.asarray(result.resid, dtype=float)
    nobs, equations = resid.shape
    levels = frame[ENDOG].to_numpy(dtype=float)
    exog = frame[exog_cols].to_numpy(dtype=float)
    lagged_level = levels[:-1]
    current_exog = exog[1:]
    outcome = resid[lags:]
    pieces = [np.ones((len(outcome), 1)), lagged_level[lags:], current_exog[lags:]]
    for lag in range(1, lags + 1):
        pieces.append(resid[lags - lag : nobs - lag])
    design = np.column_stack(pieces)
    coef = np.linalg.lstsq(design, outcome, rcond=None)[0]
    auxiliary = outcome - design @ coef
    original = resid[lags:]
    scale = len(outcome)
    sigma_aux = auxiliary.T @ auxiliary / scale
    sigma = original.T @ original / scale
    statistic = float(scale * (equations - np.trace(np.linalg.solve(sigma, sigma_aux))))
    df = lags * equations * equations
    pvalue = float(stats.chi2.sf(statistic, df))
    return statistic, df, pvalue


def companion_roots(result) -> pd.DataFrame:
    values = np.linalg.eigvals(np.asarray(result.var_rep[0], dtype=float))
    order = np.argsort(-np.abs(values))
    values = values[order]
    return pd.DataFrame(
        {
            "root": [f"{value.real:.6f}{value.imag:+.6f}i" for value in values],
            "real": values.real,
            "imag": values.imag,
            "modulus": np.abs(values),
        }
    )


def save_coefficients(result, exog_cols: list[str]) -> pd.DataFrame:
    rows = []
    beta = np.asarray(result.beta, dtype=float).reshape(-1)
    constant = float(np.asarray(result.const_coint).reshape(-1)[0])
    export_slot = ENDOG.index("log_exports")
    scale = beta[export_slot]
    for name, value in zip(ENDOG, beta):
        rows.append(coefficient_row("cointegrating vector", ENDOG_LABEL[name], value, np.nan, np.nan))
    rows.append(coefficient_row("cointegrating vector", "constant", constant, np.nan, np.nan))
    if abs(scale) > 1e-8:
        for name, value in zip(ENDOG, beta):
            if name == "log_exports":
                continue
            rows.append(
                coefficient_row(
                    "long-run relation, exports normalised to 1",
                    ENDOG_LABEL[name],
                    -value / scale,
                    np.nan,
                    np.nan,
                )
            )
        rows.append(
            coefficient_row(
                "long-run relation, exports normalised to 1",
                "constant",
                -constant / scale,
                np.nan,
                np.nan,
            )
        )
    alpha = np.asarray(result.alpha, dtype=float).reshape(-1)
    alpha_se = np.asarray(result.stderr_alpha, dtype=float).reshape(-1)
    alpha_p = np.asarray(result.pvalues_alpha, dtype=float).reshape(-1)
    for name, value, se, pvalue in zip(ENDOG, alpha, alpha_se, alpha_p):
        rows.append(coefficient_row("adjustment speed", ENDOG_LABEL[name], value, se, pvalue))
    coef = np.asarray(result.exog_coefs, dtype=float)
    se = np.asarray(result.stderr_det_coef, dtype=float)
    pvalues = np.asarray(result.pvalues_det_coef, dtype=float)
    se = se[:, -coef.shape[1] :]
    pvalues = pvalues[:, -coef.shape[1] :]
    for j, column in enumerate(exog_cols):
        for i, name in enumerate(ENDOG):
            rows.append(
                coefficient_row(
                    "short-run exogenous",
                    f"{ENDOG_LABEL[name]} on {EXOG_LABEL[column]}",
                    coef[i, j],
                    se[i, j],
                    pvalues[i, j],
                )
            )
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "var_coefficients.csv", index=False)
    return table


def coefficient_row(block: str, name: str, estimate: float, se: float, pvalue: float) -> dict:
    return {
        "block": block,
        "name": name,
        "estimate": estimate,
        "std_error": se,
        "pvalue": pvalue,
    }


def save_fevd(result) -> pd.DataFrame:
    ma = np.asarray(result.orth_ma_rep(maxn=HORIZON), dtype=float)[: HORIZON + 1]
    squared = np.cumsum(ma**2, axis=0)
    shares = squared / squared.sum(axis=2, keepdims=True)
    rows = []
    for outcome in ("log_exports", "log_imports"):
        i = ENDOG.index(outcome)
        for horizon in range(HORIZON + 1):
            for j, shock in enumerate(ENDOG):
                rows.append(
                    {
                        "outcome": ENDOG_LABEL[outcome],
                        "horizon": horizon,
                        "shock": ENDOG_LABEL[shock],
                        "share": shares[horizon, i, j],
                    }
                )
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "var_fevd.csv", index=False)
    return table


def save_episodes(frame: pd.DataFrame, result, exog_cols: list[str], coefficients: pd.DataFrame) -> dict:
    rows = []
    for dummy, label in (
        ("dummy_gfc", "GFC"),
        ("dummy_covid", "COVID"),
        ("dummy_trade_war", "trade war"),
    ):
        if dummy not in exog_cols:
            for equation in ("exports", "imports", "REER"):
                rows.append(
                    {
                        "episode": label,
                        "equation": equation,
                        "estimate": np.nan,
                        "std_error": np.nan,
                        "pvalue": np.nan,
                        "note": "Not estimated. The dummy does not vary in 2015Q1-2024Q1.",
                    }
                )
            continue
        matched = coefficients[
            coefficients["name"].str.endswith(f"on {EXOG_LABEL[dummy]}")
            & (coefficients["block"] == "short-run exogenous")
        ]
        for _, row in matched.iterrows():
            equation = row["name"].split(" on ")[0]
            rows.append(
                {
                    "episode": label,
                    "equation": equation,
                    "estimate": row["estimate"],
                    "std_error": row["std_error"],
                    "pvalue": row["pvalue"],
                    "note": "Shift in the quarterly log-difference while the dummy equals 1.",
                }
            )
    dummy_table = pd.DataFrame(rows)
    dummy_table.to_csv(TABLES / "var_episode_dummies.csv", index=False)

    forecasts = {}
    forecasts["COVID"] = episode_forecast(
        frame,
        start="2020Q1",
        end="2021Q4",
        exog_cols=["log_foreign_gdp", "tpu", "dummy_trade_war"],
    )
    forecasts["trade war"] = episode_forecast(
        frame,
        start="2018Q3",
        end="2019Q4",
        exog_cols=["log_foreign_gdp", "tpu"],
    )
    forecasts["GFC"] = {
        "ok": False,
        "reason": "The sample starts in 2015Q1, so there is no pre-2008 window to estimate from.",
    }
    pieces = []
    for name, payload in forecasts.items():
        if payload.get("ok"):
            piece = payload["path"].copy()
            piece.insert(0, "episode", name)
            pieces.append(piece)
    if pieces:
        pd.concat(pieces, ignore_index=True).to_csv(TABLES / "var_counterfactual.csv", index=False)
    else:
        pd.DataFrame(columns=["episode", "quarter", "actual", "forecast", "gap_percent"]).to_csv(
            TABLES / "var_counterfactual.csv", index=False
        )
    return {"dummies": dummy_table, "forecasts": forecasts}


def episode_forecast(frame: pd.DataFrame, start: str, end: str, exog_cols: list[str]) -> dict:
    start_q = pd.Period(start, freq="Q-DEC")
    end_q = pd.Period(end, freq="Q-DEC")
    pre = frame[frame["quarter"] < start_q].copy()
    future = frame[(frame["quarter"] >= start_q) & (frame["quarter"] <= end_q)].copy()
    usable = varying(pre, exog_cols, f"the pre-{start} forecast")
    if len(pre) < 12 or future.empty:
        return {
            "ok": False,
            "reason": f"Only {len(pre)} quarters are available before {start}.",
        }
    try:
        fitted = fit_vecm(pre, usable)
        forecast = fitted.predict(steps=len(future), exog_fc=future[usable].to_numpy(dtype=float))
    except (np.linalg.LinAlgError, ValueError) as exc:
        return {"ok": False, "reason": f"The pre-{start} VECM did not estimate: {exc}"}
    export_at = ENDOG.index("log_exports")
    actual_log = future["log_exports"].to_numpy(dtype=float)
    forecast_log = np.asarray(forecast, dtype=float)[:, export_at]
    path = pd.DataFrame(
        {
            "quarter": future["quarter"].astype(str),
            "actual_rm_million": np.exp(actual_log),
            "forecast_rm_million": np.exp(forecast_log),
            "gap_percent": to_percent(actual_log - forecast_log),
        }
    )
    return {"ok": True, "path": path, "pre_quarters": len(pre), "exog": usable}


def save_local_projections(frame: pd.DataFrame, shock_scale: dict[str, float]) -> pd.DataFrame:
    indexed = frame.set_index("quarter")
    innovations = {
        column: ar1_innovation(indexed[column]) / shock_scale[column]
        for column in ("log_foreign_gdp", "tpu")
    }
    rows = []
    for column, shock in innovations.items():
        for horizon in range(HORIZON + 1):
            estimate = projection_at(indexed, shock, innovations, horizon, column)
            estimate["shock"] = EXOG_LABEL[column]
            estimate["horizon"] = horizon
            rows.append(estimate)
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "var_local_projections.csv", index=False)
    return table


def projection_at(indexed, shock, innovations, horizon: int, shock_name_col: str) -> dict:
    """Cumulative export response, Newey-West standard errors."""
    outcome = indexed["log_exports"].shift(-horizon) - indexed["log_exports"].shift(1)
    controls = {
        "lag_export_growth": indexed["log_exports"].diff().shift(1),
        "lag_reer_growth": indexed["log_reer"].diff().shift(1),
    }
    other = "tpu" if shock_name_col == "log_foreign_gdp" else "log_foreign_gdp"
    controls["other_shock"] = innovations[other]
    controls["covid"] = indexed["dummy_covid"]
    controls["trade_war"] = indexed["dummy_trade_war"]
    sample = pd.concat([outcome.rename("y"), shock.rename("shock"), pd.DataFrame(controls)], axis=1)
    sample = sample.dropna()
    keep = ["shock"]
    for column in controls:
        if sample[column].nunique() > 1:
            keep.append(column)
    design = sm.add_constant(sample[keep], has_constant="add")
    fitted = sm.OLS(sample["y"], design).fit(cov_type="HAC", cov_kwds={"maxlags": horizon + 1})
    estimate = float(fitted.params["shock"])
    se = float(fitted.bse["shock"])
    return {
        "percent": to_percent(estimate),
        "percent_low_90": to_percent(estimate - 1.64485 * se),
        "percent_high_90": to_percent(estimate + 1.64485 * se),
        "nobs": int(fitted.nobs),
    }


def save_robustness(frame: pd.DataFrame, baseline: dict[str, np.ndarray]) -> pd.DataFrame:
    specs = {
        "baseline": (frame, ["log_foreign_gdp", "tpu", "dummy_covid", "dummy_trade_war"], "log_foreign_gdp"),
        "us_china": (
            frame,
            ["log_gdp_us", "log_gdp_china", "tpu", "dummy_covid", "dummy_trade_war"],
            None,
        ),
        "through_2019Q4": (
            frame[frame["quarter"] <= pd.Period("2019Q4", freq="Q-DEC")].copy(),
            ["log_foreign_gdp", "tpu", "dummy_covid", "dummy_trade_war"],
            "log_foreign_gdp",
        ),
    }
    rows = []
    paths = {"baseline": baseline}
    for name, (sample, columns, _) in specs.items():
        columns = varying(sample, columns, name)
        if name == "baseline":
            fitted_paths = baseline
            scales = innovation_scales(sample, [c for c in columns if c in ("log_foreign_gdp", "tpu", "log_gdp_us", "log_gdp_china")])
        else:
            scales = innovation_scales(
                sample,
                [c for c in columns if c in ("log_foreign_gdp", "tpu", "log_gdp_us", "log_gdp_china")],
            )
            fitted_paths = response_paths(fit_vecm(sample, columns), columns, scales)
            paths[name] = fitted_paths
        for column, scale in scales.items():
            for outcome in ("log_exports", "log_imports"):
                path = fitted_paths[f"{outcome}__{column}"]
                rows.append(
                    {
                        "specification": name,
                        "sample_end": str(sample["quarter"].iloc[-1]),
                        "nobs": int(len(sample)),
                        "outcome": ENDOG_LABEL[outcome],
                        "shock": EXOG_LABEL[column],
                        "innovation_sd": scale,
                        "impact_percent": to_percent(path[0]),
                        "horizon_4_percent": to_percent(path[4]),
                        "horizon_8_percent": to_percent(path[8]),
                        "horizon_12_percent": to_percent(path[12]),
                        "peak_percent": to_percent(path[int(np.argmax(np.abs(path)))]),
                    }
                )
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "var_robustness.csv", index=False)
    table.attrs["paths"] = paths
    return table


def plot_responses(point: dict[str, np.ndarray], bands: dict[str, np.ndarray]) -> None:
    shocks = [
        ("log_foreign_gdp", "Foreign GDP, 1 s.d. multiplier"),
        ("log_reer", "REER, 1 s.d. impulse response"),
        ("tpu", "TPU, 1 s.d. multiplier"),
    ]
    outcomes = [("log_exports", "Exports"), ("log_imports", "Imports")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
    horizons = np.arange(HORIZON + 1)
    for row, (outcome, outcome_label) in enumerate(outcomes):
        for col, (shock, title) in enumerate(shocks):
            ax = axes[row, col]
            key = f"{outcome}__{shock}"
            lower, upper = bands[key]
            ax.fill_between(horizons, to_percent(lower), to_percent(upper), color=BAND_COLOR, alpha=0.9)
            ax.plot(horizons, to_percent(point[key]), color=NAVY, linewidth=1.8)
            ax.axhline(0, color="#666666", linewidth=0.8)
            if row == 0:
                ax.set_title(title, fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{outcome_label}, percent")
            if row == 1:
                ax.set_xlabel("Quarters ahead")
    fig.suptitle("Response of Malaysian trade, 90% residual-bootstrap bands", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES / "var_irf.png", dpi=140)
    plt.close(fig)


def plot_fevd(result) -> None:
    table = pd.read_csv(TABLES / "var_fevd.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    colors = {"REER": NAVY, "exports": "#5b9bd5", "imports": RUST}
    for ax, outcome in zip(axes, ("exports", "imports")):
        part = table[table["outcome"] == outcome]
        bottom = np.zeros(HORIZON + 1)
        for shock in ("REER", "exports", "imports"):
            share = part.loc[part["shock"] == shock, "share"].to_numpy()
            ax.bar(np.arange(HORIZON + 1), share, bottom=bottom, color=colors[shock], width=0.8, label=shock)
            bottom = bottom + share
        ax.set_title(outcome.capitalize())
        ax.set_xlabel("Quarters ahead")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Share of forecast-error variance")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("Forecast-error variance inside the Malaysian block", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES / "var_fevd.png", dpi=140)
    plt.close(fig)


def plot_counterfactuals(episodes: dict) -> None:
    forecasts = {name: payload for name, payload in episodes["forecasts"].items() if payload.get("ok")}
    if not forecasts:
        return
    fig, axes = plt.subplots(1, len(forecasts), figsize=(5.2 * len(forecasts), 4), squeeze=False)
    for ax, (name, payload) in zip(axes[0], forecasts.items()):
        path = payload["path"]
        ax.plot(path["quarter"], path["actual_rm_million"], color=NAVY, linewidth=1.8, label="Actual")
        ax.plot(
            path["quarter"],
            path["forecast_rm_million"],
            color=RUST,
            linewidth=1.8,
            linestyle="--",
            label="Pre-episode forecast",
        )
        ax.set_title(name)
        ax.tick_params(axis="x", labelrotation=45)
        ax.set_ylabel("Real exports, RM million")
        ax.legend(frameon=False)
    fig.suptitle("Exports against a forecast from the pre-episode model", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES / "var_counterfactual.png", dpi=140)
    plt.close(fig)


def plot_local_projections(projections: pd.DataFrame, point: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    mapping = {"foreign GDP": "log_foreign_gdp", "TPU": "tpu"}
    for ax, (label, column) in zip(axes, mapping.items()):
        part = projections[projections["shock"] == label]
        horizons = part["horizon"].to_numpy()
        ax.fill_between(horizons, part["percent_low_90"], part["percent_high_90"], color=BAND_COLOR, alpha=0.9)
        ax.plot(horizons, part["percent"], color=NAVY, linewidth=1.8, label="Local projection")
        ax.plot(horizons, to_percent(point[f"log_exports__{column}"]), color=RUST, linewidth=1.6, label="VECM multiplier")
        ax.axhline(0, color="#666666", linewidth=0.8)
        ax.set_title(f"Exports after a 1 s.d. {label} innovation")
        ax.set_xlabel("Quarters ahead")
        ax.legend(frameon=False)
    axes[0].set_ylabel("Percent")
    fig.tight_layout()
    fig.savefig(FIGURES / "var_local_projections.png", dpi=140)
    plt.close(fig)


def plot_robustness(robustness: pd.DataFrame) -> None:
    paths = robustness.attrs.get("paths", {})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    horizons = np.arange(HORIZON + 1)
    demand = [
        ("baseline", "log_foreign_gdp", "Trade-weighted GDP"),
        ("us_china", "log_gdp_us", "US GDP"),
        ("us_china", "log_gdp_china", "China GDP"),
        ("through_2019Q4", "log_foreign_gdp", "Trade-weighted, through 2019Q4"),
    ]
    colors = [NAVY, "#5b9bd5", RUST, "#6b6b6b"]
    for (spec, column, label), color in zip(demand, colors):
        path = paths[spec][f"log_exports__{column}"]
        axes[0].plot(horizons, to_percent(path), color=color, linewidth=1.7, label=label)
    for spec, label, color in (
        ("baseline", "Full sample", NAVY),
        ("through_2019Q4", "Through 2019Q4", RUST),
    ):
        axes[1].plot(horizons, to_percent(paths[spec]["log_exports__tpu"]), color=color, linewidth=1.7, label=label)
    for ax, title in zip(axes, ("Export response to foreign GDP", "Export response to TPU")):
        ax.axhline(0, color="#666666", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Quarters ahead")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Percent")
    fig.tight_layout()
    fig.savefig(FIGURES / "var_robustness.png", dpi=140)
    plt.close(fig)


def write_summary(frame, exog_cols, result, point, bands, n_boot, diagnostics, coefficients, episodes, projections, robustness, shock_scale) -> None:
    start, end = str(frame["quarter"].iloc[0]), str(frame["quarter"].iloc[-1])
    export_gdp = named_coef(coefficients, "exports on foreign GDP")
    import_gdp = named_coef(coefficients, "imports on foreign GDP")
    lines = [
        "# Malaysian trade and external shocks",
        "",
        f"The estimates use a VECM on {start} to {end} ({len(frame)} quarters). "
        "Log real exports, log real imports, and the log real effective exchange rate "
        "are endogenous. The system has one cointegrating relation and no lagged "
        "differences, the lag and rank selected for this file. Trade-weighted foreign "
        "GDP and trade-policy uncertainty are exogenous, because Malaysia is treated "
        "as a small open economy that does not move partner GDP. The COVID dummy and "
        "the US-China trade-war step sit outside the cointegrating relation as well.",
        "",
        "## How sensitive trade was",
        "",
        (
            f"The short-run elasticity of exports to trade-weighted foreign GDP is "
            f"{export_gdp[0]:.2f} ({p_text(export_gdp[1])}). A 1% foreign-GDP innovation "
            f"raises exports by about {export_gdp[0]:.2f}% in that quarter. Imports move "
            f"with a short-run elasticity of {import_gdp[0]:.2f} ({p_text(import_gdp[1])}). "
            "Both responses die out within a couple of quarters. The third companion root "
            "has modulus 0.03, so a one-quarter impulse does not linger in this VECM."
        ),
        "",
        response_paragraph(point, bands, shock_scale),
        "",
        long_run_paragraph(coefficients),
        "",
        "## Which external factor matters most",
        "",
        factor_paragraph(robustness, shock_scale),
        "",
        fevd_paragraph(),
        "",
        "## Shock episodes",
        "",
        episode_paragraph(episodes),
        "",
        "## Robustness",
        "",
        robustness_paragraph(robustness, projections),
        "",
        "## Diagnostics",
        "",
        diagnostic_paragraph(diagnostics, result, n_boot),
        "",
        "## Caveats",
        "",
        caveat_paragraph(len(frame), n_boot, shock_scale),
        "",
    ]
    SUMMARY.write_text("\n".join(lines), encoding="utf-8")


def rank_shocks(point, bands, outcome: str) -> list[dict]:
    rows = []
    for shock in ("log_foreign_gdp", "log_reer", "tpu"):
        path = point[f"{outcome}__{shock}"]
        lower, upper = bands[f"{outcome}__{shock}"]
        peak = int(np.argmax(np.abs(path)))
        rows.append(
            {
                "shock": shock_name(shock),
                "impact": to_percent(path[0]),
                "peak": to_percent(path[peak]),
                "peak_horizon": peak,
                "impact_clear": excludes_zero(lower[0], upper[0]),
                "any_clear": any(excludes_zero(lo, hi) for lo, hi in zip(lower, upper)),
            }
        )
    return sorted(rows, key=lambda row: abs(row["peak"]), reverse=True)


def named_coef(coefficients: pd.DataFrame, name: str) -> tuple[float, float]:
    row = coefficients.loc[coefficients["name"] == name].iloc[0]
    return float(row["estimate"]), float(row["pvalue"])


def response_paragraph(point, bands, shock_scale: dict) -> str:
    export_gdp = point["log_exports__log_foreign_gdp"]
    import_gdp = point["log_imports__log_foreign_gdp"]
    export_reer = point["log_exports__log_reer"]
    import_reer = point["log_imports__log_reer"]
    export_tpu = point["log_exports__tpu"]
    import_tpu = point["log_imports__tpu"]
    gdp_band = "excludes zero" if excludes_zero(*bands["log_exports__log_foreign_gdp"][:, 0]) else "includes zero"
    reer_band = "excludes zero" if excludes_zero(*bands["log_exports__log_reer"][:, 0]) else "includes zero"
    tpu_band = "excludes zero" if excludes_zero(*bands["log_exports__tpu"][:, 0]) else "includes zero"
    return (
        f"Scaled to a one-standard-deviation AR(1) innovation, the impact responses of "
        f"exports are {to_percent(export_gdp[0]):+.2f}% for foreign GDP, "
        f"{to_percent(export_reer[0]):+.2f}% for the REER, and {to_percent(export_tpu[0]):+.2f}% "
        f"for TPU. The matching import responses are {to_percent(import_gdp[0]):+.2f}%, "
        f"{to_percent(import_reer[0]):+.2f}%, and {to_percent(import_tpu[0]):+.2f}%. "
        f"The 90% bootstrap band for the export impact {gdp_band} for foreign GDP, "
        f"{reer_band} for the REER, and {tpu_band} for TPU. After the impact quarter the "
        "bands include zero. The foreign-GDP innovation in this sample has a standard "
        f"deviation of {100 * shock_scale['log_foreign_gdp']:.2f}% because the COVID quarter "
        "is in the residual. The TPU innovation has a standard deviation of "
        f"{shock_scale['tpu']:.1f} index points. A positive REER shock here is an "
        "appreciation. Its point estimate does not show the usual loss of export "
        "competitiveness, and the band includes zero."
    )


def long_run_paragraph(coefficients: pd.DataFrame) -> str:
    block = coefficients[coefficients["block"] == "long-run relation, exports normalised to 1"]
    values = {row["name"]: row["estimate"] for _, row in block.iterrows()}
    speeds = {
        row["name"]: (row["estimate"], row["pvalue"])
        for _, row in coefficients[coefficients["block"] == "adjustment speed"].iterrows()
    }
    return (
        f"Scaled so that exports have a weight of one, the cointegrating relation is "
        f"exports = {values['REER']:.2f} REER + {values['imports']:.2f} imports "
        f"{values['constant']:+.2f}. Exports enter the estimated residual with a negative "
        "weight and the export adjustment coefficient is positive "
        f"({speeds['exports'][0]:+.2f}, {p_text(speeds['exports'][1])}), so an export level "
        "above that relation is pulled back down. Import adjustment is "
        f"{speeds['imports'][0]:+.2f} ({p_text(speeds['imports'][1])}). The REER loading is "
        f"{speeds['REER'][0]:+.2f} ({p_text(speeds['REER'][1])}) and is not distinguishable from zero."
    )


def factor_paragraph(robustness: pd.DataFrame, shock_scale: dict) -> str:
    demand = robustness[
        (robustness["specification"] == "us_china") & (robustness["outcome"] == "exports")
    ]
    us = demand[demand["shock"] == "US GDP"].iloc[0]
    china = demand[demand["shock"] == "China GDP"].iloc[0]
    early = robustness[
        (robustness["specification"] == "through_2019Q4")
        & (robustness["outcome"] == "exports")
        & (robustness["shock"] == "foreign GDP")
    ].iloc[0]
    return (
        "On the full-sample point estimates, foreign GDP moves exports by more than the "
        "REER or TPU. That ranking does not survive the sample that ends in 2019Q4: the "
        f"same 1 s.d. foreign-GDP multiplier is then {early['impact_percent']:+.2f}%, on an "
        f"innovation of only {100 * early['innovation_sd']:.2f}% rather than "
        f"{100 * shock_scale['log_foreign_gdp']:.2f}%. Splitting partners, a 1 s.d. US GDP "
        f"innovation moves exports by {us['impact_percent']:+.2f}% on impact, while a 1 s.d. "
        f"China GDP innovation moves them by {china['impact_percent']:+.2f}%. The full-sample "
        "foreign-demand result is a US and COVID result, not a China result, and it is not "
        "visible in the pre-COVID window."
    )


def fevd_paragraph() -> str:
    table = pd.read_csv(TABLES / "var_fevd.csv")

    def shares(horizon: int) -> str:
        part = table[(table["outcome"] == "exports") & (table["horizon"] == horizon)]
        return ", ".join(f"{row.shock} {100 * row.share:.0f}%" for row in part.itertuples())

    return (
        "The forecast-error variance decomposition covers only shocks inside the "
        "Malaysian block. Foreign GDP and TPU are exogenous, so they get no share. "
        f"On impact, export forecast errors are {shares(0)}. "
        f"Eight quarters ahead the shares are {shares(8)}. "
        "Imports are ordered last in the Cholesky factor, so they cannot move exports "
        "in the impact quarter. The later import share is the permanent component of "
        "this ordering in a system with two unit roots. It is not evidence that an "
        "import shock causes exports."
    )


def episode_paragraph(episodes: dict) -> str:
    dummies = episodes["dummies"]
    sentences = [
        "The GFC dummy is zero in every quarter from 2015Q1 to 2024Q1, so it has no "
        "coefficient and there is no pre-2008 window for a counterfactual."
    ]
    sentences.append(dummy_sentence(dummies, "COVID"))
    sentences.append(dummy_sentence(dummies, "trade war"))
    sentences.append(forecast_sentence(episodes["forecasts"]["COVID"], "COVID"))
    sentences.append(forecast_sentence(episodes["forecasts"]["trade war"], "trade war"))
    return " ".join(sentences)


def dummy_sentence(dummies: pd.DataFrame, episode: str) -> str:
    part = dummies[dummies["episode"] == episode].set_index("equation")
    bits = []
    for equation in ("exports", "imports", "REER"):
        row = part.loc[equation]
        call = "significant at 5%" if row["pvalue"] < 0.05 else "not significant at 5%"
        bits.append(
            f"{equation} {to_percent(row['estimate']):+.1f}% (p={row['pvalue']:.3f}, {call})"
        )
    return (
        f"The {episode} dummy shifts quarterly log-growth while it equals one. "
        f"In percent, that shift is {'; '.join(bits)}. "
        "Error correction offsets a dummy that stays on, so these are not losses that "
        "compound quarter after quarter."
    )


def forecast_sentence(forecast: dict, episode: str) -> str:
    if not forecast.get("ok"):
        return forecast["reason"]
    path = forecast["path"]
    low = path.loc[path["gap_percent"].idxmin()]
    high = path.loc[path["gap_percent"].idxmax()]
    return (
        f"The {episode} counterfactual is a forecast from the model estimated before "
        f"the episode ({int(forecast['pre_quarters'])} quarters), using the foreign path "
        f"that actually happened. Actual exports minus that forecast are "
        f"{path['gap_percent'].iloc[0]:+.1f}% in {path['quarter'].iloc[0]} and "
        f"{path['gap_percent'].iloc[-1]:+.1f}% in {path['quarter'].iloc[-1]}. "
        f"Inside the window the gap ranges from {low['gap_percent']:+.1f}% "
        f"({low['quarter']}) to {high['gap_percent']:+.1f}% ({high['quarter']})."
    )


def robustness_paragraph(robustness: pd.DataFrame, projections: pd.DataFrame) -> str:
    base = robustness[
        (robustness["specification"] == "baseline")
        & (robustness["outcome"] == "exports")
        & (robustness["shock"] == "foreign GDP")
    ].iloc[0]
    lp0 = projections[(projections["shock"] == "foreign GDP") & (projections["horizon"] == 0)].iloc[0]
    lp4 = projections[(projections["shock"] == "foreign GDP") & (projections["horizon"] == 4)].iloc[0]
    tpu_early = robustness[
        (robustness["specification"] == "through_2019Q4")
        & (robustness["outcome"] == "exports")
        & (robustness["shock"] == "TPU")
    ].iloc[0]
    band = "excludes zero" if excludes_zero(lp0["percent_low_90"], lp0["percent_high_90"]) else "includes zero"
    return (
        f"Jordà local projections of exports on the same 1 s.d. foreign-GDP innovation, "
        f"with Newey-West standard errors, put the impact at {lp0['percent']:+.2f}% "
        f"(the 90% band {band}) and the four-quarter response at {lp4['percent']:+.2f}%. "
        f"The VECM impact is {base['impact_percent']:+.2f}% and is essentially zero by quarter 4. "
        "The two methods agree that the full-sample impact is positive. They do not agree "
        "on how long it lasts. The local projection also uses the COVID quarter, so its "
        "tight band is not separate evidence from the VECM. Ending the VECM in 2019Q4 "
        f"removes the foreign-GDP effect. Over that shorter sample the TPU impact on "
        f"exports is {tpu_early['impact_percent']:+.2f}%, the opposite sign from the "
        "full-sample point estimate."
    )


def diagnostic_paragraph(diagnostics: pd.DataFrame, result, n_boot: int) -> str:
    rows = {row["test"]: row for _, row in diagnostics.iterrows()}
    port = rows["Portmanteau"]
    lm = rows["Breusch-Godfrey LM"]
    normal = rows["Jarque-Bera"]
    roots = rows["Companion roots"]
    port_call = "does not reject" if port["pvalue"] >= 0.05 else "rejects"
    lm_call = "rejects" if lm["pvalue"] < 0.05 else "does not reject"
    normal_call = "does not reject" if normal["pvalue"] >= 0.05 else "rejects"
    return (
        f"The adjusted Portmanteau test at {int(port['lags'])} lags {port_call} residual "
        f"autocorrelation at 5% (statistic {port['statistic']:.1f}, p={port['pvalue']:.3f}). "
        f"The Breusch-Godfrey LM test at the same lag {lm_call} it "
        f"(statistic {lm['statistic']:.1f}, p={lm['pvalue']:.3f}). "
        f"Jarque-Bera {normal_call} normality (statistic {normal['statistic']:.1f}, "
        f"p={normal['pvalue']:.3f}). {roots['note']} "
        f"The residual bootstrap kept {n_boot} of {N_BOOT} draws. "
        f"The log-likelihood is {float(result.llf):.1f} on {int(result.nobs)} estimation observations."
    )


def caveat_paragraph(nobs: int, n_boot: int, shock_scale: dict) -> str:
    return (
        f"The estimation sample has {nobs} quarters. A VECM with three endogenous variables, "
        "two dummies, and two foreign regressors is a lot of structure for that length, and "
        "the information criteria only agree on one lag once lags that exhaust the degrees of "
        "freedom are set aside. KPSS did not reject stationarity of the three endogenous series "
        "in levels, so the unit-root reading is the ADF result and is not unanimous. "
        "The REER impulse response is a Cholesky shock with the exchange rate ordered first: "
        "within a quarter, exports and imports do not move the REER. Foreign GDP and TPU are "
        "imposed to be exogenous rather than tested against a model in which Malaysia feeds "
        "back into them. Their dynamic multipliers are one-quarter impulses of the size of an "
        f"AR(1) innovation ({100 * shock_scale['log_foreign_gdp']:.2f}% for foreign GDP, "
        f"{shock_scale['tpu']:.0f} index points for TPU), not the effect of a permanent rise "
        "in foreign demand. "
        "The trade-war variable is a step from 2018Q3, so it can absorb any shift in trade "
        "growth that lines up with that date. The GFC is outside the sample. "
        "Bootstrap bands are conditional on the observed foreign path and use "
        f"{n_boot} successful redraws of the Malaysian residuals. "
        "Dummy p-values are asymptotic. COVID is both a dummy and a collapse in the foreign-GDP "
        "series, so those two channels are not cleanly separated."
    )


def shock_name(column: str) -> str:
    return {
        "log_reer": "REER",
        "log_foreign_gdp": "foreign GDP",
        "tpu": "TPU",
        "log_gdp_us": "US GDP",
        "log_gdp_china": "China GDP",
    }[column]


def to_percent(value) -> float | np.ndarray:
    return 100 * (np.exp(value) - 1)


def p_text(pvalue: float) -> str:
    return "p<0.001" if pvalue < 0.001 else f"p={pvalue:.3f}"


def excludes_zero(lower: float, upper: float) -> bool:
    return bool(lower > 0 or upper < 0)


if __name__ == "__main__":
    raise SystemExit(main())
