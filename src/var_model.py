"""Estimate the Malaysian trade VECM and the robustness checks.

The specification is the one selected in src/var_data.py: one lag in levels
(no lagged differences), one cointegrating relation, and a constant inside
that relation. Foreign GDP and trade-policy uncertainty are exogenous.
Malaysia is treated as a small open economy.

    python src/var_model.py

The GFC dummy is a regressor only when it varies inside the estimation sample.
2026Q3 is already absent from the file.
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

try:
    from ardl_demand import estimate as estimate_ardl
except ImportError:
    from src.ardl_demand import estimate as estimate_ardl

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
BASE_EXOG = [
    "log_foreign_gdp",
    "tpu",
    "log_commodity",
    "dummy_gfc",
    "dummy_covid",
    "dummy_trade_war",
]
EXOG_LABEL = {
    "log_foreign_gdp": "foreign GDP",
    "log_gdp_us": "US GDP",
    "log_gdp_china": "China GDP",
    "tpu": "TPU",
    "log_commodity": "commodity prices",
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


SPEC = {"k_ar_diff": 0, "coint_rank": 1, "deterministic": "ci"}


def main() -> int:
    global SPEC, PORTMANTEAU_LAGS, LM_LAGS
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    spec_path = TABLES / "var_specification.csv"
    if not spec_path.exists():
        raise RuntimeError("Missing outputs/tables/var_specification.csv. Run python src/var_data.py first.")
    spec = pd.read_csv(spec_path).iloc[0]
    SPEC["k_ar_diff"] = int(spec["k_ar_diff"])
    SPEC["coint_rank"] = int(spec["coint_rank"])
    SPEC["deterministic"] = str(spec["deterministic"]) if "deterministic" in spec else "ci"
    PORTMANTEAU_LAGS = int(spec["portmanteau_lags"])
    LM_LAGS = int(spec["bg_lags"])

    frame = load_frame()
    if "log_commodity" not in frame.columns:
        raise RuntimeError("log_commodity is missing. Run python src/var_data.py after adding commodity prices.")
    estimate_ardl(frame)
    short_exog = varying(frame, [column for column in BASE_EXOG if column in frame.columns])
    coint_cols = ["log_foreign_gdp"]
    restricted_exog = [column for column in short_exog if column not in coint_cols]

    short = fit_vecm(frame, short_exog)
    check_reconstruction(frame, short_exog, short)
    restricted = fit_vecm(frame, restricted_exog, coint_cols)
    check_reconstruction(frame, restricted_exog, restricted, coint_cols)

    shock_scale = innovation_scales(frame, ["log_foreign_gdp", "tpu"])
    short_paths = response_paths(short, short_exog, shock_scale)
    point = response_paths(restricted, restricted_exog, shock_scale, coint_cols)
    bands, n_boot_ok = bootstrap_bands(frame, restricted_exog, shock_scale, coint_cols)
    save_responses(point, bands)
    save_path_comparison(short_paths, point)

    diagnostics = save_diagnostics(frame, restricted_exog, restricted, coint_cols)
    save_diagnostics(frame, short_exog, short, path=TABLES / "var_diagnostics_short_run.csv")
    coefficients = save_coefficients(restricted, restricted_exog)
    short_coefficients = save_coefficients(short, short_exog)
    short_coefficients.to_csv(TABLES / "var_coefficients_short_run.csv", index=False)
    coefficients.to_csv(TABLES / "var_coefficients.csv", index=False)
    save_fevd(restricted)
    save_fevd(short, TABLES / "var_fevd_short_run.csv")
    episodes = save_episodes(frame, restricted, restricted_exog, coefficients, coint_cols)
    projections = save_local_projections(frame, shock_scale)
    robustness, long_run = save_robustness(frame, point)
    save_without_commodity(frame, coint_cols, shock_scale)

    plot_responses(point, bands)
    plot_fevd(result=None)
    plot_counterfactuals(episodes)
    plot_local_projections(projections, point)
    plot_robustness(robustness)

    write_summary(
        frame,
        restricted_exog,
        restricted,
        point,
        bands,
        n_boot_ok,
        diagnostics,
        short_coefficients,
        episodes,
        projections,
        robustness,
        shock_scale,
        long_run,
        short_paths,
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


def fit_vecm(frame: pd.DataFrame, exog_cols: list[str], coint_cols: list[str] | None = None):
    """Fit the VECM. Columns in coint_cols are restricted to the cointegrating relation."""
    k_ar_diff = int(SPEC["k_ar_diff"])
    coint_rank = int(SPEC["coint_rank"])
    coint_cols = coint_cols or []
    exog = frame[exog_cols].to_numpy(dtype=float) if exog_cols else None
    exog_coint = frame[coint_cols].to_numpy(dtype=float) if coint_cols else None
    model = VECM(
        frame[ENDOG],
        exog=exog,
        exog_coint=exog_coint,
        k_ar_diff=k_ar_diff,
        coint_rank=coint_rank,
        deterministic=str(SPEC["deterministic"]),
    )
    return model.fit()


def check_reconstruction(
    frame: pd.DataFrame, exog_cols: list[str], result, coint_cols: list[str] | None = None
) -> None:
    """The estimated recursion plus the original residuals should recover the data."""
    coint_cols = coint_cols or []
    exog = frame[exog_cols].to_numpy(dtype=float) if exog_cols else np.zeros((len(frame), 0))
    coint = frame[coint_cols].to_numpy(dtype=float) if coint_cols else None
    simulated = simulate(
        frame[ENDOG].to_numpy(dtype=float),
        exog,
        result,
        np.asarray(result.resid, dtype=float),
        coint,
    )
    gap = np.max(np.abs(simulated - frame[ENDOG].to_numpy(dtype=float)))
    if not np.isfinite(gap) or gap > 1e-6:
        raise RuntimeError(f"VECM reconstruction error is {gap}. The exog timing is wrong.")


def simulate(
    levels: np.ndarray,
    exog: np.ndarray,
    result,
    residuals: np.ndarray,
    exog_coint: np.ndarray | None = None,
) -> np.ndarray:
    """Rebuild levels from y0, the foreign path, and a residual sequence."""
    path = np.array(levels, dtype=float, copy=True)
    beta = np.asarray(result.beta, dtype=float)
    alpha = np.asarray(result.alpha, dtype=float)
    constant = np.asarray(result.const_coint, dtype=float).reshape(-1)
    loadings = np.asarray(result.exog_coefs, dtype=float)
    phi = None if result.exog_coint_coefs is None else np.asarray(result.exog_coint_coefs, dtype=float)
    k_diff = int(result.k_ar - 1)
    p = int(result.k_ar)
    gamma = np.asarray(result.gamma, dtype=float) if k_diff else None
    for step, residual in enumerate(residuals):
        t = step + p
        if result.coint_rank:
            equilibrium = beta.T @ path[t - 1] + constant
            if phi is not None and phi.size and exog_coint is not None:
                equilibrium = equilibrium + phi.T @ exog_coint[t - 1]
            growth = alpha @ equilibrium
        else:
            growth = np.zeros(path.shape[1])
        if k_diff:
            deltas = [path[t - lag] - path[t - lag - 1] for lag in range(1, k_diff + 1)]
            growth = growth + gamma @ np.concatenate(deltas)
        if loadings.size and exog.size:
            growth = growth + loadings @ exog[t]
        growth = growth + residual
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


def response_paths(
    result,
    exog_cols: list[str],
    shock_scale: dict[str, float],
    coint_cols: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Log-level responses from horizon 0 through HORIZON.

    REER is an orthogonalised impulse response. A variable outside the
    cointegrating relation moves trade in the impact quarter. A variable
    restricted to the cointegrating relation is lagged, so a one-quarter
    innovation first moves trade one quarter later.
    """
    horizon = HORIZON + 1
    coint_cols = coint_cols or []
    # ma_rep returns maxn + 1 matrices, so maxn=HORIZON covers horizons 0..12.
    ma = np.asarray(result.orth_ma_rep(maxn=HORIZON), dtype=float)[:horizon]
    paths = {}
    reer_at = ENDOG.index("log_reer")
    for name in ENDOG:
        paths[f"{name}__log_reer"] = ma[:, ENDOG.index(name), reer_at]
    loadings = np.asarray(result.exog_coefs, dtype=float)
    for column, scale in shock_scale.items():
        if column not in exog_cols:
            continue
        shock = np.zeros(len(exog_cols))
        shock[exog_cols.index(column)] = scale
        level = loadings @ shock
        companion_paths = _exog_paths(result, level, horizon)
        for name in ENDOG:
            series = np.array([state[ENDOG.index(name)] for state in companion_paths])
            paths[f"{name}__{column}"] = series
    phi = None if result.exog_coint_coefs is None else np.asarray(result.exog_coint_coefs, dtype=float)
    alpha = np.asarray(result.alpha, dtype=float)
    if phi is not None and phi.size:
        for column, scale in shock_scale.items():
            if column not in coint_cols:
                continue
            slot = coint_cols.index(column)
            growth = (alpha @ phi[slot : slot + 1].T).reshape(-1) * scale
            tail = _exog_paths(result, growth, horizon - 1)
            states = [np.zeros(len(ENDOG)), *tail]
            for name in ENDOG:
                paths[f"{name}__{column}"] = np.array([state[ENDOG.index(name)] for state in states])
    return paths


def _exog_paths(result, impact: np.ndarray, horizon: int) -> list[np.ndarray]:
    """Levels path of a one-quarter exogenous impulse, using the full VAR lag polynomial."""
    coefs = [np.asarray(matrix, dtype=float) for matrix in result.var_rep]
    current = np.asarray(impact, dtype=float).copy()
    stored = [current]
    lags = [np.zeros(current.shape[0]) for _ in coefs]
    for _ in range(1, horizon):
        lags = [current, *lags[:-1]]
        current = sum((coef @ lag for coef, lag in zip(coefs, lags)), np.zeros(impact.shape[0]))
        stored.append(current)
    return stored


def bootstrap_bands(
    frame: pd.DataFrame,
    exog_cols: list[str],
    shock_scale: dict[str, float],
    coint_cols: list[str] | None = None,
) -> tuple[dict[str, np.ndarray], int]:
    coint_cols = coint_cols or []
    levels = frame[ENDOG].to_numpy(dtype=float)
    exog = frame[exog_cols].to_numpy(dtype=float) if exog_cols else np.zeros((len(frame), 0))
    coint = frame[coint_cols].to_numpy(dtype=float) if coint_cols else None
    base = fit_vecm(frame, exog_cols, coint_cols)
    residuals = np.asarray(base.resid, dtype=float)
    keys = [f"{name}__{shock}" for name in ENDOG for shock in ("log_reer", *shock_scale)]
    draws = {key: [] for key in keys}
    rng = np.random.default_rng(BOOT_SEED)
    kept = 0
    for _ in range(N_BOOT):
        choice = rng.integers(0, len(residuals), size=len(residuals))
        simulated = simulate(levels, exog, base, residuals[choice], coint)
        if not np.all(np.isfinite(simulated)):
            continue
        try:
            boot_frame = pd.DataFrame(simulated, columns=ENDOG)
            for column in (*exog_cols, *coint_cols):
                boot_frame[column] = frame[column].to_numpy()
            boot_frame["quarter"] = frame["quarter"].to_numpy()
            boot = fit_vecm(boot_frame, exog_cols, coint_cols)
            paths = response_paths(boot, exog_cols, shock_scale, coint_cols)
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


def save_path_comparison(short_paths: dict[str, np.ndarray], restricted_paths: dict[str, np.ndarray]) -> None:
    rows = []
    for specification, paths in (("short_run", short_paths), ("restricted", restricted_paths)):
        for key, path in paths.items():
            outcome, shock = key.split("__")
            if outcome not in ("log_exports", "log_imports"):
                continue
            rows.append(
                {
                    "specification": specification,
                    "outcome": ENDOG_LABEL[outcome],
                    "shock": shock_name(shock),
                    "impact_percent": to_percent(path[0]),
                    "horizon_1_percent": to_percent(path[1]),
                    "horizon_4_percent": to_percent(path[4]),
                    "horizon_8_percent": to_percent(path[8]),
                    "horizon_12_percent": to_percent(path[12]),
                }
            )
    pd.DataFrame(rows).to_csv(TABLES / "var_irf_comparison.csv", index=False)


def save_diagnostics(
    frame: pd.DataFrame,
    exog_cols: list[str],
    result,
    coint_cols: list[str] | None = None,
    path: Path | None = None,
) -> pd.DataFrame:
    rows = []
    white = result.test_whiteness(nlags=PORTMANTEAU_LAGS, signif=0.05, adjusted=True)
    rows.append(test_row("Portmanteau", PORTMANTEAU_LAGS, white))
    lm_stat, lm_df, lm_p = breusch_godfrey(frame, exog_cols, result, LM_LAGS, coint_cols)
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
                f"{unit_roots} roots lie on the unit circle, against {int(result.neqs - result.coint_rank)} expected "
                f"for rank {int(result.coint_rank)} among {int(result.neqs)} variables. "
                f"The largest other modulus is {float(other.max()) if len(other) else float('nan'):.3f}."
            ),
        }
    )
    table = pd.DataFrame(rows)
    table.to_csv(path or (TABLES / "var_diagnostics.csv"), index=False)
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


def breusch_godfrey(
    frame: pd.DataFrame,
    exog_cols: list[str],
    result,
    lags: int,
    coint_cols: list[str] | None = None,
) -> tuple[float, int, float]:
    """System LM test: residuals on lagged residuals and the VECM regressors."""
    coint_cols = coint_cols or []
    resid = np.asarray(result.resid, dtype=float)
    n_est, equations = resid.shape
    p = int(result.k_ar)
    k_diff = int(result.k_ar - 1)
    levels = frame[ENDOG].to_numpy(dtype=float)
    use = np.arange(lags, n_est)
    t = p + use
    outcome = resid[use]
    pieces = [np.ones((len(use), 1)), levels[t - 1]]
    if exog_cols:
        pieces.append(frame[exog_cols].to_numpy(dtype=float)[t])
    if coint_cols:
        pieces.append(frame[coint_cols].to_numpy(dtype=float)[t - 1])
    for lag in range(1, k_diff + 1):
        pieces.append(levels[t - lag] - levels[t - lag - 1])
    for lag in range(1, lags + 1):
        pieces.append(resid[use - lag])
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
    coefs = [np.asarray(matrix, dtype=float) for matrix in result.var_rep]
    width = coefs[0].shape[0]
    order = len(coefs)
    companion = np.zeros((width * order, width * order))
    companion[:width, :] = np.concatenate(coefs, axis=1)
    if order > 1:
        companion[width:, : width * (order - 1)] = np.eye(width * (order - 1))
    values = np.linalg.eigvals(companion)
    values = values[np.argsort(-np.abs(values))]
    return pd.DataFrame(
        {
            "root": [f"{value.real:.6f}{value.imag:+.6f}i" for value in values],
            "real": values.real,
            "imag": values.imag,
            "modulus": np.abs(values),
        }
    )


def free_coint_covariance(result) -> np.ndarray:
    """Covariance of the cointegrating coefficients that are not normalised to one."""
    from statsmodels.tsa.vector_ar.vecm import _r_matrices

    rank = int(result.coint_rank)
    _, r1 = _r_matrices(result._delta_y_1_T, result._y_lag1, result._delta_x)
    r12 = r1[rank:]
    mat1 = np.kron(np.linalg.inv(r12 @ r12.T).T, np.eye(rank))
    det = int(result.det_coef_coint.shape[0])
    alpha = np.asarray(result.alpha, dtype=float)
    sigma = np.asarray(result.sigma_u, dtype=float)
    inner = np.linalg.inv(alpha.T @ np.linalg.inv(sigma) @ alpha)
    mat2 = np.kron(np.eye(result.neqs - rank + det), inner)
    return mat1 @ mat2


def normalised_relation(result, coint_cols: list[str], specification: str) -> pd.DataFrame:
    """Cointegrating vector renormalised so exports equal 1, with delta-method standard errors.

    The import block uses the same vector renormalised on imports, so the foreign-demand
    and REER elasticities can be read for both trade flows.
    """
    if int(result.coint_rank) != 1:
        raise RuntimeError("The long-run table is written for one cointegrating relation.")
    beta = np.asarray(result.beta, dtype=float)[:, 0]
    det = np.asarray(result.det_coef_coint, dtype=float).reshape(-1)
    det_names = []
    if "ci" in str(result.deterministic):
        det_names.append("constant")
    det_names.extend(EXOG_LABEL.get(column, column) for column in coint_cols)
    if len(det_names) != len(det):
        raise RuntimeError(
            f"Cointegrating deterministic terms {det.shape} do not match {det_names}."
        )
    names = [ENDOG_LABEL[column] for column in ENDOG] + det_names
    raw = np.concatenate([beta, det])
    covariance = free_coint_covariance(result)
    free_se = np.sqrt(np.diag(covariance))
    published = np.asarray(result.stderr_coint, dtype=float).reshape(-1)[1:]
    if free_se.shape != published.shape or np.max(np.abs(free_se - published)) > 1e-6:
        raise RuntimeError("The cointegrating covariance does not match statsmodels standard errors.")
    rows = []
    for normalised_on, scale_name in (("exports", "exports"), ("imports", "imports")):
        scale_at = names.index(scale_name)
        scale = float(raw[scale_at])
        if abs(scale) < 1e-8:
            raise RuntimeError(f"Cannot normalise the cointegrating vector on {scale_name}.")
        for index, name in enumerate(names):
            weight = float(raw[index] / scale)
            if index == scale_at:
                standard_error = 0.0
            else:
                gradient = np.zeros(covariance.shape[0])
                # raw[0] is the Johansen unit weight and is not in the free covariance.
                scale_free = scale_at - 1
                gradient[scale_free] += -raw[index] / scale**2
                if index != 0:
                    gradient[index - 1] += 1.0 / scale
                standard_error = float(np.sqrt(gradient @ covariance @ gradient))
            elasticity = -weight if name != scale_name else 1.0
            if name == "constant" or name == scale_name:
                elasticity = np.nan if name == "constant" else 1.0
            t_stat = 0.0 if standard_error == 0 else weight / standard_error
            pvalue = 1.0 if standard_error == 0 else float(2 * stats.norm.sf(abs(t_stat)))
            rows.append(
                {
                    "specification": specification,
                    "normalised_on": normalised_on,
                    "name": name,
                    "weight": weight,
                    "std_error": standard_error,
                    "pvalue": pvalue,
                    "elasticity": elasticity,
                }
            )
        alpha = np.asarray(result.alpha, dtype=float).reshape(-1) * scale
        alpha_se = np.abs(scale) * np.asarray(result.stderr_alpha, dtype=float).reshape(-1)
        alpha_p = np.asarray(result.pvalues_alpha, dtype=float).reshape(-1)
        for name, value, se, pvalue in zip(ENDOG_LABEL.values(), alpha, alpha_se, alpha_p):
            rows.append(
                {
                    "specification": specification,
                    "normalised_on": normalised_on,
                    "name": f"alpha {name}",
                    "weight": float(value),
                    "std_error": float(se),
                    "pvalue": float(pvalue),
                    "elasticity": np.nan,
                }
            )
    return pd.DataFrame(rows)


def save_coefficients(result, exog_cols: list[str]) -> pd.DataFrame:
    rows = []
    beta = np.asarray(result.beta, dtype=float)
    if beta.ndim == 1:
        beta = beta.reshape(-1, 1)
    constants = np.asarray(result.const_coint, dtype=float).reshape(-1)
    export_slot = ENDOG.index("log_exports")
    for relation in range(beta.shape[1]):
        column = beta[:, relation]
        constant = float(constants[relation])
        scale = column[export_slot]
        tag = "" if beta.shape[1] == 1 else f" {relation + 1}"
        for name, value in zip(ENDOG, column):
            rows.append(coefficient_row(f"cointegrating vector{tag}", ENDOG_LABEL[name], value, np.nan, np.nan))
        rows.append(coefficient_row(f"cointegrating vector{tag}", "constant", constant, np.nan, np.nan))
        if relation == 0 and abs(scale) > 1e-8:
            for name, value in zip(ENDOG, column):
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
    alpha = np.asarray(result.alpha, dtype=float)
    alpha_se = np.asarray(result.stderr_alpha, dtype=float)
    alpha_p = np.asarray(result.pvalues_alpha, dtype=float)
    if alpha.ndim == 1:
        alpha = alpha.reshape(-1, 1)
        alpha_se = alpha_se.reshape(-1, 1)
        alpha_p = alpha_p.reshape(-1, 1)
    for relation in range(alpha.shape[1]):
        tag = "" if alpha.shape[1] == 1 else f" {relation + 1}"
        for name, value, se, pvalue in zip(ENDOG, alpha[:, relation], alpha_se[:, relation], alpha_p[:, relation]):
            rows.append(coefficient_row(f"adjustment speed{tag}", ENDOG_LABEL[name], value, se, pvalue))
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


def save_fevd(result, path: Path | None = None) -> pd.DataFrame:
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
    table.to_csv(path or (TABLES / "var_fevd.csv"), index=False)
    return table


def save_without_commodity(frame: pd.DataFrame, coint_cols: list[str], shock_scale: dict[str, float]) -> None:
    """Same VECM with commodity prices left out, for the comparison."""
    short_cols = varying(
        frame,
        [column for column in BASE_EXOG if column != "log_commodity" and column in frame.columns],
        "the VECM without commodity prices",
    )
    restricted_cols = [column for column in short_cols if column not in coint_cols]
    short = fit_vecm(frame, short_cols)
    restricted = fit_vecm(frame, restricted_cols, coint_cols)
    relation = normalised_relation(restricted, coint_cols, "without_commodity")
    relation.to_csv(TABLES / "var_long_run_no_commodity.csv", index=False)
    short_coef = save_coefficients(short, short_cols)
    restricted_paths = response_paths(restricted, restricted_cols, shock_scale, coint_cols)
    short_paths = response_paths(short, short_cols, shock_scale)
    rows = []
    for name in ("exports on foreign GDP", "imports on foreign GDP"):
        matched = short_coef.loc[short_coef["name"] == name]
        if matched.empty:
            continue
        row = matched.iloc[0]
        rows.append(
            {
                "specification": "short_run",
                "name": name,
                "impact_percent": np.nan,
                "horizon_1_percent": np.nan,
                "horizon_4_percent": np.nan,
                "estimate": float(row["estimate"]),
                "pvalue": float(row["pvalue"]),
            }
        )
    for specification, paths in (("restricted", restricted_paths), ("short_run", short_paths)):
        for shock in ("log_foreign_gdp", "log_reer", "tpu"):
            path = paths[f"log_exports__{shock}"]
            label = EXOG_LABEL.get(shock, ENDOG_LABEL.get(shock, shock))
            rows.append(
                {
                    "specification": specification,
                    "name": f"exports after {label}",
                    "impact_percent": to_percent(path[0]),
                    "horizon_1_percent": to_percent(path[1]),
                    "horizon_4_percent": to_percent(path[4]),
                    "estimate": np.nan,
                    "pvalue": np.nan,
                }
            )
    pd.DataFrame(rows).to_csv(TABLES / "var_no_commodity.csv", index=False)


def save_episodes(
    frame: pd.DataFrame,
    result,
    exog_cols: list[str],
    coefficients: pd.DataFrame,
    coint_cols: list[str] | None = None,
) -> dict:
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
                        "note": (
                            "Not estimated. The dummy does not vary in "
                            f"{frame['quarter'].iloc[0]}-{frame['quarter'].iloc[-1]}."
                        ),
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
    controls = [column for column in (
        "log_foreign_gdp", "tpu", "log_commodity", "dummy_gfc", "dummy_covid", "dummy_trade_war",
    ) if column in frame.columns]
    forecasts["COVID"] = episode_forecast(
        frame,
        start="2020Q1",
        end="2022Q4",
        exog_cols=[column for column in controls if column != "dummy_covid"],
        coint_cols=coint_cols,
    )
    forecasts["COVID, no commodity"] = episode_forecast(
        frame,
        start="2020Q1",
        end="2022Q4",
        exog_cols=[column for column in controls if column not in ("dummy_covid", "log_commodity")],
        coint_cols=coint_cols,
    )
    forecasts["trade war"] = episode_forecast(
        frame,
        start="2018Q3",
        end="2019Q4",
        exog_cols=[column for column in controls if column != "dummy_trade_war"],
        coint_cols=coint_cols,
    )
    if frame["quarter"].iloc[0] <= pd.Period("2008Q3", freq="Q-DEC"):
        forecasts["GFC"] = episode_forecast(
            frame,
            start="2008Q4",
            end="2009Q2",
            exog_cols=[column for column in controls if column != "dummy_gfc"],
            coint_cols=coint_cols,
        )
    else:
        forecasts["GFC"] = {
            "ok": False,
            "reason": (
                f"The sample starts in {frame['quarter'].iloc[0]}, so there is no "
                "pre-2008 window to estimate from."
            ),
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


def episode_forecast(
    frame: pd.DataFrame,
    start: str,
    end: str,
    exog_cols: list[str],
    coint_cols: list[str] | None = None,
) -> dict:
    coint_cols = list(coint_cols or [])
    start_q = pd.Period(start, freq="Q-DEC")
    end_q = pd.Period(end, freq="Q-DEC")
    pre = frame[frame["quarter"] < start_q].copy()
    future = frame[(frame["quarter"] >= start_q) & (frame["quarter"] <= end_q)].copy()
    # A restricted variable is not also a short-run regressor in the forecast.
    short_cols = [column for column in exog_cols if column not in coint_cols]
    usable = varying(pre, short_cols, f"the pre-{start} forecast")
    coint_usable = varying(pre, coint_cols, f"the pre-{start} cointegrating relation")
    if len(pre) < 12 or future.empty:
        return {
            "ok": False,
            "reason": f"Only {len(pre)} quarters are available before {start}.",
        }
    try:
        fitted = fit_vecm(pre, usable, coint_usable)
        forecast = fitted.predict(
            steps=len(future),
            exog_fc=future[usable].to_numpy(dtype=float) if usable else None,
            exog_coint_fc=future[coint_usable].to_numpy(dtype=float) if coint_usable else None,
        )
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
    if "log_commodity" in indexed.columns:
        controls["commodity"] = indexed["log_commodity"].diff()
    other = "tpu" if shock_name_col == "log_foreign_gdp" else "log_foreign_gdp"
    controls["other_shock"] = innovations[other]
    controls["gfc"] = indexed["dummy_gfc"]
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


def usd_trade_sample(frame: pd.DataFrame) -> pd.DataFrame:
    """Continuous prefix of the USD CPI-deflated trade series. The gap is not filled."""
    present = set(frame.loc[frame["log_exports_usd"].notna(), "quarter"])
    kept = []
    for quarter in frame["quarter"]:
        if quarter not in present:
            break
        kept.append(quarter)
    sample = frame[frame["quarter"].isin(kept)].copy()
    sample["log_exports"] = sample["log_exports_usd"]
    sample["log_imports"] = sample["log_imports_usd"]
    return sample


def save_robustness(frame: pd.DataFrame, restricted_paths: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = {
        "restricted": (
            frame,
            ["tpu", "log_commodity", "dummy_gfc", "dummy_covid", "dummy_trade_war"],
            ["log_foreign_gdp"],
            restricted_paths,
        ),
        "short_run": (
            frame,
            ["log_foreign_gdp", "tpu", "log_commodity", "dummy_gfc", "dummy_covid", "dummy_trade_war"],
            [],
            None,
        ),
        "us_china": (
            frame,
            ["tpu", "log_commodity", "dummy_gfc", "dummy_covid", "dummy_trade_war"],
            ["log_gdp_us", "log_gdp_china"],
            None,
        ),
        "through_2019Q4": (
            frame[frame["quarter"] <= pd.Period("2019Q4", freq="Q-DEC")].copy(),
            ["tpu", "log_commodity", "dummy_gfc", "dummy_covid", "dummy_trade_war"],
            ["log_foreign_gdp"],
            None,
        ),
        "usd_cpi": (
            usd_trade_sample(frame),
            ["tpu", "log_commodity", "dummy_gfc", "dummy_covid", "dummy_trade_war"],
            ["log_foreign_gdp"],
            None,
        ),
    }
    rows = []
    relations = []
    paths = {}
    shock_names = ("log_foreign_gdp", "tpu", "log_gdp_us", "log_gdp_china")
    for name, (sample, exog_cols, coint_cols, ready) in specs.items():
        exog_cols = varying(sample, exog_cols, name)
        coint_cols = varying(sample, coint_cols, name)
        scales = innovation_scales(sample, [c for c in (*exog_cols, *coint_cols) if c in shock_names])
        fitted = fit_vecm(sample, exog_cols, coint_cols)
        fitted_paths = ready if ready is not None else response_paths(fitted, exog_cols, scales, coint_cols)
        paths[name] = fitted_paths
        if coint_cols:
            relations.append(normalised_relation(fitted, coint_cols, name))
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
                        "horizon_1_percent": to_percent(path[1]),
                        "horizon_4_percent": to_percent(path[4]),
                        "horizon_8_percent": to_percent(path[8]),
                        "horizon_12_percent": to_percent(path[12]),
                        "peak_percent": to_percent(path[int(np.argmax(np.abs(path)))]),
                    }
                )
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "var_robustness.csv", index=False)
    table.attrs["paths"] = paths
    relation = pd.concat(relations, ignore_index=True) if relations else pd.DataFrame()
    relation.to_csv(TABLES / "var_long_run.csv", index=False)
    return table, relation


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
    forecasts = {name: payload for name, payload in episodes["forecasts"].items() if payload.get("ok") and "no commodity" not in name}
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
        ax.set_ylabel("CPI-deflated exports")
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
        ("restricted", "log_foreign_gdp", "Restricted, trade-weighted"),
        ("short_run", "log_foreign_gdp", "Short-run, trade-weighted"),
        ("us_china", "log_gdp_us", "US GDP, in the relation"),
        ("us_china", "log_gdp_china", "China GDP, in the relation"),
        ("through_2019Q4", "log_foreign_gdp", "Restricted, through 2019Q4"),
    ]
    colors = [NAVY, "#5b9bd5", RUST, "#c4a35a", "#6b6b6b"]
    for (spec, column, label), color in zip(demand, colors):
        path = paths[spec][f"log_exports__{column}"]
        axes[0].plot(horizons, to_percent(path), color=color, linewidth=1.7, label=label)
    for spec, label, color in (
        ("restricted", "Restricted", NAVY),
        ("short_run", "Short-run", "#5b9bd5"),
        ("through_2019Q4", "Restricted, through 2019Q4", RUST),
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


def write_summary(
    frame,
    exog_cols,
    result,
    point,
    bands,
    n_boot,
    diagnostics,
    coefficients,
    episodes,
    projections,
    robustness,
    shock_scale,
    long_run,
    short_paths,
) -> None:
    start, end = str(frame["quarter"].iloc[0]), str(frame["quarter"].iloc[-1])
    roots = companion_roots(result)
    transient_roots = roots.loc[np.abs(roots["modulus"] - 1) >= 1e-6, "modulus"]
    transient = float(transient_roots.max()) if len(transient_roots) else float("nan")
    export_gdp = named_coef(coefficients, "exports on foreign GDP")
    import_gdp = named_coef(coefficients, "imports on foreign GDP")
    lines = [
        "# Malaysian trade and external shocks",
        "",
        f"Long-run trade elasticities come from single-equation ARDL models on {start} to {end} "
        f"({len(frame)} quarters). Shock dynamics come from a VECM on the same sample. "
        "The cointegrating relation estimated without commodity prices is dominated by the link "
        "between exports and imports: Malaysia imports intermediates and re-exports them, so "
        "imports absorb foreign demand and the foreign-GDP coefficient in that relation is not "
        "an export-demand elasticity. "
        "Export demand therefore excludes imports. Import demand includes exports. "
        "Log commodity prices (the IMF all-commodity index, PALLFNFINDEXQ) enter both demand "
        "equations and the VECM as an exogenous control, because dividing nominal trade by the "
        "consumer price index turns commodity-price swings into movements that look like volume. "
        "The GFC, COVID, and trade-war dummies are short-run regressors. "
        f"The VECM uses the lag and rank selected for this file: {int(result.coint_rank)} "
        f"cointegrating relation and {int(result.k_ar - 1)} lagged differences. "
        f"{lag_sentence()} "
        "Malaysia is treated as a small open economy that does not move partner GDP.",
        "",
        series_note(),
        "",
        "## Long-run elasticities",
        "",
        ardl_paragraph(),
        "",
        "## Shock dynamics",
        "",
        (
            "Commodity prices stay outside the cointegrating relation. Two placements of foreign "
            "GDP are still estimated. In the short-run specification it is an unrestricted "
            "regressor. In the restricted specification it is weakly exogenous: it enters the "
            "cointegrating relation only, lagged one quarter. TPU, commodity prices, and the "
            "episode dummies stay outside the relation in both. "
            f"With foreign GDP outside the relation, the same-quarter elasticity of exports is "
            f"{export_gdp[0]:.2f} ({p_text(export_gdp[1])}) and of imports is "
            f"{import_gdp[0]:.2f} ({p_text(import_gdp[1])}). "
            f"The largest companion root inside the unit circle has modulus {transient:.2f}."
        ),
        "",
        restricted_paragraph(long_run),
        "",
        without_commodity_paragraph(),
        "",
        response_paragraph(point, bands, shock_scale),
        "",
        comparison_paragraph(short_paths, point),
        "",
        fevd_paragraph(),
        "",
        "## Shock episodes",
        "",
        "The episode estimates below are from the restricted VECM with commodity prices included. "
        + episode_paragraph(episodes),
        "",
        "## Robustness",
        "",
        robustness_paragraph(robustness, projections),
        "",
        "## Diagnostics",
        "",
        diagnostic_paragraph(diagnostics, result, n_boot),
        "",
        "## What each model is for",
        "",
        role_paragraph(),
        "",
        "## Caveats",
        "",
        caveat_paragraph(frame, n_boot, shock_scale),
        "",
    ]
    SUMMARY.write_text("\n".join(lines), encoding="utf-8")


def lag_sentence() -> str:
    spec = pd.read_csv(TABLES / "var_specification.csv").iloc[0]
    if "no lag cleared" not in str(spec["rule"]):
        return (
            f"At that lag the adjusted Portmanteau p-value is {float(spec['portmanteau_p']):.3f} "
            f"and the Breusch-Godfrey p-value is {float(spec['bg_p']):.3f}."
        )
    return (
        "No lag from 1 to 12 clears both residual-correlation tests once commodity prices "
        f"are in the exogenous set. Lag {int(spec['lag'])} is the one with the higher minimum "
        f"p-value (Portmanteau {float(spec['portmanteau_p']):.3f}, "
        f"Breusch-Godfrey {float(spec['bg_p']):.3f})."
    )


def join_and(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def series_note() -> str:
    trade = pd.read_csv(TABLES / "var_trade_deflator_check.csv")
    bits = [
        (
            f"{row.flow} {row.growth_correlation:.3f} "
            f"({row.start} to {row.end}, {int(row.quarters)} growth quarters)"
        )
        for row in trade.itertuples()
    ]
    china = pd.read_csv(TABLES / "var_china_overlap.csv")
    gap = china.loc[china["difference_points"].abs().idxmax()]
    return (
        "The trade series is OpenDOSM monthly exports and imports of goods, in nominal ringgit, "
        "divided by Malaysia's headline CPI (division overall), summed across the three months of "
        "each complete quarter, then seasonally adjusted with STL. DOSM monthly goods trade starts "
        "in January 2000, so it is used on its own and Bank Negara nominal trade was not spliced. "
        "The CPI is a consumer price index, not a trade price or unit-value index, and the series "
        "is goods only, not national-accounts goods and services. "
        "Over the overlap with DOSM constant-2015 seasonally adjusted goods and services, the "
        f"correlation of quarterly log growth is {'; '.join(bits)}. "
        "China GDP keeps the existing growth index through 2024Q1 and is then chained on NBS "
        "current-quarter preceding-year indices, where 104.3 means 4.3% above the same quarter "
        "a year earlier. Where the two overlap, the mean absolute gap between the index's implied "
        f"year-on-year and the NBS index is {china['difference_points'].abs().mean():.2f} points, "
        f"and the largest gap is {float(gap.difference_points):+.2f} points in {gap.quarter}. "
        "The USD check converts the same nominal goods trade at EXMAUS and deflates by US CPI. "
        "CPIAUCSL is missing October 2025, so 2025Q4 is incomplete and that check stops before the gap."
    )


def ardl_paragraph() -> str:
    bounds = pd.read_csv(TABLES / "ardl_bounds.csv").set_index("equation")
    long_run = pd.read_csv(TABLES / "ardl_long_run.csv")
    diagnostics = pd.read_csv(TABLES / "ardl_diagnostics.csv").set_index("equation")
    labels = {
        "log_foreign_gdp": "foreign GDP",
        "log_reer": "the REER",
        "log_commodity": "commodity prices",
        "log_exports": "exports",
        "const": "the intercept",
    }

    def equation_text(key: str, title: str, regressors: str) -> str:
        test = bounds.loc[key]
        diag = diagnostics.loc[key]
        slopes = long_run[(long_run["equation"] == key) & (long_run["variable"] != "ect")]
        bits = []
        for _, row in slopes.iterrows():
            if row["variable"] in ("const", "intercept"):
                continue
            bits.append(
                f"{labels.get(row['variable'], row['variable'])} {row['elasticity']:+.2f} "
                f"(se {row['std_error']:.2f}, {p_text(row['p_value'])})"
            )
        ect = long_run[(long_run["equation"] == key) & (long_run["variable"] == "ect")].iloc[0]
        ect_ok = float(ect["elasticity"]) < 0 and float(ect["p_value"]) < 0.05
        ect_call = (
            "negative and significant at 5%, so the levels error-correct"
            if ect_ok
            else "not a significant negative error correction at 5%"
        )
        bg_call = "does not reject" if diag["bg_pvalue"] >= 0.05 else "rejects"
        normal_call = "does not reject" if diag["jarque_bera_pvalue"] >= 0.05 else "rejects"
        decision = {
            "cointegration": "rejects no cointegration",
            "no cointegration": "does not reject no cointegration",
            "inconclusive": "is inconclusive",
        }[test["decision"]]
        lag_word = "lag" if int(diag["lag"]) == 1 else "lags"
        return (
            f"{title} is an ARDL with {int(diag['lag'])} {lag_word} of every variable: {regressors}. "
            f"The Pesaran bounds test (case {int(test['case'])}, unrestricted intercept and no trend) "
            f"has F = {test['f_stat']:.2f}. The 5% critical bounds are {test['lower_05']:.2f} and "
            f"{test['upper_05']:.2f}, so the test {decision}. "
            f"The long-run elasticities are {'; '.join(bits)}. "
            f"The error-correction coefficient is {ect['elasticity']:+.3f} "
            f"(se {ect['std_error']:.3f}, {p_text(ect['p_value'])}) and is {ect_call}. "
            f"Breusch-Godfrey at {int(diag['bg_lags'])} lags {bg_call} residual correlation "
            f"(p={diag['bg_pvalue']:.3f}). Jarque-Bera {normal_call} normality "
            f"(p={diag['jarque_bera_pvalue']:.3f}). The equation uses {int(test['nobs'])} observations."
        )

    export = equation_text(
        "export_demand",
        "Export demand",
        "exports on foreign GDP, the REER, and commodity prices, with the GFC, COVID, and "
        "trade-war dummies in the short run only. Imports are excluded",
    )
    imports = equation_text(
        "import_demand",
        "Import demand",
        "imports on exports, the REER, and commodity prices, with the same dummies in the short run only",
    )
    return export + " " + imports


def without_commodity_paragraph() -> str:
    relation = pd.read_csv(TABLES / "var_long_run_no_commodity.csv")
    paths = pd.read_csv(TABLES / "var_no_commodity.csv")
    exports = relation[(relation["normalised_on"] == "exports")].set_index("name")
    irf = paths[(paths["specification"] == "restricted") & (paths["name"] == "exports after foreign GDP")].iloc[0]
    short = paths[(paths["specification"] == "short_run") & (paths["name"] == "exports on foreign GDP")].iloc[0]
    return (
        "The same restricted VECM without commodity prices, on this sample and this lag, has "
        f"a partial long-run export elasticity to foreign GDP of {exports.loc['foreign GDP', 'elasticity']:.2f} "
        f"({p_text(exports.loc['foreign GDP', 'pvalue'])}), an import weight of "
        f"{exports.loc['imports', 'elasticity']:.2f}, and a foreign-GDP export response of "
        f"{irf['impact_percent']:+.2f}% on impact and {irf['horizon_1_percent']:+.2f}% one quarter later. "
        f"Its short-run export elasticity is {short['estimate']:.2f} ({p_text(short['pvalue'])}). "
        "Those are the figures the commodity-price control is being compared with. "
        "With commodity prices in the short-run equations the import coefficient in the solved "
        "relation is no longer that 1.35 processing-trade weight, and its standard error is large "
        "enough that the coefficient is not significant at 5%."
    )


def role_paragraph() -> str:
    long_run = pd.read_csv(TABLES / "ardl_long_run.csv")
    bounds = pd.read_csv(TABLES / "ardl_bounds.csv").set_index("equation")
    ect = long_run[(long_run["equation"] == "export_demand") & (long_run["variable"] == "ect")].iloc[0]
    decision = bounds.loc["export_demand", "decision"]
    ect_ok = float(ect["elasticity"]) < 0 and float(ect["p_value"]) < 0.05
    if decision == "cointegration" and ect_ok:
        support = (
            "The export-demand bounds test finds cointegration and the error-correction "
            "coefficient is negative and significant, so those ARDL elasticities are the "
            "long-run export-demand estimates."
        )
    elif ect_ok:
        support = (
            "The export error-correction coefficient is negative and significant. "
            f"The bounds test is {decision.replace('_', ' ')}, so the long-run elasticities "
            "are reported with that caveat."
        )
    else:
        support = (
            "The export error-correction coefficient is not a significant negative adjustment, "
            "so the ARDL levels equation is not a confirmed long-run demand curve. The elasticities "
            "are still the ones to read, because the VECM foreign-GDP coefficient is partialled "
            "on imports."
        )
    return (
        "The ARDL equations are the headline for long-run elasticities. The VECM is the model "
        "for shock dynamics: impulse responses, the forecast-error decomposition, and the "
        f"episode forecasts. {support} The VECM cointegrating vector still contains imports, "
        "so its foreign-GDP coefficient remains a partial association inside the processing-trade "
        "relation rather than an export-demand elasticity."
    )


def relation_slice(table: pd.DataFrame, specification: str, normalised_on: str) -> pd.DataFrame:
    part = table[(table["specification"] == specification) & (table["normalised_on"] == normalised_on)]
    return part.set_index("name")


def format_weight(row) -> str:
    if row["std_error"] == 0 or not np.isfinite(row["std_error"]):
        return f"{row['weight']:.2f}"
    return f"{row['weight']:.2f} (se {row['std_error']:.2f}, {p_text(row['pvalue'])})"


def restricted_paragraph(long_run: pd.DataFrame) -> str:
    exports = relation_slice(long_run, "restricted", "exports")
    imports = relation_slice(long_run, "restricted", "imports")
    alpha_bits = []
    for equation in ("exports", "imports", "REER"):
        row = exports.loc[f"alpha {equation}"]
        alpha_bits.append(
            f"{equation} {row['weight']:+.3f} (se {row['std_error']:.3f}, {p_text(row['pvalue'])})"
        )
    def term(name: str) -> str:
        value = float(exports.loc[name, "elasticity"]) if name != "constant" else -float(exports.loc[name, "weight"])
        return (
            f"{value:+.2f} (se {float(exports.loc[name, 'std_error']):.2f}, "
            f"{p_text(float(exports.loc[name, 'pvalue']))})"
        )

    return (
        "Foreign GDP is restricted to the cointegrating relation. Normalised so the export "
        f"weight is 1, the relation is exports = {term('REER')} REER {term('imports')} imports "
        f"{term('foreign GDP')} foreign GDP {term('constant')}. "
        "These are partial coefficients, holding the other variables in the relation fixed. "
        f"The long-run partial elasticity of exports to foreign demand is {exports.loc['foreign GDP', 'elasticity']:.2f} "
        f"({p_text(exports.loc['foreign GDP', 'pvalue'])}) and to the REER is "
        f"{exports.loc['REER', 'elasticity']:.2f} ({p_text(exports.loc['REER', 'pvalue'])}). "
        "The same vector, renormalised on imports, gives a long-run partial import elasticity of "
        f"{imports.loc['foreign GDP', 'elasticity']:.2f} to foreign demand "
        f"({p_text(imports.loc['foreign GDP', 'pvalue'])}) and "
        f"{imports.loc['REER', 'elasticity']:.2f} to the REER "
        f"({p_text(imports.loc['REER', 'pvalue'])}). "
        "Speeds of adjustment for the export-normalised equilibrium error are "
        + "; ".join(alpha_bits)
        + "."
        + (
            " A negative export loading means an export level above the relation is pulled back down."
            if float(exports.loc["alpha exports", "weight"]) < 0
            else " The export loading is positive."
        )
    )


def comparison_paragraph(short_paths: dict[str, np.ndarray], restricted_paths: dict[str, np.ndarray]) -> str:
    def at(paths, shock, horizon):
        return to_percent(paths[f"log_exports__{shock}"][horizon])

    short_fevd = pd.read_csv(TABLES / "var_fevd_short_run.csv")
    restricted_fevd = pd.read_csv(TABLES / "var_fevd.csv")

    def share(table, horizon, shock):
        row = table[(table["outcome"] == "exports") & (table["horizon"] == horizon) & (table["shock"] == shock)]
        return 100 * float(row["share"].iloc[0])

    return (
        "Compared with the short-run specification, a 1 s.d. foreign-GDP innovation moves "
        f"exports by {at(short_paths, 'log_foreign_gdp', 0):+.2f}% on impact "
        f"({at(short_paths, 'log_foreign_gdp', 4):+.2f}% at quarter 4). In the restricted "
        f"specification the impact is {at(restricted_paths, 'log_foreign_gdp', 0):+.2f}% and the "
        f"response one quarter later is {at(restricted_paths, 'log_foreign_gdp', 1):+.2f}% "
        f"({at(restricted_paths, 'log_foreign_gdp', 4):+.2f}% at quarter 4). "
        f"The REER export impact is {at(short_paths, 'log_reer', 0):+.2f}% in the short-run "
        f"specification and {at(restricted_paths, 'log_reer', 0):+.2f}% in the restricted one. "
        f"The TPU export impact is {at(short_paths, 'tpu', 0):+.2f}% and "
        f"{at(restricted_paths, 'tpu', 0):+.2f}%. "
        f"Eight quarters ahead, the export forecast-error shares in the short-run specification "
        f"are REER {share(short_fevd, 8, 'REER'):.0f}%, exports {share(short_fevd, 8, 'exports'):.0f}%, "
        f"imports {share(short_fevd, 8, 'imports'):.0f}%. In the restricted specification they are "
        f"REER {share(restricted_fevd, 8, 'REER'):.0f}%, exports {share(restricted_fevd, 8, 'exports'):.0f}%, "
        f"imports {share(restricted_fevd, 8, 'imports'):.0f}%."
    )


def preference_paragraph(long_run: pd.DataFrame) -> str:
    exports = relation_slice(long_run, "restricted", "exports")
    elasticity = float(exports.loc["foreign GDP", "elasticity"])
    elasticity_p = float(exports.loc["foreign GDP", "pvalue"])
    alpha = float(exports.loc["alpha exports", "weight"])
    alpha_p = float(exports.loc["alpha exports", "pvalue"])
    positive = elasticity > 0 and elasticity_p < 0.05
    corrects = alpha < 0 and alpha_p < 0.05
    if positive and corrects:
        choice = (
            "The restricted specification is preferred. It is the one that identifies a "
            "long-run export elasticity to foreign demand, that elasticity is positive and "
            "significant at 5%, and the export equation error-corrects."
        )
    elif positive:
        choice = (
            "The restricted specification is preferred for the long-run question, because the "
            "export elasticity to foreign demand is positive. The export adjustment speed is "
            "not a significant error correction at 5%, so the short-run specification remains "
            "the cleaner description of quarter-to-quarter comovement."
        )
    else:
        choice = (
            "The short-run specification is preferred. The restricted relation does not give a "
            "positive long-run export elasticity to foreign demand that is significant at 5%, "
            "so it is not a usable long-run export-demand curve. The short-run regression is "
            "the one that describes the quarterly association. The restricted estimates are "
            "still reported, because that is the specification in which a long-run elasticity "
            "would have been identified."
        )
    return (
        f"{choice} The short-run export elasticity reported above is a same-quarter coefficient, "
        "not the long-run trade elasticity."
    )


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
    later = []
    for shock, label in (("log_foreign_gdp", "foreign GDP"), ("log_reer", "the REER"), ("tpu", "TPU")):
        band = bands[f"log_exports__{shock}"]
        if any(excludes_zero(band[0, step], band[1, step]) for step in range(1, band.shape[1])):
            later.append(label)
    later_text = (
        "After the impact quarter the export bands include zero."
        if not later
        else "After the impact quarter the export band still excludes zero for " + join_and(later) + "."
    )
    return (
        f"Scaled to a one-standard-deviation AR(1) innovation, the restricted-specification "
        f"responses of exports are {to_percent(export_gdp[0]):+.2f}% on impact and "
        f"{to_percent(export_gdp[1]):+.2f}% one quarter later for foreign GDP, "
        f"{to_percent(export_reer[0]):+.2f}% for the REER, and {to_percent(export_tpu[0]):+.2f}% "
        f"for TPU. The matching import responses on impact are {to_percent(import_gdp[0]):+.2f}%, "
        f"{to_percent(import_reer[0]):+.2f}%, and {to_percent(import_tpu[0]):+.2f}%, and imports "
        f"move {to_percent(import_gdp[1]):+.2f}% one quarter after the foreign-GDP innovation. "
        f"The 90% bootstrap band for the export impact {gdp_band} for foreign GDP, "
        f"{reer_band} for the REER, and {tpu_band} for TPU. {later_text} "
        "Foreign GDP is zero on impact in this specification because it enters the "
        "cointegrating relation lagged one quarter. The foreign-GDP innovation has a standard "
        f"deviation of {100 * shock_scale['log_foreign_gdp']:.2f}%. The TPU innovation has a "
        f"standard deviation of {shock_scale['tpu']:.1f} index points. A positive REER shock "
        "is an appreciation."
    )


def long_run_paragraph(coefficients: pd.DataFrame) -> str:
    block = coefficients[coefficients["block"] == "long-run relation, exports normalised to 1"]
    speed_block = "adjustment speed" if (coefficients["block"] == "adjustment speed").any() else "adjustment speed 1"
    values = {row["name"]: row["estimate"] for _, row in block.iterrows()}
    speeds = {
        row["name"]: (row["estimate"], row["pvalue"])
        for _, row in coefficients[coefficients["block"] == speed_block].iterrows()
    }
    if not values:
        return "The trace test did not select a cointegrating relation, so there is no long-run vector to report."
    extra = ""
    n_relations = coefficients["block"].str.startswith("cointegrating vector").sum() // (len(ENDOG) + 1)
    if n_relations > 1:
        extra = f" This is the first of {n_relations} cointegrating relations."
    reer_p = speeds["REER"][1]
    reer_call = "not significant at 5%" if reer_p >= 0.05 else "significant at 5%"
    return (
        f"Scaled so that exports have a weight of one, the cointegrating relation is "
        f"exports = {values['REER']:.2f} REER + {values['imports']:.2f} imports "
        f"{values['constant']:+.2f}. The export adjustment coefficient is "
        f"{speeds['exports'][0]:+.2f} ({p_text(speeds['exports'][1])}). Import adjustment is "
        f"{speeds['imports'][0]:+.2f} ({p_text(speeds['imports'][1])}). The REER loading is "
        f"{speeds['REER'][0]:+.2f} ({p_text(reer_p)}) and is {reer_call}.{extra}"
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
        "On the full-sample point estimates, the ranking of the three external factors is "
        "by the absolute peak of the export response. Ending the sample in 2019Q4, the "
        f"same 1 s.d. foreign-GDP multiplier is {early['impact_percent']:+.2f}%, on an "
        f"innovation of {100 * early['innovation_sd']:.2f}% against "
        f"{100 * shock_scale['log_foreign_gdp']:.2f}% on the full sample. Splitting partners, "
        f"a 1 s.d. US GDP innovation moves exports by {us['impact_percent']:+.2f}% on impact, "
        f"while a 1 s.d. China GDP innovation moves them by {china['impact_percent']:+.2f}%."
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
        "this ordering. It is not evidence that an import shock causes exports."
    )


def episode_paragraph(episodes: dict) -> str:
    dummies = episodes["dummies"]
    sentences = []
    gfc = dummies[dummies["episode"] == "GFC"]
    if gfc["pvalue"].notna().all():
        sentences.append(dummy_sentence(dummies, "GFC"))
    sentences.append(forecast_sentence(episodes["forecasts"]["GFC"], "GFC"))
    sentences.append(dummy_sentence(dummies, "COVID"))
    sentences.append(dummy_sentence(dummies, "trade war"))
    sentences.append(forecast_sentence(episodes["forecasts"]["COVID"], "COVID"))
    sentences.append(commodity_gap_sentence(episodes["forecasts"]))
    sentences.append(forecast_sentence(episodes["forecasts"]["trade war"], "trade war"))
    return " ".join(sentences)


def dummy_sentence(dummies: pd.DataFrame, episode: str) -> str:
    part = dummies[dummies["episode"] == episode].set_index("equation")
    bits = []
    for equation in ("exports", "imports", "REER"):
        row = part.loc[equation]
        call = "significant at 5%" if row["pvalue"] < 0.05 else "not significant at 5%"
        bits.append(
            f"{equation} {to_percent(row['estimate']):+.1f}% ({p_text(row['pvalue'])}, {call})"
        )
    return (
        f"The {episode} dummy shifts quarterly log-growth while it equals one. "
        f"In percent, that shift is {'; '.join(bits)}. "
        "Error correction offsets a dummy that stays on, so these are not losses that "
        "compound quarter after quarter."
    )


def commodity_gap_sentence(forecasts: dict) -> str:
    with_prices = forecasts.get("COVID")
    without = forecasts.get("COVID, no commodity")
    if not with_prices or not without or not with_prices.get("ok") or not without.get("ok"):
        return "The commodity-price comparison of the 2021-22 export gap could not be estimated."
    left = with_prices["path"].set_index("quarter")["gap_percent"]
    right = without["path"].set_index("quarter")["gap_percent"]
    window = [quarter for quarter in left.index if "2021Q1" <= quarter <= "2022Q4"]
    gap_with = left.loc[window]
    gap_without = right.loc[window]
    mean_with = float(gap_with.mean())
    mean_without = float(gap_without.mean())
    if mean_without == 0:
        share = "the no-commodity gap averages zero"
    elif np.sign(mean_with) == np.sign(mean_without):
        share = f"{100 * mean_with / mean_without:.0f}% of the no-commodity gap remains"
    else:
        share = "the gap changes sign once commodity prices are included"
    peak = gap_with.abs().idxmax()
    return (
        "Both COVID forecasts run from 2020Q1 through 2022Q4. "
        f"Over 2021Q1-2022Q4 the average gap between actual exports and the pre-COVID forecast "
        f"is {mean_with:+.1f}% when commodity prices follow their actual path and "
        f"{mean_without:+.1f}% when the same pre-COVID model omits commodity prices, so {share}. "
        f"The largest remaining gap in that window is {gap_with.loc[peak]:+.1f}% in {peak}."
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
        (robustness["specification"] == "restricted")
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
    early = robustness[
        (robustness["specification"] == "through_2019Q4")
        & (robustness["outcome"] == "exports")
        & (robustness["shock"] == "foreign GDP")
    ].iloc[0]
    usd = robustness[
        (robustness["specification"] == "usd_cpi")
        & (robustness["outcome"] == "exports")
        & (robustness["shock"] == "foreign GDP")
    ].iloc[0]
    band = "excludes zero" if excludes_zero(lp0["percent_low_90"], lp0["percent_high_90"]) else "includes zero"
    long_run = pd.read_csv(TABLES / "var_long_run.csv")

    def elasticity(specification: str, normalised_on: str, name: str) -> str:
        row = long_run[
            (long_run["specification"] == specification)
            & (long_run["normalised_on"] == normalised_on)
            & (long_run["name"] == name)
        ].iloc[0]
        return f"{row['elasticity']:.2f} ({p_text(row['pvalue'])})"

    return (
        f"Jordà local projections of exports on the same 1 s.d. foreign-GDP innovation, "
        f"with Newey-West standard errors, put the impact at {lp0['percent']:+.2f}% "
        f"(the 90% band {band}) and the four-quarter response at {lp4['percent']:+.2f}%. "
        f"The restricted VECM impact is {base['impact_percent']:+.2f}% and the response one "
        f"quarter later is {base['horizon_1_percent']:+.2f}%. "
        f"Ending that specification in 2019Q4, the foreign-GDP export response one quarter "
        f"later is {early['horizon_1_percent']:+.2f}% and the TPU export impact is "
        f"{tpu_early['impact_percent']:+.2f}%. The long-run export elasticity to foreign demand "
        f"is {elasticity('through_2019Q4', 'exports', 'foreign GDP')} on the sample that ends "
        f"in 2019Q4, {elasticity('usd_cpi', 'exports', 'foreign GDP')} when trade is deflated "
        f"by US CPI (through {usd['sample_end']}, {int(usd['nobs'])} quarters), and "
        f"{elasticity('us_china', 'exports', 'US GDP')} for US GDP and "
        f"{elasticity('us_china', 'exports', 'China GDP')} for China GDP when those two "
        "replace the trade-weighted index inside the cointegrating relation. "
        f"The matching import elasticities are {elasticity('us_china', 'imports', 'US GDP')} "
        f"for the US and {elasticity('us_china', 'imports', 'China GDP')} for China."
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
        f"The log-likelihood is {float(np.real(result.llf)):.1f} on {int(result.nobs)} estimation observations."
    )


def caveat_paragraph(frame: pd.DataFrame, n_boot: int, shock_scale: dict) -> str:
    gfc = (
        "The GFC dummy varies in this sample."
        if frame["dummy_gfc"].nunique() > 1
        else "The GFC dummy is zero throughout this sample, so the 2008-09 episode is not identified."
    )
    return (
        f"The estimation sample has {len(frame)} quarters. "
        "The deflator is the headline consumer price index, so the real trade series is not a "
        "constant-price national-accounts series. Log commodity prices are included so that "
        "a rise in commodity prices is not read as a rise in trade volume. "
        "The VECM still puts imports in the cointegrating relation, which is the "
        "processing-trade link, and that coefficient is not used as the export-demand elasticity. "
        "KPSS did not reject stationarity of the three endogenous series in levels, so the "
        "unit-root reading is the ADF result and is not unanimous. "
        "The REER impulse response is a Cholesky shock with the exchange rate ordered first: "
        "within a quarter, exports and imports do not move the REER. In the restricted "
        "specification, foreign GDP enters only through the lagged cointegrating relation, "
        "so its dynamic multiplier is zero in the quarter the innovation arrives. "
        "Foreign GDP and TPU are imposed to be exogenous rather than tested against a model "
        "in which Malaysia feeds back into them. Their multipliers are one-quarter impulses "
        f"of the size of an AR(1) innovation ({100 * shock_scale['log_foreign_gdp']:.2f}% for "
        f"foreign GDP, {shock_scale['tpu']:.0f} index points for TPU), not the effect of a "
        "permanent rise in foreign demand. "
        "The trade-war variable is a step from 2018Q3, so it can absorb any shift in trade "
        f"growth that lines up with that date. {gfc} "
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
