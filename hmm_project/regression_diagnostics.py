"""
Regression-assumption diagnostics for the Recovery-vs-Disruption scatter.

Produces, for each regime (p=0.08 and p=0.04):
  - A 4-panel diagnostic figure: residuals-vs-fitted, Q-Q, scale-location,
    residuals-vs-leverage (with Cook's distance contours).
  - Formal tests: Breusch-Pagan (heteroscedasticity), Shapiro-Wilk (residual
    normality), Durbin-Watson (residual autocorrelation), Ramsey RESET
    (linearity / omitted non-linear terms).

All tests implemented manually with numpy + scipy.stats (no statsmodels).
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from config import DISRUPTION, RECOVERY, STEADY

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def _load_durations(processed_dir, filter_truncated=True):
    """Returns (x, y, n_total, n_excluded) arrays of disruption and recovery
    durations, optionally filtering right-censored runs (final state != Steady)."""
    train = np.load(os.path.join(processed_dir, "train.npz"), allow_pickle=True)
    test = np.load(os.path.join(processed_dir, "test.npz"), allow_pickle=True)
    all_states = list(train["states"]) + list(test["states"])
    x_all = np.array([int((s == DISRUPTION).sum()) for s in all_states], dtype=float)
    y_all = np.array([int((s == RECOVERY).sum()) for s in all_states], dtype=float)
    truncated = np.array([int(s[-1]) != STEADY for s in all_states])
    if filter_truncated:
        keep = ~truncated
        return x_all[keep], y_all[keep], len(all_states), int(truncated.sum())
    return x_all, y_all, len(all_states), 0


def _ols_simple(x, y):
    """Simple OLS with intercept. Returns slope, intercept, residuals,
    fitted, leverage, cook's distance, R^2, n."""
    n = len(x)
    X = np.column_stack([np.ones(n), x])      # n x 2 design matrix
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    intercept, slope = beta
    fitted = X @ beta
    resid = y - fitted

    # leverage = diagonal of hat matrix H = X (X'X)^-1 X'
    XtX_inv = np.linalg.inv(X.T @ X)
    leverage = np.einsum('ij,jk,ik->i', X, XtX_inv, X)

    # Cook's distance
    p = 2
    mse = float(np.sum(resid ** 2) / (n - p))
    cooks_d = (resid ** 2 / (p * mse)) * (leverage / (1 - leverage) ** 2)

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "slope": float(slope), "intercept": float(intercept),
        "residuals": resid, "fitted": fitted,
        "leverage": leverage, "cooks_d": cooks_d,
        "r_squared": r_squared, "n": int(n), "p": p, "mse": mse,
    }


# --------------- formal tests ---------------

def breusch_pagan(x, resid):
    """Regress squared residuals on x (plus intercept). BP stat = n*R^2_aux,
    distributed ~ chi^2(k-1) where k = # predictors incl. intercept."""
    sq = resid ** 2
    n = len(sq)
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, sq, rcond=None)
    pred = X @ beta
    ss_res = float(np.sum((sq - pred) ** 2))
    ss_tot = float(np.sum((sq - np.mean(sq)) ** 2))
    r2_aux = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    lm_stat = n * r2_aux
    p_value = 1.0 - stats.chi2.cdf(lm_stat, df=1)
    return {"statistic": float(lm_stat), "p_value": float(p_value),
            "df": 1, "null": "homoscedastic"}


def durbin_watson(resid):
    """DW = sum((e_t - e_{t-1})^2) / sum(e_t^2). ~2 under no autocorrelation."""
    diffs = np.diff(resid)
    dw = float(np.sum(diffs ** 2) / np.sum(resid ** 2))
    return {"statistic": dw, "null": "no first-order autocorrelation (DW ~ 2)"}


def ramsey_reset(x, y, fitted, resid, k=2):
    """Add fitted^2 ... fitted^(k+1) to the linear model and F-test their joint
    significance. Null: correctly specified (no missing nonlinear term)."""
    n = len(y)
    powers = np.column_stack([fitted ** (i + 2) for i in range(k)])
    X0 = np.column_stack([np.ones(n), x])
    X1 = np.column_stack([X0, powers])

    beta1, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid1 = y - X1 @ beta1

    rss0 = float(np.sum(resid ** 2))
    rss1 = float(np.sum(resid1 ** 2))
    q = k                       # restrictions
    df2 = n - (2 + k)           # residual dof under unrestricted
    f_stat = ((rss0 - rss1) / q) / (rss1 / df2) if rss1 > 0 else np.inf
    p_value = 1.0 - stats.f.cdf(f_stat, dfn=q, dfd=df2)
    return {"statistic": float(f_stat), "p_value": float(p_value),
            "df_num": q, "df_den": df2, "null": "correctly specified (linear)"}


# --------------- diagnostic plot ---------------

def diagnostic_plot(fit, title, output_path):
    resid = fit["residuals"]
    fitted = fit["fitted"]
    leverage = fit["leverage"]
    cooks = fit["cooks_d"]
    n = fit["n"]
    p = fit["p"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)

    # --- 1. Residuals vs Fitted ---
    ax = axes[0, 0]
    ax.scatter(fitted, resid, alpha=0.6, edgecolor='black', s=45)
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    # LOWESS-style moving trend via polynomial smoothing on sorted values
    order = np.argsort(fitted)
    if len(fitted) >= 10:
        smooth = np.poly1d(np.polyfit(fitted[order], resid[order], 2))(fitted[order])
        ax.plot(fitted[order], smooth, color='#c0392b', linewidth=1.5)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted\n(look for: fan shape = heteroscedasticity; curve = nonlinearity)", fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- 2. Normal Q-Q ---
    ax = axes[0, 1]
    # Standardized residuals
    std_resid = resid / np.sqrt(fit["mse"])
    (osm, osr), (slope_qq, inter_qq, _r) = stats.probplot(std_resid, dist='norm', plot=None)
    ax.scatter(osm, osr, alpha=0.6, edgecolor='black', s=45)
    lo, hi = osm.min(), osm.max()
    ax.plot([lo, hi], [slope_qq * lo + inter_qq, slope_qq * hi + inter_qq],
            color='red', linestyle='--', linewidth=1)
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Standardized residuals")
    ax.set_title("Normal Q-Q\n(look for: tail deviations from the line = non-normality)", fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- 3. Scale-Location (sqrt|standardized resid| vs fitted) ---
    ax = axes[1, 0]
    sqrt_abs = np.sqrt(np.abs(std_resid))
    ax.scatter(fitted, sqrt_abs, alpha=0.6, edgecolor='black', s=45)
    order = np.argsort(fitted)
    if len(fitted) >= 10:
        smooth = np.poly1d(np.polyfit(fitted[order], sqrt_abs[order], 2))(fitted[order])
        ax.plot(fitted[order], smooth, color='#c0392b', linewidth=1.5)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel(r"$\sqrt{|{\rm standardized\ residuals}|}$")
    ax.set_title("Scale-Location\n(look for: rising trend = increasing residual variance)", fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- 4. Residuals vs Leverage with Cook's distance contours ---
    ax = axes[1, 1]
    ax.scatter(leverage, std_resid, alpha=0.6, edgecolor='black', s=45)
    # draw Cook's distance contours at D = 0.5 and 1
    x_grid = np.linspace(max(leverage.min(), 1e-4), leverage.max() * 1.1, 200)
    for D, style, label in [(0.5, '--', "Cook's D = 0.5"), (1.0, ':', "Cook's D = 1.0")]:
        cd = np.sqrt(D * p * (1 - x_grid) ** 2 / x_grid)
        ax.plot(x_grid,  cd, color='red', linestyle=style, linewidth=1, label=label)
        ax.plot(x_grid, -cd, color='red', linestyle=style, linewidth=1)
    # annotate any point with Cook's D > 0.5
    flagged = np.where(cooks > 0.5)[0]
    for i in flagged:
        ax.annotate(str(i), (leverage[i], std_resid[i]),
                    textcoords="offset points", xytext=(6, 3), fontsize=8, color='red')
    ax.axhline(0, color='grey', linewidth=0.7)
    ax.set_xlabel("Leverage")
    ax.set_ylabel("Standardized residuals")
    ax.set_title(f"Residuals vs Leverage  (n={n}; {len(flagged)} point(s) with Cook's D > 0.5 labeled by run-index)",
                 fontsize=10)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.975])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# --------------- driver ---------------

def run_diagnostics_for_regime(processed_dir, label, output_dir,
                                filter_truncated=True):
    os.makedirs(output_dir, exist_ok=True)
    x, y, n_total, n_excluded = _load_durations(processed_dir, filter_truncated=filter_truncated)
    fit = _ols_simple(x, y)

    print(f"\n=== {label} ===")
    print(f"  total runs = {n_total}; truncated excluded = {n_excluded}; fitted n = {fit['n']}")
    print(f"  slope = {fit['slope']:.4f}, intercept = {fit['intercept']:.4f}, R² = {fit['r_squared']:.4f}")

    bp = breusch_pagan(x, fit["residuals"])
    sw = stats.shapiro(fit["residuals"])
    dw = durbin_watson(fit["residuals"])
    rs = ramsey_reset(x, y, fit["fitted"], fit["residuals"], k=2)

    print(f"  Breusch-Pagan (heteroscedasticity): LM={bp['statistic']:.3f}, p={bp['p_value']:.4f}  "
          f"[reject constant variance if p < 0.05]")
    print(f"  Shapiro-Wilk  (normality):          W={sw.statistic:.4f}, p={sw.pvalue:.6f}  "
          f"[reject normality if p < 0.05]")
    print(f"  Durbin-Watson (autocorrelation):    DW={dw['statistic']:.4f}  "
          f"[~2 = no autocorrelation]")
    print(f"  Ramsey RESET  (linearity):          F={rs['statistic']:.4f}, p={rs['p_value']:.4f}  "
          f"[reject correct specification if p < 0.05]")

    # Influential points
    n_high_cooks = int(np.sum(fit["cooks_d"] > 0.5))
    max_cooks_idx = int(np.argmax(fit["cooks_d"]))
    print(f"  Cook's distance: max = {fit['cooks_d'][max_cooks_idx]:.3f} (run-index {max_cooks_idx}); "
          f"{n_high_cooks} point(s) with D > 0.5")

    plot_path = os.path.join(output_dir, "regression_diagnostics.png")
    diagnostic_plot(fit, f"Regression Diagnostics — {label}", plot_path)

    return {
        "label": label, "fit": {k: (v.tolist() if hasattr(v, "tolist") else v)
                                 for k, v in fit.items()
                                 if k not in ("residuals", "fitted", "leverage", "cooks_d")},
        "breusch_pagan": bp, "shapiro_wilk": {"W": float(sw.statistic), "p_value": float(sw.pvalue)},
        "durbin_watson": dw, "ramsey_reset": rs,
        "cooks_max": float(fit["cooks_d"][max_cooks_idx]),
        "cooks_max_run_index": max_cooks_idx,
        "n_high_cooks": n_high_cooks,
    }


if __name__ == "__main__":
    here = os.path.dirname(__file__)

    summaries = []
    summaries.append(run_diagnostics_for_regime(
        os.path.join(here, "p08", "data", "processed"),
        "p = 0.08 baseline",
        os.path.join(here, "p08", "results"),
    ))
    summaries.append(run_diagnostics_for_regime(
        os.path.join(here, "p04", "data", "processed"),
        "p = 0.04 robustness check",
        os.path.join(here, "p04", "results"),
    ))

    # Save JSON summary
    json_path = os.path.join(RESULTS_DIR, "regression_diagnostics_summary.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(json_path, "w") as fp:
        json.dump(summaries, fp, indent=2)
    print(f"\nSaved JSON summary: {json_path}")
