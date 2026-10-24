"""Cross-check app/engine/factors/regression.py against statsmodels (independent implementation).

statsmodels is NOT a project dependency. Run this in a throwaway virtualenv that has the
project (for the engine and cache code) plus statsmodels, e.g. from the repo root on Windows:

    .venv\\Scripts\\python -m venv %TEMP%\\sm-venv          # same Python 3.12 as the project
    %TEMP%\\sm-venv\\Scripts\\python -m pip install -e apps/api statsmodels
    %TEMP%\\sm-venv\\Scripts\\python scripts\\crosscheck_factor_regression.py

(bash: python -m venv /tmp/sm-venv && /tmp/sm-venv/bin/pip install -e apps/api statsmodels)

Data: the equal-weight portfolio of the 20 universe stocks regressed on the IIMA factors if
both caches are seeded (scripts/seed_data.py), otherwise synthetic returns with
heteroskedastic AR(1) errors. Use --synthetic to force the latter.

statsmodels settings that correspond to our convention:
    OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": L, "use_correction": True})
The script also fits use_correction=False, statsmodels' default for HAC, to show that it
differs from ours by exactly the factor n / (n - p) in the covariance.

Exit code 0 if every compared number agrees to a relative 1e-8, 1 otherwise, 2 if
statsmodels is not installed.
"""

import argparse
import math
import sys

import numpy as np
import pandas as pd

from app.engine.factors.regression import DEFAULT_FACTORS, factor_regression, newey_west_lags

RTOL = 1e-8


def synthetic(n: int = 2000, seed: int = 42) -> tuple[pd.Series, pd.DataFrame, str]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-01", periods=n, name="date")
    f = rng.normal(size=(n, 4)) * [0.012, 0.007, 0.008, 0.009]
    rf = np.full(n, 0.00025)
    vol = 0.004 * (1 + np.abs(np.sin(np.arange(n) / 40)))  # heteroskedastic
    e = np.zeros(n)
    shocks = rng.normal(size=n) * vol
    for t in range(1, n):
        e[t] = 0.2 * e[t - 1] + shocks[t]  # autocorrelated
    excess = 0.0002 + f @ np.array([0.95, -0.3, 0.2, 0.05]) + e
    factors = pd.DataFrame(np.column_stack([f, rf]), index=idx, columns=[*DEFAULT_FACTORS, "RF"])
    return pd.Series(excess + rf, index=idx), factors, f"synthetic, n={n}"


def real_cache() -> tuple[pd.Series, pd.DataFrame, str] | None:
    from app.data.errors import NotInCacheError
    from app.data.factor_repository import ParquetFactorRepository
    from app.data.repository import ParquetRepository
    from app.data.universe import load_universe
    from app.engine.returns import simple_returns
    from app.engine.risk.var import portfolio_returns
    from app.services.factors import load_aligned_factors
    from app.services.market_data import load_aligned_prices

    try:
        universe = load_universe()
        prices = load_aligned_prices(ParquetRepository(), universe).prices
        aligned = load_aligned_factors(ParquetFactorRepository("iima_ff4_daily"), prices.index)
    except (FileNotFoundError, NotInCacheError):
        return None
    factors = aligned.factors.dropna()
    stocks = [s.asset().id for s in universe.stocks]
    returns = simple_returns(prices)[stocks].loc[factors.index]
    ew = portfolio_returns(returns, {s: 1 / len(stocks) for s in stocks})
    return ew, factors, f"EW 20 stocks on IIMA factors, n={len(factors)}"


def compare(label: str, ours: np.ndarray, theirs: np.ndarray) -> bool:
    ours, theirs = np.atleast_1d(ours), np.atleast_1d(theirs)
    rel = np.max(np.abs(ours - theirs) / np.maximum(np.abs(theirs), 1e-300))
    ok = bool(rel <= RTOL)
    print(f"  {label:<26} max rel diff {rel:.2e}  {'ok' if ok else 'MISMATCH'}")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--synthetic", action="store_true", help="ignore the real cache")
    parser.add_argument("--lags", type=int, default=None, help="HAC lags (default: NW rule)")
    args = parser.parse_args(argv)

    try:
        import statsmodels
        import statsmodels.api as sm
    except ImportError:
        print("statsmodels is not installed in this interpreter; see this script's docstring")
        return 2

    data = None if args.synthetic else real_cache()
    returns, factors, label = data if data is not None else synthetic()
    lags = newey_west_lags(len(returns)) if args.lags is None else args.lags
    names = list(DEFAULT_FACTORS)
    print(f"Data: {label}; HAC lags = {lags}; statsmodels {statsmodels.__version__}")

    ours = factor_regression(returns, factors, hac_lags=lags)
    y = (returns - factors["RF"]).to_numpy()
    X = sm.add_constant(factors[names].to_numpy(), has_constant="add")
    fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True})
    plain = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})

    keys = ["alpha", *names]
    our_params = np.array([ours.alpha_daily, *(ours.betas[k] for k in names)])
    our_se = np.array([ours.std_errors[k] for k in keys])
    our_t = np.array([ours.t_stats[k] for k in keys])
    our_p = np.array([ours.p_values[k] for k in keys])
    n, p = X.shape

    print("\nOurs vs statsmodels HAC(use_correction=True):")
    checks = [
        compare("coefficients", our_params, fit.params),
        compare("HAC standard errors", our_se, fit.bse),
        compare("t-statistics", our_t, fit.tvalues),
        compare("p-values (normal)", our_p, fit.pvalues),
        compare("R-squared", np.array(ours.r_squared), np.array(fit.rsquared)),
        compare("adj R-squared", np.array(ours.adj_r_squared), np.array(fit.rsquared_adj)),
        compare(
            "residual vol (annual)",
            np.array(ours.residual_vol_annual),
            np.array(math.sqrt(fit.mse_resid * 252)),
        ),
    ]
    print("\nstatsmodels default HAC (use_correction=False) vs ours:")
    ratio = (plain.bse / fit.bse) ** 2
    print(f"  variance ratio {ratio.min():.10f}..{ratio.max():.10f}; (n-p)/n = {(n - p) / n:.10f}")
    checks.append(compare("ratio == (n-p)/n", ratio, np.full(p, (n - p) / n)))

    print("\nOur result:")
    for k in keys:
        coef = ours.alpha_annual if k == "alpha" else ours.betas[k]
        unit = " (annual)" if k == "alpha" else ""
        print(f"  {k:<7}{unit:<10} {coef: .4f}   t = {ours.t_stats[k]: .2f}")
    print(f"  R^2 = {ours.r_squared:.4f}, residual vol = {ours.residual_vol_annual:.4f}")

    ok = all(checks)
    print("\nALL MATCH" if ok else "\nMISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
