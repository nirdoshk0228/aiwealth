"""Factor exposure service: Carhart four-factor regression of the portfolio and its holdings.

Uses every return row that has factor data (2015-01-05 -> 2025-12-31 for IIMA release 2025-12),
exactly as docs/methodology.md describes; rows after the factor file ends are excluded.
"""

from app.engine.factors.regression import factor_regression, factor_regression_by_asset
from app.engine.risk.var import portfolio_returns
from app.schemas.analytics import (
    AssetFactorRow,
    FactorCoefficient,
    FactorFit,
    FactorRequest,
    FactorResponse,
)
from app.services._common import engine_call, num, weights_for, window_of
from app.services.market import MarketData

MODEL = "Carhart four-factor: r - RF = alpha + b_MKT MKT_RF + b_SMB SMB + b_HML HML + b_WML WML"


def factor_report(md: MarketData, req: FactorRequest) -> FactorResponse:
    w = weights_for(md, req.portfolio)
    f = md.require_factors().factors.dropna()
    r = md.returns.loc[f.index]
    port = engine_call(lambda: portfolio_returns(r, w))
    fit = engine_call(lambda: factor_regression(port, f))
    assets: list[AssetFactorRow] = []
    if req.include_assets:
        table = engine_call(lambda: factor_regression_by_asset(r[list(w)], f))
        for asset_id, row in table.iterrows():
            assets.append(
                AssetFactorRow(
                    asset_id=str(asset_id),
                    name=md.name(str(asset_id)),
                    alpha_annual=float(row["alpha_annual"]),
                    betas={k: float(row[f"beta_{k}"]) for k in fit.factors},
                    t_stats={k: num(row[f"t_{k}"]) for k in fit.factors},
                    r_squared=float(row["r_squared"]),
                )
            )
    meta = md.factor_meta
    source = (
        f"{meta.source}, release {meta.release}. Cite: {meta.citation}"
        if meta is not None
        else "factor cache (metadata unavailable)"
    )
    return FactorResponse(
        window=window_of(f.index),
        model=MODEL,
        source=source,
        portfolio=FactorFit(
            alpha_annual=fit.alpha_annual,
            alpha_t_stat=num(fit.t_stats["alpha"]),
            coefficients=[
                FactorCoefficient(
                    factor=k,
                    beta=fit.betas[k],
                    std_error=fit.std_errors[k],
                    t_stat=num(fit.t_stats[k]),
                    p_value=num(fit.p_values[k]),
                )
                for k in fit.factors
            ],
            r_squared=fit.r_squared,
            adj_r_squared=fit.adj_r_squared,
            residual_vol_annual=fit.residual_vol_annual,
            n_obs=fit.n_obs,
            hac_lags=fit.hac_lags,
            warnings=list(fit.warnings),
        ),
        assets=assets,
        notes=[
            "Standard errors are Newey-West (HAC), robust to heteroskedasticity and "
            "autocorrelation; t-statistics above about 2 in absolute value are conventionally "
            "called significant.",
            "MKT is IIMA's broad value-weighted market (mid and small caps included), not the "
            "Nifty 50, so large-cap holdings load negatively on SMB.",
            "For debt, liquid and gold funds R-squared is near 0: the factors do not explain "
            "them and their betas carry little information.",
            "A positive alpha here is not evidence of skill: the universe was chosen in 2026 "
            "from surviving, popular funds and current index members (survivorship bias).",
            f"Only rows with factor data are used; the factor file ends on {f.index[-1].date()}.",
        ],
    )
