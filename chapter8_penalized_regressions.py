"""
Chapter 8 Companion Code
Multivariate Analysis, Statistical Arbitrage and Penalized Regressions
Financial Engineering with AI, ML and Python -- Irene Aldridge

This script reproduces, in runnable form, every quantitative idea covered in
the chapter:

    1. Multi-factor regression: AAL ~ SPY + Oil, statsmodels + sklearn,
       plus a 3D scatter/plane visualization                        (8.1)
    2. Statistical arbitrage / pairs trading spread, extended to a
       multi-factor (CAPM-augmented) specification a la Aldridge & Li
       (2024)                                                        (8.2)
    3. Where OLS breaks down: high-dimensionality (p > n), multi-
       collinearity, and coefficient instability across resamples    (8.3)
    4. Ridge regression: closed-form vs. sklearn, and a rolling annual
       OLS-vs-Ridge beta comparison (Table 8.2) showing Ridge's
       year-to-year stability                                        (8.4)
    5. LASSO: sparse factor selection out of many candidate (mostly
       noise) predictors                                             (8.5)
    6. Elastic Net: grouped selection among correlated factors,
       plus a Ridge/LASSO/Elastic-Net method-selection reference     (8.6)
    7. Kempf-Memmel: minimum-variance portfolio weights recovered as
       regression coefficients, penalized-regression portfolio
       construction (OLS/Ridge/LASSO/Elastic Net), and a rolling-
       window backtest                                               (8.7)

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------

TRADING_DAYS = 252
YEARS = [2018, 2019, 2020, 2021, 2022]

np.random.seed(42)


# ---------------------------------------------------------------------------
# 1. Data: SPY, OIL, AAL, UAL daily returns, 2018-2022
# ---------------------------------------------------------------------------

def load_price_data():
    """
    Returns a DataFrame indexed by trading date with columns SPY, OIL, AAL,
    UAL (daily simple returns).

    """
    import yfinance as yf
    tickers = ["SPY", "CL=F", "AAL", "UAL"]
    prices = yf.download(tickers, start="2018-01-01", end="2023-01-01")["Adj Close"]
    prices = prices.rename(columns={"CL=F": "OIL"})
    return prices.pct_change().dropna()

# ---------------------------------------------------------------------------
# 2. Section 8.1: Multi-factor regression -- AAL ~ SPY + Oil
# ---------------------------------------------------------------------------

def single_vs_multifactor_regression(df):
    """
    Reproduces the chapter's headline comparison: single-factor CAPM
    (AAL ~ SPY) vs. two-factor (AAL ~ SPY + OIL), reporting the adjusted
    R^2 improvement from 0.183-style to 0.237-style (Section 8.1).
    """
    X1 = sm.add_constant(df["SPY"])
    single = sm.OLS(df["AAL"], X1).fit()

    X2 = sm.add_constant(df[["SPY", "OIL"]])
    multi = sm.OLS(df["AAL"], X2).fit()

    print("\n=== Single-Factor vs. Two-Factor CAPM for AAL ===")
    print(f"  Single-factor (SPY only):   Adj R^2 = {single.rsquared_adj:.3f}, "
          f"beta_SPY = {single.params['SPY']:.3f}")
    print(f"  Two-factor (SPY + OIL):     Adj R^2 = {multi.rsquared_adj:.3f}, "
          f"beta_SPY = {multi.params['SPY']:.3f}, beta_OIL = {multi.params['OIL']:.3f}")
    print(f"\n{multi.summary()}")

    return single, multi


def plot_3d_regression_plane(df, x_col="SPY", y_col="OIL", z_col="AAL"):
    """Reproduces the chapter's 3D visualization of the AAL ~ SPY + OIL fit."""
    X_fit = sm.add_constant(df[[x_col, y_col]])
    model = sm.OLS(df[z_col], X_fit).fit()
    alpha, beta_x, beta_y = model.params

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(df[x_col], df[y_col], df[z_col], s=4, alpha=0.4, color="steelblue")

    x_range = np.linspace(df[x_col].min(), df[x_col].max(), 20)
    y_range = np.linspace(df[y_col].min(), df[y_col].max(), 20)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    Z_grid = alpha + beta_x * X_grid + beta_y * Y_grid
    ax.plot_surface(X_grid, Y_grid, Z_grid, alpha=0.3, color="orange")

    ax.set_xlabel(f"{x_col} daily returns")
    ax.set_ylabel(f"{y_col} daily returns")
    ax.set_zlabel(f"{z_col} daily returns")
    ax.set_title(f"{z_col} ~ {x_col} + {y_col}: OLS Fit Plane")
    plt.tight_layout()
    plt.savefig("fig_8_1_3d_regression_plane.png", dpi=150)
    plt.show()

    return model


# ---------------------------------------------------------------------------
# 3. Section 8.2: Statistical arbitrage -- pairs spread, multi-factor extension
# ---------------------------------------------------------------------------

def pairs_spread(df, asset_i="AAL", asset_j="UAL"):
    """
    Classical two-asset stat-arb spread: epsilon_ij,t = log(p_i,t) - log(p_j,t)
    under the equal-notional-at-t0 simplification the chapter derives.
    """
    price_i = 100 * (1 + df[asset_i]).cumprod()
    price_j = 100 * (1 + df[asset_j]).cumprod()
    spread = np.log(price_i) - np.log(price_j)

    plt.figure(figsize=(10, 4))
    plt.plot(df.index, spread, color="purple", linewidth=1)
    plt.axhline(spread.mean(), color="black", linestyle="--", linewidth=1, label="mean")
    plt.title(f"Classical Pairs Spread: log({asset_i}) - log({asset_j})")
    plt.ylabel("Spread")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_8_2_classical_pairs_spread.png", dpi=150)
    plt.show()

    return spread, price_i, price_j


def multifactor_stat_arb(df, asset_i="AAL", asset_j="UAL", factor_cols=("OIL",)):
    """
    Reproduces the Aldridge & Li (2024) CAPM-extended stat-arb specification:

        log(p_i,t) = alpha + beta*log(p_j,t) + sum_k beta_Fk*log(F_k,t)
                      + gamma_i*log(p_i,t-1) - gamma_j*log(p_j,t-1) + eps_t

    Trading rule: alpha > 0 => i overpriced relative to j => sell i, buy j
                  alpha < 0 => i underpriced relative to j => buy i, sell j

    Because OIL returns can be negative (making log(F_k,t) undefined), we
    use a cumulative price-level proxy for each factor, consistent with the
    chapter's use of price levels (not returns) as regressors here.
    """
    price_i = 100 * (1 + df[asset_i]).cumprod()
    price_j = 100 * (1 + df[asset_j]).cumprod()
    factor_prices = {f: 100 * (1 + df[f]).cumprod() for f in factor_cols}

    log_i = np.log(price_i)
    log_j = np.log(price_j)
    log_j_lag = np.log(price_j.shift(1))
    log_i_lag = np.log(price_i.shift(1))
    log_factors = {f"log_{f}": np.log(factor_prices[f]) for f in factor_cols}

    reg_df = pd.DataFrame({
        "log_i": log_i,
        "log_j": log_j,
        "log_i_lag": log_i_lag,
        "log_j_lag": log_j_lag,
        **log_factors,
    }).dropna()

    X_cols = ["log_j", "log_i_lag", "log_j_lag"] + list(log_factors.keys())
    X = sm.add_constant(reg_df[X_cols])
    fit = sm.OLS(reg_df["log_i"], X).fit()

    print(f"\n=== Multi-Factor Stat-Arb Regression: log({asset_i}) ~ log({asset_j}) "
          f"+ {', '.join(factor_cols)} + lags ===")
    print(fit.summary())

    alpha_hat = fit.params["const"]
    signal = "SELL i / BUY j (i overpriced)" if alpha_hat > 0 else "BUY i / SELL j (i underpriced)"
    print(f"\nEstimated alpha = {alpha_hat:.5f}  ->  trading signal: {signal}")

    return fit


# ---------------------------------------------------------------------------
# 4. Section 8.3: Where OLS breaks down
# ---------------------------------------------------------------------------

def demonstrate_high_dimensionality_failure(n_obs=20, n_features=30):
    """
    Reproduces the p > n failure mode: with more candidate factors than
    trading days, X'X is singular and the OLS closed-form solution cannot
    be computed (or is wildly unstable via pinv).
    """
    np.random.seed(1)
    X = np.random.normal(0, 1, (n_obs, n_features))
    true_beta = np.zeros(n_features)
    true_beta[:3] = [0.5, -0.3, 0.2]
    y = X @ true_beta + np.random.normal(0, 0.1, n_obs)

    XtX = X.T @ X
    cond_number = np.linalg.cond(XtX)
    is_singular = np.linalg.matrix_rank(XtX) < n_features

    print(f"\n=== High-Dimensionality Failure (p={n_features} > n={n_obs}) ===")
    print(f"  Condition number of X'X: {cond_number:.3e}")
    print(f"  X'X is singular (rank-deficient): {is_singular}")
    print("  A regularization penalty (Ridge's + lambda*I) is required to invert this.")


def demonstrate_multicollinearity_instability(df, n_bootstrap=200):
    """
    Reproduces the multicollinearity failure mode: regressing AAL on the
    highly correlated pair (SPY, UAL) and showing how much the individual
    coefficient estimates swing across bootstrap resamples, even though the
    combined fit is stable.
    """
    X = df[["SPY", "UAL"]].values
    y = df["AAL"].values
    n = len(y)

    corr_spy_ual = df["SPY"].corr(df["UAL"])

    boot_coefs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        Xb, yb = X[idx], y[idx]
        beta_b = np.linalg.lstsq(np.column_stack([np.ones(n), Xb]), yb, rcond=None)[0]
        boot_coefs.append(beta_b[1:])   # drop intercept

    boot_coefs = np.array(boot_coefs)
    print(f"\n=== Multicollinearity Instability: AAL ~ SPY + UAL ===")
    print(f"  Corr(SPY, UAL): {corr_spy_ual:.3f}")
    print(f"  Bootstrap beta_SPY: mean={boot_coefs[:,0].mean():.3f}, std={boot_coefs[:,0].std():.3f}")
    print(f"  Bootstrap beta_UAL: mean={boot_coefs[:,1].mean():.3f}, std={boot_coefs[:,1].std():.3f}")
    print("  Large coefficient std devs relative to their means indicate the "
          "classic 'coefficients swing wildly' signature of collinearity.")

    return boot_coefs


# ---------------------------------------------------------------------------
# 5. Section 8.4: Ridge regression
# ---------------------------------------------------------------------------

def ridge_closed_form(X, y, lam):
    """Closed-form Ridge solution: beta_hat = (X'X + lambda*I)^-1 X'y (Appendix)."""
    X = np.asarray(X)
    y = np.asarray(y)
    n_features = X.shape[1]
    beta = np.linalg.inv(X.T @ X + lam * np.eye(n_features)) @ X.T @ y
    return beta


def verify_ridge_closed_form_vs_sklearn(df, lam=1.0):
    """Cross-checks the closed-form Ridge solution against sklearn's Ridge."""
    X = df[["SPY", "UAL"]].values
    y = df["AAL"].values

    beta_scratch = ridge_closed_form(X, y, lam)

    ridge = Ridge(alpha=lam, fit_intercept=False)
    ridge.fit(X, y)
    beta_sklearn = ridge.coef_

    print(f"\n=== Ridge Closed-Form vs. sklearn (lambda={lam}) ===")
    print(f"  Closed-form (X'X + lambda*I)^-1 X'y: {np.round(beta_scratch, 4)}")
    print(f"  sklearn Ridge.coef_:                  {np.round(beta_sklearn, 4)}")


def rolling_annual_ols_vs_ridge(df, lam=0.002):
    """
    Reproduces Table 8.2: annual OLS vs. Ridge estimates of AAL's market
    beta (controlling for UAL), showing Ridge's smoother year-to-year path.
    """
    results = []
    scaler_cache = {}
    for year in YEARS:
        sub = df[df.index.year == year]
        X = sub[["SPY", "UAL"]].values
        y = sub["AAL"].values

        ols = LinearRegression(fit_intercept=True).fit(X, y)
        ridge = Ridge(alpha=lam, fit_intercept=True).fit(X, y)

        results.append({
            "year": year,
            "ols_beta_spy": ols.coef_[0],
            "ridge_beta_spy": ridge.coef_[0],
        })

    res = pd.DataFrame(results)
    print("\n=== Table 8.2: OLS vs. Ridge Market Beta (AAL, controlling for UAL) ===")
    print(res.round(3).to_string(index=False))
    print(f"\n  Std dev across years -- OLS: {res['ols_beta_spy'].std():.3f}, "
          f"Ridge: {res['ridge_beta_spy'].std():.3f}  (Ridge should be smaller)")

    plt.figure(figsize=(8, 5))
    plt.plot(res["year"], res["ols_beta_spy"], marker="o", label="OLS beta_SPY")
    plt.plot(res["year"], res["ridge_beta_spy"], marker="s", label=f"Ridge beta_SPY (lambda={lam})")
    plt.title("OLS vs. Ridge Market Beta of AAL by Year")
    plt.xlabel("Year")
    plt.ylabel("beta_SPY")
    plt.legend()
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig("fig_8_ols_vs_ridge_beta.png", dpi=150)
    plt.show()

    return res


# ---------------------------------------------------------------------------
# 6. Section 8.5: LASSO -- sparse factor models
# ---------------------------------------------------------------------------

def build_candidate_factor_set(df, n_noise_factors=40):
    """
    Builds a wide candidate-factor matrix for AAL: the two genuine factors
    (SPY, OIL) plus many noise factors with no true relationship to AAL
    returns -- the "50+ candidate factors" scenario Section 8.5 describes.
    """
    n = len(df)
    noise = pd.DataFrame(
        np.random.normal(0, 0.01, (n, n_noise_factors)),
        index=df.index,
        columns=[f"noise_{i+1}" for i in range(n_noise_factors)],
    )
    X = pd.concat([df[["SPY", "OIL"]], noise], axis=1)
    y = df["AAL"]
    return X, y


def lasso_sparse_selection(df, alpha=0.0003, n_noise_factors=40):
    """
    Fits LASSO on the wide candidate-factor set and shows it recovers the
    two genuine factors (SPY, OIL) while zeroing out most/all of the noise.
    Standardization is applied first, per the chapter's explicit warning.
    """
    X, y = build_candidate_factor_set(df, n_noise_factors)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso = Lasso(alpha=alpha, max_iter=20000)
    lasso.fit(X_scaled, y)

    nonzero = pd.Series(lasso.coef_, index=X.columns)
    nonzero = nonzero[nonzero != 0].sort_values(key=np.abs, ascending=False)

    print(f"\n=== LASSO Sparse Factor Selection (alpha={alpha}, "
          f"{X.shape[1]} candidate factors) ===")
    print(f"  Nonzero coefficients: {len(nonzero)} / {X.shape[1]}")
    print(nonzero.round(5).to_string())
    print("  A well-behaved LASSO fit should keep SPY and OIL and drop "
          "most/all noise_* factors.")

    return lasso, nonzero


# ---------------------------------------------------------------------------
# 7. Section 8.6: Elastic Net -- grouped selection
# ---------------------------------------------------------------------------

def elastic_net_grouped_selection(df, alpha=0.0003, l1_ratio=0.5):
    """
    Demonstrates Elastic Net's grouped-selection behavior on a correlated
    pair (AAL, UAL both regressed against SPY/OIL/and each other's proxy),
    comparing coefficients against pure LASSO and pure Ridge on the same
    wide candidate set from Section 8.5.
    """
    X, y = build_candidate_factor_set(df, n_noise_factors=40)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso = Lasso(alpha=alpha, max_iter=20000).fit(X_scaled, y)
    ridge = Ridge(alpha=1.0).fit(X_scaled, y)
    enet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000, random_state=42).fit(X_scaled, y)

    comparison = pd.DataFrame({
        "LASSO": lasso.coef_,
        "Ridge": ridge.coef_,
        "ElasticNet": enet.coef_,
    }, index=X.columns)

    print(f"\n=== LASSO vs. Ridge vs. Elastic Net (alpha={alpha}, l1_ratio={l1_ratio}) ===")
    print(f"  Nonzero coefficients -- LASSO: {(lasso.coef_ != 0).sum()}, "
          f"Ridge: {(ridge.coef_ != 0).sum()}, ElasticNet: {(enet.coef_ != 0).sum()}")
    print(comparison.loc[["SPY", "OIL"]].round(5))

    return comparison


METHOD_SELECTION_GUIDE = pd.DataFrame([
    {"Scenario": "Many correlated predictors (sector stocks)",
     "Recommended": "Ridge or Elastic Net", "Reason": "Ridge handles multicollinearity; Elastic Net groups selection"},
    {"Scenario": "Sparse signal (few factors matter)",
     "Recommended": "LASSO", "Reason": "Aggressive zeroing removes noise variables"},
    {"Scenario": "Uncertainty about sparsity",
     "Recommended": "Elastic Net", "Reason": "Interpolates between Ridge and LASSO"},
    {"Scenario": "Need stability (rolling windows)",
     "Recommended": "Ridge", "Reason": "Always converges; differentiable objective"},
    {"Scenario": "Need interpretability",
     "Recommended": "LASSO or Elastic Net", "Reason": "Sparse solutions easier to explain"},
    {"Scenario": "p >> n (more features than observations)",
     "Recommended": "Ridge or Elastic Net", "Reason": "LASSO selects at most n variables"},
])


def print_method_selection_guide():
    print("\n=== Method Selection Guide ===")
    print(METHOD_SELECTION_GUIDE.to_string(index=False))


# ---------------------------------------------------------------------------
# 8. Section 8.7: Kempf-Memmel -- minimum-variance portfolio via regression
# ---------------------------------------------------------------------------

def min_var_weights(returns_df, method="ols", alpha=1.0, l1_ratio=0.5):
    """
    Estimates global minimum-variance portfolio weights via the Kempf-Memmel
    (2006) regression: pick the last asset as reference N, regress R_N on
    (R_N - R_i) for i = 1..N-1. The coefficients ARE the weights w_1..w_{N-1};
    w_N = 1 - sum(coefficients). Reproduces the chapter's exact function.
    """
    tickers = returns_df.columns.tolist()
    ref = tickers[-1]

    X = pd.DataFrame(index=returns_df.index)
    for t in tickers[:-1]:
        X[t] = returns_df[ref] - returns_df[t]
    y = returns_df[ref].values

    if method == "ols":
        model = LinearRegression(fit_intercept=True)
    elif method == "ridge":
        model = Ridge(alpha=alpha, fit_intercept=True)
    elif method == "lasso":
        model = Lasso(alpha=alpha, max_iter=10000, fit_intercept=True)
    elif method == "elasticnet":
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, fit_intercept=True)
    else:
        raise ValueError(f"Unknown method: {method}")

    model.fit(X.values, y)
    betas = model.coef_
    w_N = 1.0 - betas.sum()
    weights = np.append(betas, w_N)
    return pd.Series(weights, index=tickers)


def verify_kempf_memmel_equivalence(returns_df):
    """
    Sanity check: the Kempf-Memmel OLS-regression weights should match the
    closed-form minimum-variance weights w* = (Sigma^-1 1) / (1' Sigma^-1 1)
    up to numerical tolerance.
    """
    cov = returns_df.cov().values
    ones = np.ones(len(returns_df.columns))
    inv_cov = np.linalg.inv(cov)
    w_closed_form = (inv_cov @ ones) / (ones @ inv_cov @ ones)

    w_regression = min_var_weights(returns_df, method="ols").values

    print("\n=== Verifying Kempf-Memmel Equivalence ===")
    print(f"  Closed-form (Sigma^-1 1)/(1'Sigma^-1 1): {np.round(w_closed_form, 4)}")
    print(f"  Kempf-Memmel OLS regression weights:      {np.round(w_regression, 4)}")
    print(f"  Max abs difference: {np.max(np.abs(w_closed_form - w_regression)):.2e} (should be ~0)")


def rolling_backtest(returns_df, W=252, method="ridge", alpha=1.0):
    """
    Rolling-window backtest of penalized-regression portfolio weights:
    estimate weights in-sample over W days, apply them out-of-sample over
    the next W days, roll forward. Reproduces the chapter's exact function.
    """
    results = []
    T, N = returns_df.shape

    for i in range(0, T - 2 * W, W):
        train = returns_df.iloc[i:i + W]
        test = returns_df.iloc[i + W:i + 2 * W]

        w = min_var_weights(train, method=method, alpha=alpha)

        oos_returns = (test * w).sum(axis=1)
        oos_total = (1 + oos_returns).prod() - 1
        oos_vol = oos_returns.std() * np.sqrt(TRADING_DAYS)
        oos_sharpe = (oos_returns.mean() / oos_returns.std() * np.sqrt(TRADING_DAYS)
                      if oos_returns.std() > 0 else np.nan)

        results.append({
            "train_start": train.index[0],
            "test_start": test.index[0],
            "weights": w.to_dict(),
            "oos_return": oos_total,
            "oos_vol": oos_vol,
            "oos_sharpe": oos_sharpe,
        })

    return pd.DataFrame(results)


def compare_portfolio_methods(returns_df, W=252, alpha_ridge=1.0, alpha_lasso=0.00005):
    """Runs the rolling backtest for OLS, Ridge, LASSO, and Elastic Net and compares."""
    methods = {
        "ols": dict(method="ols", alpha=1.0),
        "ridge": dict(method="ridge", alpha=alpha_ridge),
        "lasso": dict(method="lasso", alpha=alpha_lasso),
        "elasticnet": dict(method="elasticnet", alpha=alpha_lasso),
    }

    summary = {}
    for name, kwargs in methods.items():
        bt = rolling_backtest(returns_df, W=W, **kwargs)
        summary[name] = {
            "mean_oos_return": bt["oos_return"].mean(),
            "mean_oos_vol": bt["oos_vol"].mean(),
            "mean_oos_sharpe": bt["oos_sharpe"].mean(),
            "n_folds": len(bt),
        }

    summary_df = pd.DataFrame(summary).T
    print(f"\n=== Rolling-Window Backtest: OLS vs. Ridge vs. LASSO vs. Elastic Net "
          f"(W={W}) ===")
    print(summary_df.round(4))

    return summary_df


PORTFOLIO_STRATEGY_MAP = pd.DataFrame([
    {"Portfolio Strategy": "Equal weight (1/N)",
     "Connection to Penalized Regression": "Extreme Ridge limit: maximum regularization, all weights equal"},
    {"Portfolio Strategy": "Volatility weighting (1/sigma^2)",
     "Connection to Penalized Regression": "Diagonal approximation to minimum variance; ignores covariances"},
    {"Portfolio Strategy": "Minimum variance (OLS)",
     "Connection to Penalized Regression": "Kempf-Memmel regression with no penalty"},
    {"Portfolio Strategy": "Sparse minimum variance",
     "Connection to Penalized Regression": "LASSO or Elastic Net applied to Kempf-Memmel regression"},
    {"Portfolio Strategy": "Regularized minimum variance",
     "Connection to Penalized Regression": "Ridge applied to Kempf-Memmel regression"},
])


# ---------------------------------------------------------------------------
# Main: run the whole chapter end to end
# ---------------------------------------------------------------------------

def main():
    # 1. Data
    df = load_price_data()

    # 2. Section 8.1: single-factor vs. multi-factor CAPM
    single, multi = single_vs_multifactor_regression(df)
    plot_3d_regression_plane(df, "SPY", "OIL", "AAL")

    # 3. Section 8.2: stat arb spread, classical and multi-factor
    pairs_spread(df, "AAL", "UAL")
    multifactor_stat_arb(df, "AAL", "UAL", factor_cols=("OIL",))

    # 4. Section 8.3: where OLS breaks down
    demonstrate_high_dimensionality_failure(n_obs=20, n_features=30)
    demonstrate_multicollinearity_instability(df)

    # 5. Section 8.4: Ridge regression
    verify_ridge_closed_form_vs_sklearn(df, lam=1.0)
    rolling_annual_ols_vs_ridge(df, lam=0.002)

    # 6. Section 8.5: LASSO sparse factor selection
    lasso_sparse_selection(df, alpha=0.0003, n_noise_factors=40)

    # 7. Section 8.6: Elastic Net
    elastic_net_grouped_selection(df, alpha=0.0003, l1_ratio=0.5)
    print_method_selection_guide()

    # 8. Section 8.7: Kempf-Memmel portfolio optimization
    universe = load_portfolio_universe(n_assets=8, n_days=TRADING_DAYS * 3)
    verify_kempf_memmel_equivalence(universe)

    ols_weights = min_var_weights(universe, method="ols")
    ridge_weights = min_var_weights(universe, method="ridge", alpha=1.0)
    lasso_weights = min_var_weights(universe, method="lasso", alpha=0.00005)
    enet_weights = min_var_weights(universe, method="elasticnet", alpha=0.00005, l1_ratio=0.5)

    weights_table = pd.DataFrame({
        "OLS": ols_weights, "Ridge": ridge_weights,
        "LASSO": lasso_weights, "ElasticNet": enet_weights,
    })
    print("\n=== Minimum-Variance Portfolio Weights by Method ===")
    print(weights_table.round(4))

    compare_portfolio_methods(universe, W=TRADING_DAYS)

    print("\n=== Portfolio-Strategy <-> Penalized-Regression Map ===")
    print(PORTFOLIO_STRATEGY_MAP.to_string(index=False))

    print("\nAll figures saved as PNG files in the working directory.")


if __name__ == "__main__":
    main()
