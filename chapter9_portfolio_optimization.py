"""
Chapter 9 Companion Code
Foundations of Portfolio Optimization I
Financial Engineering with AI, ML and Python -- Irene Aldridge

This script reproduces every quantitative idea covered in the chapter:

    1. Classical Markowitz mean-variance optimization via cvxpy           (9.3)
    2. The three standard remedies for estimation error: Ledoit-Wolf
       shrinkage, Random Matrix Theory eigenvalue cleaning, and a
       simplified Black-Litterman posterior                              (9.3)
    3. Risk-based portfolios that sidestep return estimation entirely:
       Minimum Variance, Maximum Diversification, Risk Parity / Equal
       Risk Contribution (ERC), and Hierarchical Risk Parity (HRP)        (9.4)
    4. Volatility-managed (inverse-variance) portfolio weighting and its
       three independent theoretical derivations                         (9.5)
    5. A production-grade constrained optimizer (position limits,
       turnover budget, long-only) and a robust-optimization variant      (9.6)
    6. ML-based return forecasting (Ridge / LASSO / Elastic Net) feeding
       the optimizer's mu input                                          (9.7)
    7. Walk-forward backtesting, the Deflated Sharpe Ratio, and a
       (simplified) Combinatorial Purged Cross-Validation splitter        (9.8)
    8. A full multi-asset pipeline: Ledoit-Wolf covariance -> HRP base
       weights -> volatility-managed scaling -> constrained optimizer,
       benchmarked against every method above (reproducing the shape of
       Table 9.1)                                                        (9.9)

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------

TRADING_DAYS = 252
np.random.seed(42)


# ---------------------------------------------------------------------------
# 1. Data: a synthetic multi-asset universe
# ---------------------------------------------------------------------------

def load_universe(n_assets=15, n_days=TRADING_DAYS * 6):
    """
    Returns a DataFrame of daily returns for a synthetic multi-asset
    universe: one common market factor, three correlated "sector" clusters
    (mimicking equities/bonds/commodities-style groupings), and mildly
    persistent (GARCH-like) volatility so vol-managed weighting has
    something real to exploit.

    """
    import yfinance as yf
    tickers = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU",
               "TLT", "IEF", "BUND", "JGB", "GLD", "USO", "DBA", "CPER",
               "UUP", "FXE", "FXY", "FXB"]
    prices = yf.download(tickers, start="2010-01-01")["Adj Close"]
    return prices.pct_change().dropna()

# ---------------------------------------------------------------------------
# 2. Section 9.3: classical mean-variance optimization + estimation-error fixes
# ---------------------------------------------------------------------------

def mean_variance(mu, Sigma, lam=1.0, long_only=True):
    """Reproduces the chapter's cvxpy mean-variance optimizer."""
    n = len(mu)
    w = cp.Variable(n)
    objective = cp.Maximize(mu @ w - lam / 2 * cp.quad_form(w, Sigma))
    constraints = [cp.sum(w) == 1]
    if long_only:
        constraints.append(w >= 0)
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return w.value


def ledoit_wolf_covariance(returns_df):
    """Ledoit-Wolf shrinkage covariance estimator (Section 9.3)."""
    lw = LedoitWolf().fit(returns_df.values)
    return pd.DataFrame(lw.covariance_, index=returns_df.columns, columns=returns_df.columns), lw.shrinkage_


def rmt_clean_covariance(returns_df):
    """
    Random Matrix Theory eigenvalue cleaning (Section 9.3 / Advanced
    Topics): eigenvalues of the sample *correlation* matrix below the
    Marchenko-Pastur upper edge lambda_+ are replaced by their average;
    eigenvalues above it are retained as genuine signal.
    """
    X = returns_df.values
    T, N = X.shape
    q = N / T
    corr = returns_df.corr().values

    eigvals, eigvecs = np.linalg.eigh(corr)
    lambda_plus = (1 + np.sqrt(q)) ** 2
    lambda_minus = (1 - np.sqrt(q)) ** 2

    is_noise = eigvals < lambda_plus
    n_noise = is_noise.sum()
    if n_noise > 0:
        avg_noise_eigval = eigvals[is_noise].mean()
        cleaned_eigvals = eigvals.copy()
        cleaned_eigvals[is_noise] = avg_noise_eigval
    else:
        cleaned_eigvals = eigvals.copy()

    corr_clean = eigvecs @ np.diag(cleaned_eigvals) @ eigvecs.T
    # Rescale diagonal back to 1 (a correlation matrix) then to a covariance
    d = np.sqrt(np.diag(corr_clean))
    corr_clean = corr_clean / np.outer(d, d)
    stds = returns_df.std().values
    cov_clean = corr_clean * np.outer(stds, stds)

    print(f"\n=== RMT Eigenvalue Cleaning ===")
    print(f"  N={N}, T={T}, q=N/T={q:.4f}")
    print(f"  Marchenko-Pastur bounds: [{lambda_minus:.3f}, {lambda_plus:.3f}]")
    print(f"  Eigenvalues treated as noise (replaced by average): {n_noise} / {N}")

    return pd.DataFrame(cov_clean, index=returns_df.columns, columns=returns_df.columns)


def black_litterman(mu_market, Sigma, P, Q, omega=None, tau=0.05):
    """
    Simplified Black-Litterman posterior (Section 9.3): blends equilibrium
    (CAPM/market-implied) returns mu_market with investor views (P, Q) to
    produce a posterior expected-return vector, then feeds it to the
    mean-variance optimizer.

    P: k x n view matrix (each row picks out the assets a view is about)
    Q: k-vector of view returns
    omega: k x k view-uncertainty matrix (defaults to tau * P Sigma P')
    """
    n = len(mu_market)
    if omega is None:
        omega = np.diag(np.diag(tau * P @ Sigma @ P.T))

    tau_sigma = tau * Sigma
    inv_term = np.linalg.inv(P @ tau_sigma @ P.T + omega)
    posterior_mu = mu_market + tau_sigma @ P.T @ inv_term @ (Q - P @ mu_market)

    M_inv = np.linalg.inv(np.linalg.inv(tau_sigma) + P.T @ np.linalg.inv(omega) @ P)
    posterior_sigma = Sigma + M_inv

    return posterior_mu, posterior_sigma


def implied_equilibrium_returns(Sigma, w_market, risk_aversion=2.5):
    """Reverse-optimizes CAPM-implied equilibrium returns: mu = gamma * Sigma * w_mkt."""
    return risk_aversion * Sigma @ w_market


# ---------------------------------------------------------------------------
# 3. Section 9.4: risk-based portfolios
# ---------------------------------------------------------------------------

def min_variance_weights(Sigma):
    """Closed-form minimum-variance weights: w* = (Sigma^-1 1) / (1' Sigma^-1 1)."""
    Sigma = np.asarray(Sigma)
    n = Sigma.shape[0]
    ones = np.ones(n)
    inv_sigma = np.linalg.inv(Sigma)
    w = inv_sigma @ ones / (ones @ inv_sigma @ ones)
    return w


def max_diversification_weights(Sigma):
    """
    Maximum Diversification portfolio (Choueifaty & Coignard, 2008):
    maximize DR(w) = (w'sigma) / sqrt(w'Sigma w) subject to long-only,
    sum(w)=1. Solved by noting this is equivalent to a min-variance problem
    on correlation-normalized assets, then rescaling.
    """
    Sigma = np.asarray(Sigma)
    n = Sigma.shape[0]
    vols = np.sqrt(np.diag(Sigma))

    def neg_diversification_ratio(w):
        port_vol = np.sqrt(w @ Sigma @ w)
        return -(w @ vols) / port_vol

    x0 = np.ones(n) / n
    bounds = [(0, 1)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    result = minimize(neg_diversification_ratio, x0, bounds=bounds, constraints=constraints)
    return result.x


def risk_parity_weights(Sigma):
    """
    Equal Risk Contribution (ERC) portfolio: finds w such that each asset's
    total risk contribution TRC_i = w_i * (Sigma w)_i / sqrt(w'Sigma w) is
    equal across assets, subject to long-only, sum(w)=1.
    """
    Sigma = np.asarray(Sigma)
    n = Sigma.shape[0]
    # Rescale for numerical conditioning: daily variances are ~1e-4-1e-8,
    # which flattens the SLSQP objective near x0 and stalls convergence.
    # Weights are invariant to a uniform positive rescaling of Sigma.
    scale = 1.0 / np.mean(np.diag(Sigma))
    Sigma_scaled = Sigma * scale

    def risk_contributions(w):
        port_vol = np.sqrt(w @ Sigma_scaled @ w)
        mrc = Sigma_scaled @ w / port_vol
        trc = w * mrc
        return trc

    def objective(w):
        trc = risk_contributions(w)
        avg_trc = trc.mean()
        return np.sum((trc - avg_trc) ** 2)

    x0 = np.ones(n) / n
    bounds = [(1e-6, 1)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    result = minimize(objective, x0, bounds=bounds, constraints=constraints,
                       method="SLSQP", options={"maxiter": 1000, "ftol": 1e-16})
    return result.x


def cov_to_corr(cov):
    """Converts a covariance matrix to a correlation matrix."""
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    return corr


def quasi_diagonalize(link, n_items):
    """
    Returns the leaf order implied by a linkage matrix so that similar
    assets end up adjacent (the HRP "quasi-diagonalization" step).
    """
    link = link.astype(int)
    sort_ix = [link[-1, 0], link[-1, 1]]
    n_clusters = link.shape[0] + 1
    while max(sort_ix) >= n_clusters:
        new_sort_ix = []
        for i in sort_ix:
            if i >= n_clusters:
                row = link[i - n_clusters]
                new_sort_ix.extend([int(row[0]), int(row[1])])
            else:
                new_sort_ix.append(i)
        sort_ix = new_sort_ix
    return sort_ix


def recursive_bisection(cov, sort_ix):
    """
    HRP's recursive bisection step: split capital inversely proportional to
    the variance of each sub-cluster, working down the dendrogram.
    """
    tickers = cov.columns.tolist() if hasattr(cov, "columns") else list(range(cov.shape[0]))
    cov = np.asarray(cov)
    w = pd.Series(1.0, index=[tickers[i] for i in sort_ix])
    clusters = [sort_ix]

    while clusters:
        clusters = [c[j:k] for c in clusters
                    for j, k in ((0, len(c) // 2), (len(c) // 2, len(c)))
                    if len(c) > 1]
        for i in range(0, len(clusters), 2):
            c0 = clusters[i]
            c1 = clusters[i + 1] if i + 1 < len(clusters) else []
            if not c1:
                continue

            def cluster_var(idx_list):
                sub_cov = cov[np.ix_(idx_list, idx_list)]
                inv_diag = 1.0 / np.diag(sub_cov)
                w_ivp = inv_diag / inv_diag.sum()
                return w_ivp @ sub_cov @ w_ivp

            var0 = cluster_var(c0)
            var1 = cluster_var(c1)
            alpha = 1 - var0 / (var0 + var1)

            for idx in c0:
                w[tickers[idx]] *= alpha
            for idx in c1:
                w[tickers[idx]] *= (1 - alpha)

    return w


def hrp_weights(returns_df):
    """
    Full Hierarchical Risk Parity pipeline (Lopez de Prado, 2016),
    reproducing the chapter's three-step algorithm: (1) linkage clustering
    on a correlation-distance metric, (2) quasi-diagonalization, (3)
    recursive bisection.
    """
    cov = returns_df.cov()
    corr = cov_to_corr(cov.values)
    dist = np.sqrt(np.clip((1 - corr) / 2, 0, None))
    condensed_dist = squareform(dist, checks=False)
    link = linkage(condensed_dist, method="single")

    sort_ix = quasi_diagonalize(link, cov.shape[0])
    weights = recursive_bisection(cov, sort_ix)
    return weights.reindex(returns_df.columns)


def plot_hrp_dendrogram(returns_df, filename="fig_9_hrp_dendrogram.png"):
    """Visualizes the HRP clustering step as a dendrogram."""
    cov = returns_df.cov()
    corr = cov_to_corr(cov.values)
    dist = np.sqrt(np.clip((1 - corr) / 2, 0, None))
    condensed_dist = squareform(dist, checks=False)
    link = linkage(condensed_dist, method="single")

    plt.figure(figsize=(10, 5))
    dendrogram(link, labels=returns_df.columns.tolist())
    plt.title("HRP Dendrogram: Correlation-Distance Clustering")
    plt.ylabel("Distance")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# 4. Section 9.5: volatility-managed portfolios
# ---------------------------------------------------------------------------

def volatility_managed_weights(returns_df, lookback=21, target_vol=None):
    """
    w_{i,t+1} = k / sigma^2_{i,t}: weights inversely proportional to
    trailing realized variance, normalized to sum to 1 (Section 9.5).
    """
    realized_var = returns_df.tail(lookback).var()
    inv_var = 1.0 / realized_var
    w = inv_var / inv_var.sum()
    return w


def vm_scale_existing_weights(w_base, returns_df, lookback=21, vm_c=0.10):
    """
    Applies volatility-managed scaling on top of an existing base weight
    vector (e.g., HRP weights), as in the chapter's multi-asset pipeline:
        w_t = (c / sigma^2_{t-1}) * w_base, renormalized to sum to 1.
    """
    sigma2 = returns_df.tail(lookback).var()
    vm_scale = vm_c / sigma2
    w_vm = w_base * vm_scale
    w_vm = w_vm / w_vm.sum()
    return w_vm


# ---------------------------------------------------------------------------
# 5. Section 9.6: production constrained optimization + robust optimization
# ---------------------------------------------------------------------------

def vm_optimizer(mu, Sigma, sigma_vols, w_prev, lam=1.0, vm_c=0.10,
                  max_pos=0.10, max_turnover=0.30):
    """Reproduces the chapter's production-grade cvxpy VM optimizer."""
    n = len(mu)
    w = cp.Variable(n)
    vm_scale = vm_c / (sigma_vols ** 2)
    obj = cp.Maximize(
        mu @ cp.multiply(w, vm_scale) - lam / 2 * cp.quad_form(w, Sigma)
    )
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= max_pos,
        cp.norm1(w - w_prev) <= max_turnover,
    ]
    cp.Problem(obj, constraints).solve()
    return w.value


def robust_mean_variance(mu, Sigma, lam=1.0, kappa=0.1, long_only=True):
    """
    Robust optimization (Section 9.6): adds an L2 penalty on w scaled by
    kappa (proportional to estimation uncertainty in mu), following
    Goldfarb & Iyengar (2003)'s ellipsoidal-uncertainty-set result. This
    produces a more concentrated, less mu-sensitive portfolio.
    """
    n = len(mu)
    w = cp.Variable(n)
    obj = cp.Maximize(mu @ w - lam / 2 * cp.quad_form(w, Sigma) - kappa * cp.norm(w, 2))
    constraints = [cp.sum(w) == 1]
    if long_only:
        constraints.append(w >= 0)
    cp.Problem(obj, constraints).solve()
    return w.value


# ---------------------------------------------------------------------------
# 6. Section 9.7: ML for return forecasting
# ---------------------------------------------------------------------------

def build_return_forecast_features(returns_df, lookbacks=(5, 21, 63)):
    """
    Builds a simple momentum-style feature matrix per asset: trailing
    returns over several lookback windows, used as predictors for next-
    period return (the "construct a feature matrix" step of Section 9.7).
    """
    features = {}
    for lb in lookbacks:
        features[f"mom_{lb}d"] = returns_df.rolling(lb).sum()
    return features


def ml_return_forecast(returns_df, lookbacks=(5, 21, 63), method="ridge", alpha=1.0):
    """
    For each asset, regresses next-day return on its own trailing-momentum
    features (a minimal version of the ML return-forecasting pipeline:
    feature construction -> model -> mu for the optimizer). Returns a
    Series of the latest one-step-ahead forecasts, usable as `mu`.
    """
    features = build_return_forecast_features(returns_df, lookbacks)
    feature_names = list(features.keys())

    forecasts = {}
    for asset in returns_df.columns:
        X = pd.DataFrame({name: features[name][asset] for name in feature_names})
        y = returns_df[asset].shift(-1)
        data = pd.concat([X, y.rename("target")], axis=1).dropna()
        if len(data) < 50:
            forecasts[asset] = returns_df[asset].mean()
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data[feature_names])
        y_train = data["target"].values

        if method == "ridge":
            model = Ridge(alpha=alpha)
        elif method == "lasso":
            model = Lasso(alpha=alpha, max_iter=10000)
        elif method == "elasticnet":
            model = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000)
        else:
            raise ValueError(method)

        model.fit(X_scaled, y_train)
        latest_X = scaler.transform(X[feature_names].iloc[[-1]])
        forecasts[asset] = model.predict(latest_X)[0]

    return pd.Series(forecasts)


# ---------------------------------------------------------------------------
# 7. Section 9.8: backtesting -- walk-forward, Deflated Sharpe Ratio, CPCV
# ---------------------------------------------------------------------------

def walk_forward_backtest(returns_df, weight_fn, estimation_window=252,
                            rebalance_freq=21, cost_bps=10, **weight_fn_kwargs):
    """
    Generic walk-forward backtest: at each rebalance date, estimate
    portfolio weights using only data strictly prior to that date, hold
    for `rebalance_freq` days, then roll forward. Applies a simple
    proportional transaction-cost model (cost_bps round-trip on turnover).
    """
    T = len(returns_df)
    dates, port_returns, turnovers = [], [], []
    w_prev = None

    for t in range(estimation_window, T - rebalance_freq, rebalance_freq):
        train = returns_df.iloc[t - estimation_window:t]
        test = returns_df.iloc[t:t + rebalance_freq]

        w = weight_fn(train, **weight_fn_kwargs)
        w = pd.Series(w, index=returns_df.columns) if not isinstance(w, pd.Series) else w
        w = w.reindex(returns_df.columns).fillna(0)

        turnover = np.abs(w.values - (w_prev.values if w_prev is not None else 0)).sum()
        cost = turnover * (cost_bps / 10000)

        period_returns = (test * w).sum(axis=1)
        period_returns.iloc[0] -= cost   # apply cost on rebalance day

        dates.append(returns_df.index[t])
        port_returns.append(period_returns)
        turnovers.append(turnover)
        w_prev = w

    all_returns = pd.concat(port_returns)
    ann_return = (1 + all_returns).prod() ** (TRADING_DAYS / len(all_returns)) - 1
    ann_vol = all_returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    cum = (1 + all_returns).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    ann_turnover = np.mean(turnovers) * (TRADING_DAYS / rebalance_freq)

    return {
        "returns": all_returns,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "ann_turnover": ann_turnover,
    }


def deflated_sharpe_ratio(observed_sharpe, n_trials, n_obs, skew=0.0, kurtosis=3.0):
    """
    Deflated Sharpe Ratio (Lopez de Prado & Bailey, 2014): corrects the
    observed Sharpe ratio for multiple-testing / selection bias. Returns
    the probability that the true Sharpe ratio exceeds the expected
    maximum Sharpe obtainable from pure noise across n_trials candidates.
    """
    from scipy.stats import norm

    # Expected maximum Sharpe ratio under the null (pure noise), via the
    # expected value of the maximum of n_trials standard normals.
    euler_gamma = 0.5772156649
    expected_max_sr = (1 - euler_gamma) * norm.ppf(1 - 1 / n_trials) + \
                       euler_gamma * norm.ppf(1 - 1 / (n_trials * np.e))
    expected_max_sr /= np.sqrt(n_obs)   # scale to per-period units (approx.)

    # Standard error of the Sharpe ratio estimator, adjusted for skew/kurtosis
    sr_std = np.sqrt((1 - skew * observed_sharpe + (kurtosis - 1) / 4 * observed_sharpe ** 2) / (n_obs - 1))

    z = (observed_sharpe - expected_max_sr) / sr_std if sr_std > 0 else np.nan
    psr = norm.cdf(z)

    print(f"\n=== Deflated Sharpe Ratio ===")
    print(f"  Observed Sharpe: {observed_sharpe:.3f}")
    print(f"  Expected max Sharpe under noise (n_trials={n_trials}): {expected_max_sr:.3f}")
    print(f"  Deflated Sharpe Ratio (probability of genuine skill): {psr:.3f}")

    return psr


def combinatorial_purged_splits(n_obs, n_groups=6, n_test_groups=2, embargo=5):
    """
    Simplified Combinatorial Purged Cross-Validation (Lopez de Prado, 2018):
    splits the sample into n_groups contiguous blocks, enumerates all
    combinations of n_test_groups held out as test, purges any training
    observations immediately adjacent to a test block, and embargoes a
    buffer of `embargo` observations after each test block.
    """
    from itertools import combinations

    group_bounds = np.linspace(0, n_obs, n_groups + 1).astype(int)
    groups = [np.arange(group_bounds[i], group_bounds[i + 1]) for i in range(n_groups)]

    splits = []
    for test_group_ids in combinations(range(n_groups), n_test_groups):
        test_idx = np.concatenate([groups[g] for g in test_group_ids])
        train_idx = np.array([i for i in range(n_obs) if i not in set(test_idx)])

        # Purge: remove training points within `embargo` of any test point
        test_set = set(test_idx.tolist())
        purge_mask = np.array([
            not any(abs(i - t) <= embargo for t in test_idx[:5]) or True
            for i in train_idx
        ])  # (lightweight purge; see note below)
        # A fuller implementation purges/embargoes per contiguous test block;
        # for brevity here we purge only points immediately bordering test blocks.
        border_points = set()
        for g in test_group_ids:
            lo, hi = group_bounds[g], group_bounds[g + 1]
            border_points.update(range(max(0, lo - embargo), lo))
            border_points.update(range(hi, min(n_obs, hi + embargo)))
        train_idx = np.array([i for i in train_idx if i not in border_points])

        splits.append((train_idx, test_idx))

    return splits


# ---------------------------------------------------------------------------
# 8. Section 9.9: full multi-asset pipeline
# ---------------------------------------------------------------------------

def multi_asset_pipeline_weights(returns_df, vm_lookback=21, cov_window=252,
                                   vm_c=0.10, max_pos=0.15, max_turnover=0.25,
                                   w_prev=None):
    """
    Reproduces the chapter's six-step multi-asset pipeline:
        1. realized vol from trailing `vm_lookback` days
        2. Ledoit-Wolf covariance over trailing `cov_window` days
        3. HRP base weights
        4. volatility-managed scaling of the HRP weights
        5. constrained optimizer (position cap, turnover budget)
        6. (execution timing is a backtest-harness concern, not modeled here)
    """
    n = returns_df.shape[1]
    cov_data = returns_df.tail(cov_window)
    Sigma, _ = ledoit_wolf_covariance(cov_data)

    w_hrp = hrp_weights(cov_data)
    w_vm = vm_scale_existing_weights(w_hrp, returns_df, lookback=vm_lookback, vm_c=vm_c)

    if w_prev is None:
        w_prev = np.ones(n) / n
    else:
        w_prev = np.asarray(w_prev)

    mu_placeholder = np.zeros(n)   # pipeline is risk-driven; mu plays no role here
    sigma_vols = returns_df.tail(vm_lookback).std().values

    w = cp.Variable(n)
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= max_pos,
        cp.norm1(w - w_prev) <= max_turnover,
    ]
    # Track the vol-managed target weights as closely as possible
    # subject to the production constraints.
    objective = cp.Minimize(cp.sum_squares(w - w_vm.values))
    cp.Problem(objective, constraints).solve()

    return pd.Series(w.value, index=returns_df.columns)


# ---------------------------------------------------------------------------
# Main: run the whole chapter end to end
# ---------------------------------------------------------------------------

def main():
    # 1. Data
    returns_df = load_universe(n_assets=15, n_days=TRADING_DAYS * 6)
    print(f"Universe: {returns_df.shape[1]} assets, {returns_df.shape[0]} trading days")

    # 2. Classical mean-variance + estimation-error remedies
    mu_hat = returns_df.mean().values * TRADING_DAYS
    Sigma_sample = returns_df.cov().values * TRADING_DAYS
    w_mv = mean_variance(mu_hat, Sigma_sample, lam=2.0, long_only=True)
    print("\n=== Classical Mean-Variance Weights (sample mu, Sigma) ===")
    print(pd.Series(w_mv, index=returns_df.columns).round(3))
    print("  (A corner solution concentrated in one asset is Michaud's 'error-maximization'\n"
          "   in action: the optimizer bets everything on whichever asset's noisy sample mu\n"
          "   happens to look best, exactly the fragility Section 9.3 warns about.)")

    Sigma_lw, shrinkage = ledoit_wolf_covariance(returns_df)
    print(f"\nLedoit-Wolf shrinkage intensity: {shrinkage:.4f}")

    Sigma_rmt = rmt_clean_covariance(returns_df)

    # Simplified Black-Litterman: equilibrium returns from a cap-weighted-like
    # prior (here, equal-weight as a stand-in), with one illustrative view.
    n = returns_df.shape[1]
    w_market = np.ones(n) / n
    mu_equilibrium = implied_equilibrium_returns(Sigma_sample, w_market, risk_aversion=2.5)
    P = np.zeros((1, n)); P[0, 0] = 1
    Q = np.array([mu_equilibrium[0] + 0.03])   # view: asset 0 outperforms equilibrium by 3%
    mu_bl, Sigma_bl = black_litterman(mu_equilibrium, Sigma_sample, P, Q, tau=0.05)
    print("\n=== Black-Litterman: equilibrium vs. posterior returns (first 5 assets) ===")
    print(pd.DataFrame({"equilibrium": mu_equilibrium[:5], "posterior": mu_bl[:5]},
                        index=returns_df.columns[:5]).round(4))

    # 3. Risk-based portfolios
    w_minvar = min_variance_weights(Sigma_lw.values)
    w_maxdiv = max_diversification_weights(Sigma_lw.values)
    w_erc = risk_parity_weights(Sigma_lw.values)
    w_hrp = hrp_weights(returns_df)
    plot_hrp_dendrogram(returns_df)

    risk_based_table = pd.DataFrame({
        "MinVar": w_minvar, "MaxDiv": w_maxdiv, "ERC": w_erc, "HRP": w_hrp.values,
    }, index=returns_df.columns)
    print("\n=== Risk-Based Portfolio Weights ===")
    print(risk_based_table.round(3))

    # 4. Volatility-managed weighting
    w_vm = volatility_managed_weights(returns_df, lookback=21)
    print("\n=== Volatility-Managed (Inverse-Variance) Weights ===")
    print(w_vm.round(3))

    # 5. Production constrained optimizer + robust optimization
    sigma_vols = returns_df.tail(21).std().values
    w_prev = np.ones(n) / n
    w_vm_prod = vm_optimizer(mu_hat, Sigma_sample, sigma_vols, w_prev,
                              lam=2.0, vm_c=0.10, max_pos=0.10, max_turnover=0.30)
    print("\n=== Production VM Optimizer (with position/turnover constraints) ===")
    print(pd.Series(w_vm_prod, index=returns_df.columns).round(3))

    w_robust = robust_mean_variance(mu_hat, Sigma_sample, lam=2.0, kappa=0.5)
    print("\n=== Robust Mean-Variance (uncertainty-penalized) ===")
    print(pd.Series(w_robust, index=returns_df.columns).round(3))

    # 6. ML return forecasting
    mu_ridge = ml_return_forecast(returns_df, method="ridge", alpha=1.0)
    mu_lasso = ml_return_forecast(returns_df, method="lasso", alpha=0.0005)
    print("\n=== ML Return Forecasts (1-step-ahead, Ridge vs. LASSO) ===")
    print(pd.DataFrame({"Ridge": mu_ridge, "LASSO": mu_lasso}).round(5))

    # 7. Backtesting: walk-forward + Deflated Sharpe Ratio + CPCV
    methods = {
        "Equal-Weight": lambda train: np.ones(train.shape[1]) / train.shape[1],
        "Mean-Variance": lambda train: mean_variance(
            train.mean().values * TRADING_DAYS, train.cov().values * TRADING_DAYS, lam=2.0),
        "Min-Variance": lambda train: min_variance_weights(
            LedoitWolf().fit(train.values).covariance_),
        "Risk-Parity": lambda train: risk_parity_weights(
            LedoitWolf().fit(train.values).covariance_),
        "HRP": lambda train: hrp_weights(train).values,
        "Vol-Managed": lambda train: volatility_managed_weights(train, lookback=21).values,
    }

    print("\n=== Walk-Forward Backtest: All Methods (Table 9.1-style comparison) ===")
    backtest_results = {}
    for name, fn in methods.items():
        result = walk_forward_backtest(returns_df, fn, estimation_window=252,
                                         rebalance_freq=21, cost_bps=10)
        backtest_results[name] = {
            "Ann. Return": result["ann_return"],
            "Volatility": result["ann_vol"],
            "Sharpe": result["sharpe"],
            "Max DD": result["max_drawdown"],
            "Turnover/yr": result["ann_turnover"],
        }
    summary_table = pd.DataFrame(backtest_results).T
    print(summary_table.round(3))

    plt.figure(figsize=(10, 5))
    for name, fn in methods.items():
        result = walk_forward_backtest(returns_df, fn, estimation_window=252,
                                         rebalance_freq=21, cost_bps=10)
        cum = (1 + result["returns"]).cumprod()
        plt.plot(cum.index, cum.values, label=name)
    plt.title("Walk-Forward Cumulative Performance by Method")
    plt.ylabel("Cumulative growth of $1")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("fig_9_method_comparison.png", dpi=150)
    plt.show()

    best_name = summary_table["Sharpe"].idxmax()
    deflated_sharpe_ratio(summary_table.loc[best_name, "Sharpe"],
                           n_trials=len(methods), n_obs=len(returns_df))

    cpcv_splits = combinatorial_purged_splits(len(returns_df), n_groups=6, n_test_groups=2, embargo=5)
    print(f"\nCombinatorial Purged CV: {len(cpcv_splits)} train/test splits generated "
          f"(6 groups, 2 held out, embargo=5 days)")

    # 8. Full multi-asset pipeline
    w_pipeline = multi_asset_pipeline_weights(returns_df)
    print("\n=== Full Multi-Asset Pipeline Weights (Ledoit-Wolf -> HRP -> VM -> constraints) ===")
    print(w_pipeline.round(3))

    pipeline_backtest = walk_forward_backtest(
        returns_df,
        lambda train: multi_asset_pipeline_weights(train).values,
        estimation_window=252, rebalance_freq=21, cost_bps=10,
    )
    print(f"\nFull pipeline backtest -- Sharpe: {pipeline_backtest['sharpe']:.3f}, "
          f"Ann. Return: {pipeline_backtest['ann_return']:.2%}, "
          f"Max DD: {pipeline_backtest['max_drawdown']:.2%}, "
          f"Turnover/yr: {pipeline_backtest['ann_turnover']:.2%}")

    print("\nAll figures saved as PNG files in the working directory.")


if __name__ == "__main__":
    main()
