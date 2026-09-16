"""
Chapter 5 Companion Code
Risk-Return Tradeoffs and Efficient Frontier
Financial Engineering with AI, ML and Python -- Irene Aldridge

This script reproduces every quantitative idea covered in the chapter:

    1. Risk (standard deviation) and return (mean) of assets
    2. The risk-return / mean-stdev scatter plot (Figure 5.1-style)
    3. Return-per-unit-of-risk (Sharpe-style) asset selection
    4. The mean-variance frontier and the efficient frontier
    5. Covariance and correlation between two assets (MORN/BUD-style example)
    6. Two-asset portfolio construction (weights x1, x2)
    7. Markowitz mean-variance optimization for N assets
    8. The risk-free rate, the tangency ("Market") portfolio, and the
       Capital Market Line (CML)

Requirements: numpy, pandas, matplotlib, scipy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------

TICKERS = ["MORN", "BUD", "AAPL", "MSFT", "XOM", "GLD"]
START, END = "2015-01-01", "2025-01-01"
RISK_FREE_RATE_ANNUAL = 0.04   # ~ current T-Bill rate; annualized
TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_price_data():
    """
    Returns a DataFrame of daily prices, columns = tickers.
    """
    import yfinance as yf
    data = yf.download(TICKERS, start=START, end=END)["Adj Close"]
    return data.dropna(how="all")

# ---------------------------------------------------------------------------
# 2. Risk & return statistics
# ---------------------------------------------------------------------------

def compute_returns(prices):
    """Daily simple returns from a price DataFrame."""
    return prices.pct_change().dropna()


def risk_return_table(returns, annualize=True):
    """
    Table of E[R] and Stdev[R] per asset, following the chapter's notation:
        s = standard deviation of returns
        m = mean (average) of returns
    """
    m = returns.mean()
    s = returns.std()
    if annualize:
        m = m * TRADING_DAYS
        s = s * np.sqrt(TRADING_DAYS)
    table = pd.DataFrame({"E[R]": m, "Stdev": s})
    table["Return_per_unit_risk"] = table["E[R]"] / table["Stdev"]
    return table.sort_values("Return_per_unit_risk", ascending=False)


# ---------------------------------------------------------------------------
# 3. Figure 5.1-style scatter: the risk-return cloud
# ---------------------------------------------------------------------------

def plot_risk_return_scatter(table, title="Mean-Stdev Frontier"):
    """
    Reproduces the book's core scatter plot (s = stdev on the x-axis,
    m = mean return on the y-axis), labeling the assets on the frontier.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(table["Stdev"], table["E[R]"], color="blue")

    # Label every point (with a real universe you'd typically only label
    # the frontier names, as the book does for CRIS/DNTH/SYBX/etc.)
    for ticker, row in table.iterrows():
        plt.annotate(ticker, (row["Stdev"], row["E[R]"]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)

    plt.title(title)
    plt.xlabel("Stdev (risk)")
    plt.ylabel("E[R]")
    plt.axhline(0, color="grey", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("fig_5_1_risk_return_scatter.png", dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# 4. Return per unit of risk:  max_i E[R_i] / sigma[R_i]
# ---------------------------------------------------------------------------

def best_return_per_risk(table):
    """
    Implements the chapter's selection criterion:
        max_i E[R_i] / sigma[R_i]
    Returns the single best asset by this Sharpe-style ratio.
    """
    best_ticker = table["Return_per_unit_risk"].idxmax()
    return best_ticker, table.loc[best_ticker]


# ---------------------------------------------------------------------------
# 5. Covariance & correlation (MORN / BUD example)
# ---------------------------------------------------------------------------

def covariance_and_correlation(prices, returns, a="MORN", b="BUD"):
    """
    Demonstrates:
      - price-level correlation (can be negative, as in the book's example)
      - return-level covariance / correlation (the one that matters for
        portfolio construction)
      - the identity Cov(R1,R2) = sigma1 * sigma2 * rho12
    """
    price_corr = prices[[a, b]].corr().iloc[0, 1]

    cov_matrix = np.cov(returns[a], returns[b])
    cov_ab = cov_matrix[0, 1]

    corr_matrix = np.corrcoef(returns[a], returns[b])
    corr_ab = corr_matrix[0, 1]

    sigma_a = returns[a].std()
    sigma_b = returns[b].std()
    implied_cov = sigma_a * sigma_b * corr_ab  # should equal cov_ab

    print(f"\n--- Covariance / Correlation: {a} vs {b} ---")
    print(f"Correlation of PRICES:            {price_corr:.2%}")
    print(f"Correlation of DAILY RETURNS:     {corr_ab:.2%}")
    print(f"Covariance of DAILY RETURNS:      {cov_ab:.6f}")
    print(f"sigma_{a}*sigma_{b}*rho check:    {implied_cov:.6f}  (should match)")

    return {
        "price_corr": price_corr,
        "return_corr": corr_ab,
        "return_cov": cov_ab,
    }


def plot_price_and_return_series(prices, returns, a="MORN", b="BUD"):
    """Figures 5.7 / 5.9 / 5.10 style plots."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(prices.index, prices[a], label=a)
    axes[0].plot(prices.index, prices[b], label=b)
    axes[0].set_title(f"Prices of {a} and {b}")
    axes[0].legend()

    axes[1].scatter(returns[a], returns[b], s=8, alpha=0.5)
    axes[1].set_title(f"Returns of {a} and {b}")
    axes[1].set_xlabel(f"Returns {a}")
    axes[1].set_ylabel(f"Returns {b}")

    plt.tight_layout()
    plt.savefig("fig_5_7_5_10_prices_and_returns.png", dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# 6. Two-asset portfolio sweep (Figure 5.11-style)
# ---------------------------------------------------------------------------

def two_asset_portfolio_sweep(returns, a="MORN", b="BUD", n=101):
    """
    Sweeps x1 in [0, 1] (weight on asset a), x2 = 1 - x1, and computes
    portfolio mean, volatility and Sharpe ratio -- reproducing Figure 5.11.
    Also returns the volatility-minimizing weight in closed form.
    """
    mu_a, mu_b = returns[a].mean() * TRADING_DAYS, returns[b].mean() * TRADING_DAYS
    sig_a, sig_b = returns[a].std() * np.sqrt(TRADING_DAYS), returns[b].std() * np.sqrt(TRADING_DAYS)
    rho = returns[[a, b]].corr().iloc[0, 1]
    cov_ab = sig_a * sig_b * rho

    x1_grid = np.linspace(0, 1, n)
    means, vols, sharpes = [], [], []

    for x1 in x1_grid:
        x2 = 1 - x1
        port_mean = x1 * mu_a + x2 * mu_b
        port_var = (x1 ** 2 * sig_a ** 2 + x2 ** 2 * sig_b ** 2
                    + 2 * x1 * x2 * cov_ab)
        port_vol = np.sqrt(port_var)
        means.append(port_mean)
        vols.append(port_vol)
        sharpes.append(port_mean / port_vol if port_vol > 0 else np.nan)

    # Closed-form minimum-variance weight for two assets:
    #   x1* = (sig_b^2 - cov_ab) / (sig_a^2 + sig_b^2 - 2*cov_ab)
    x1_min_var = (sig_b ** 2 - cov_ab) / (sig_a ** 2 + sig_b ** 2 - 2 * cov_ab)
    x1_min_var = np.clip(x1_min_var, 0, 1)

    print(f"\nMinimum-variance weight: {a} = {x1_min_var:.1%}, "
          f"{b} = {1 - x1_min_var:.1%}")

    plt.figure(figsize=(8, 6))
    plt.plot(x1_grid, means, label="portfolio mean", linewidth=2, color="black")
    plt.plot(x1_grid, vols, "--", label="portfolio volatility")
    plt.plot(x1_grid, sharpes, label="portfolio Sharpe", color="tab:blue")
    plt.axvline(x1_min_var, color="grey", linestyle=":",
                label=f"min-variance x1={x1_min_var:.2f}")
    plt.title(f"Portfolio Characteristics of {a} and {b}")
    plt.xlabel("x1 (weight on " + a + ")")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_5_11_two_asset_sweep.png", dpi=150)
    plt.show()

    return pd.DataFrame({"x1": x1_grid, "mean": means, "vol": vols, "sharpe": sharpes}), x1_min_var


# ---------------------------------------------------------------------------
# 7. Markowitz mean-variance optimization for N assets
# ---------------------------------------------------------------------------

def portfolio_stats(weights, mu, cov):
    """Annualized portfolio return and volatility for a weight vector."""
    port_return = weights @ mu
    port_vol = np.sqrt(weights @ cov @ weights)
    return port_return, port_vol


def minimum_variance_portfolio(mu, cov, long_only=True):
    """
    Solves:  min_w  w' Cov w   s.t.  sum(w) = 1  (and w >= 0 if long_only)
    """
    n = len(mu)
    x0 = np.ones(n) / n
    bounds = [(0, 1)] * n if long_only else [(-1, 1)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    result = minimize(lambda w: w @ cov @ w, x0, bounds=bounds,
                       constraints=constraints)
    return result.x


def efficient_frontier(mu, cov, n_points=50, long_only=True):
    """
    Traces the efficient frontier by solving, for a grid of target returns:
        min_w  w' Cov w
        s.t.   w' mu = target_return
               sum(w) = 1
               (w >= 0 if long_only)
    Returns arrays of frontier (return, volatility, weights).
    """
    n = len(mu)
    bounds = [(0, 1)] * n if long_only else [(-1, 1)] * n
    target_returns = np.linspace(mu.min(), mu.max(), n_points)

    frontier_vol = []
    frontier_weights = []

    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, target=target: w @ mu - target},
        ]
        x0 = np.ones(n) / n
        result = minimize(lambda w: w @ cov @ w, x0, bounds=bounds,
                           constraints=constraints)
        if result.success:
            frontier_vol.append(np.sqrt(result.fun))
            frontier_weights.append(result.x)
        else:
            frontier_vol.append(np.nan)
            frontier_weights.append(np.full(n, np.nan))

    return pd.DataFrame({
        "target_return": target_returns,
        "volatility": frontier_vol,
    }), np.array(frontier_weights)


def max_sharpe_portfolio(mu, cov, rf, long_only=True):
    """
    Solves for the tangency ("Market") portfolio that maximizes the
    Sharpe ratio:  max_w (w'mu - rf) / sqrt(w'Cov w)   s.t. sum(w)=1
    """
    n = len(mu)
    bounds = [(0, 1)] * n if long_only else [(-1, 1)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    x0 = np.ones(n) / n

    def neg_sharpe(w):
        ret, vol = portfolio_stats(w, mu, cov)
        return -(ret - rf) / vol

    result = minimize(neg_sharpe, x0, bounds=bounds, constraints=constraints)
    return result.x


def tangency_portfolio_closed_form(mu, cov, rf):
    """
    Closed-form (unconstrained, allows shorting) tangency weights:
        w* = Cov^-1 (mu - rf) / [1' Cov^-1 (mu - rf)]
    Matches the book's code snippet exactly.
    """
    inv_cov = np.linalg.inv(cov)
    excess = mu - rf
    z = inv_cov @ excess
    w_tan = z / z.sum()
    return w_tan


# ---------------------------------------------------------------------------
# 8. Capital Market Line
# ---------------------------------------------------------------------------

def plot_efficient_frontier_with_cml(table, frontier_df, rf,
                                      tangency_return, tangency_vol):
    """Reproduces Figures 5.14 / 5.15: efficient frontier + CML."""
    plt.figure(figsize=(8, 6))
    plt.scatter(table["Stdev"], table["E[R]"], color="blue", s=15, label="assets")
    plt.plot(frontier_df["volatility"], frontier_df["target_return"],
              color="green", linewidth=2, label="efficient frontier")

    # Capital Market Line: from (0, rf) through the tangency point
    sharpe_tan = (tangency_return - rf) / tangency_vol
    cml_std = np.linspace(0, table["Stdev"].max() * 1.15, 100)
    cml_ret = rf + sharpe_tan * cml_std
    plt.plot(cml_std, cml_ret, color="grey", linewidth=2, label="Capital Market Line")

    plt.scatter([0], [rf], color="black", zorder=5)
    plt.annotate("Rf", (0, rf), textcoords="offset points", xytext=(8, -8))
    plt.scatter([tangency_vol], [tangency_return], color="orange", s=80, zorder=5)
    plt.annotate("M", (tangency_vol, tangency_return),
                 textcoords="offset points", xytext=(8, 8))

    plt.title("Mean-Stdev Frontier with Risk-Free Rate")
    plt.xlabel("Stdev (risk)")
    plt.ylabel("E[R]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_5_14_cml.png", dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Main: run the whole chapter end to end
# ---------------------------------------------------------------------------

def main():
    # 1. Load data & compute returns
    prices = load_price_data()
    returns = compute_returns(prices)

    # 2. Risk-return table (Table 5.1-style) and scatter (Figure 5.1)
    table = risk_return_table(returns)
    print("\n=== Risk-Return Table (annualized) ===")
    print(table.round(3))
    plot_risk_return_scatter(table)

    # 3. Return per unit of risk
    best_ticker, best_row = best_return_per_risk(table)
    print(f"\nBest return-per-unit-of-risk asset: {best_ticker}")
    print(best_row.round(3))

    # 4. Covariance / correlation, MORN vs BUD
    covariance_and_correlation(prices, returns, "MORN", "BUD")
    plot_price_and_return_series(prices, returns, "MORN", "BUD")

    # 5. Two-asset portfolio sweep (Figure 5.11)
    sweep_df, x1_min_var = two_asset_portfolio_sweep(returns, "MORN", "BUD")

    # 6. Markowitz N-asset efficient frontier
    mu = returns.mean() * TRADING_DAYS
    cov = returns.cov() * TRADING_DAYS

    min_var_w = minimum_variance_portfolio(mu.values, cov.values)
    min_var_ret, min_var_vol = portfolio_stats(min_var_w, mu.values, cov.values)
    print("\n=== Minimum-Variance Portfolio (long-only) ===")
    for t, w in zip(mu.index, min_var_w):
        if w > 1e-4:
            print(f"  {t}: {w:.1%}")
    print(f"  -> Return: {min_var_ret:.2%}, Volatility: {min_var_vol:.2%}")

    frontier_df, frontier_weights = efficient_frontier(mu.values, cov.values)

    # 7. Tangency ("Market") portfolio and Capital Market Line
    tangency_w = max_sharpe_portfolio(mu.values, cov.values, RISK_FREE_RATE_ANNUAL)
    tan_ret, tan_vol = portfolio_stats(tangency_w, mu.values, cov.values)
    print("\n=== Tangency / Market Portfolio (max Sharpe, long-only) ===")
    for t, w in zip(mu.index, tangency_w):
        if w > 1e-4:
            print(f"  {t}: {w:.1%}")
    print(f"  -> Return: {tan_ret:.2%}, Volatility: {tan_vol:.2%}, "
          f"Sharpe: {(tan_ret - RISK_FREE_RATE_ANNUAL) / tan_vol:.2f}")

    plot_efficient_frontier_with_cml(table, frontier_df, RISK_FREE_RATE_ANNUAL,
                                      tan_ret, tan_vol)


if __name__ == "__main__":
    main()
