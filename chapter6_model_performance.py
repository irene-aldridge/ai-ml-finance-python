"""
Chapter 6 Companion Code
Model Performance and Optimization
Financial Engineering with AI, ML and Python -- Irene Aldridge

This script reproduces every quantitative idea covered in
the chapter:

    1. Year-over-year instability of the mean-variance frontier (Figures 6.1-6.5)
    2. Type I / Type II error framework applied to a frontier-based forecasting rule (Table 6.1, precision & recall)
    3. ROC-AUC vs. Precision-Recall AUC on imbalanced data (Figure 6.6), plus the same comparison applied to the frontier-forecast problem
       (Figure 6.7)
    4. In-sample vs. out-of-sample evaluation: one-shot train/test split and the rolling-window estimator (Figures 6.8 / 6.9)
    5. Loss functions: raw error, MAE (L1), MSE/RMSE (L2), Huber, the general Wasserstein-style Lp loss, CDF-based W1 and the Kolmogorov-Smirnov statistic (Figures 6.10 / 6.11)
    6. Using a quadratic loss to derive a loss-minimizing, regression-based two-asset portfolio (the AAL/SPY example, Figures 6.12 / 6.13)
    7. A reusable metrics-comparison helper mirroring Table 6.3

Requirements: numpy, pandas, matplotlib, scipy, scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
)

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------

TICKERS_15 = ["A", "AA", "AAL", "AAP", "AAT", "AAPL", "AAON", "AAWW", "AAU", "AAME", "AAOI", "AAIC", "AADI", "AAMC", "AACG"]
YEARS = [2017, 2018, 2019, 2020, 2021]
TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# 1. Data: a 15-stock universe whose risk-return profile shifts year to year
# ---------------------------------------------------------------------------

def load_price_data():
    import yfinance as yf
    real_tickers = TICKERS_15
    prices = yf.download(real_tickers, start="2017-01-01", end="2022-01-01")["Adj Close"]
    returns_by_year = {}
    for year in YEARS:
        yr = prices.loc[str(year)]
        returns_by_year[year] = yr.pct_change().dropna()
    return returns_by_year

    
def load_two_asset_data():
    import yfinance as yf
    prices = yf.download(["AAL", "SPY"], start="2005-10-01", end="2025-10-01")["Adj Close"]
    return prices.pct_change().dropna()

# ---------------------------------------------------------------------------
# 2. Figures 6.1-6.5: the mean-variance frontier is NOT stable over time
# ---------------------------------------------------------------------------

def plot_yearly_frontiers(returns_by_year):
    """
    Plots one mean-variance scatter per year, annualized, replicating Figures 6.1 through 6.5. Also returns a DataFrame of annualized
    (mean, stdev) per stock per year for later use.
    """
    fig, axes = plt.subplots(1, len(YEARS), figsize=(4 * len(YEARS), 4), sharey=False)
    records = []

    for ax, year in zip(axes, YEARS):
        rets = returns_by_year[year]
        m = rets.mean() * TRADING_DAYS
        s = rets.std() * np.sqrt(TRADING_DAYS)
        ax.scatter(s, m, color="tab:blue")
        ax.set_title(f"Mean-Variance Frontier, {year}")
        ax.set_xlabel("Std Dev Returns")
        if year == YEARS[0]:
            ax.set_ylabel("Mean Returns")
        for t in rets.columns:
            records.append({"year": year, "ticker": t, "mean": m[t], "stdev": s[t]})

    plt.tight_layout()
    plt.savefig("fig_6_1_6_5_yearly_frontiers.png", dpi=150)
    plt.show()

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 3. Type I / Type II errors from a frontier-based forecasting rule
# ---------------------------------------------------------------------------

def frontier_forecast_confusion_matrix(stats_df, top_frac=0.3):
    """
    Defines "favorable" (predicted/realized profitable event) as being in
    the top `top_frac` of the universe by mean return in a given year
    (a simple proxy for "upper-left region of the frontier"), following the
    chapter's AAOI / AACG walkthrough. For each consecutive year pair
    (year_t -> year_t+1), a stock is:
        TP: favorable in year_t AND favorable in year_t+1
        FP (Type I): favorable in year_t, NOT favorable in year_t+1
        FN (Type II): NOT favorable in year_t, favorable in year_t+1
        TN: NOT favorable in either year

    Returns the pooled confusion matrix counts and precision/recall/accuracy.
    """
    tp = fp = fn = tn = 0
    n_favorable = max(1, int(round(top_frac * stats_df["ticker"].nunique())))

    for y0, y1 in zip(YEARS[:-1], YEARS[1:]):
        s0 = stats_df[stats_df.year == y0].set_index("ticker")["mean"]
        s1 = stats_df[stats_df.year == y1].set_index("ticker")["mean"]
        favorable_0 = set(s0.sort_values(ascending=False).head(n_favorable).index)
        favorable_1 = set(s1.sort_values(ascending=False).head(n_favorable).index)

        for ticker in s0.index:
            pred = ticker in favorable_0     # forecast: profitable event
            real = ticker in favorable_1     # realized: profitable event
            if pred and real:
                tp += 1
            elif pred and not real:
                fp += 1        # Type I error
            elif not pred and real:
                fn += 1        # Type II error
            else:
                tn += 1

    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan

    print("\n=== Frontier-Based Forecast: Confusion Matrix (pooled across years) ===")
    print(f"  TP={tp}  FP(Type I)={fp}  FN(Type II)={fn}  TN={tn}")
    print(f"  Accuracy:  {accuracy:.2%}")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall:    {recall:.2%}")

    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "accuracy": accuracy, "precision": precision, "recall": recall}


# ---------------------------------------------------------------------------
# 4. ROC-AUC vs. Precision-Recall AUC
# ---------------------------------------------------------------------------

def plot_roc_and_pr(y_true, y_scores, title="Model Evaluation", filename="roc_pr.png"):
    """
    Plots ROC and Precision-Recall curves side by side (reproduces the
    chapter's `plot_roc_and_pr` helper and Figure 6.6 / 6.7 style output).
    """
    prevalence = y_true.mean()
    roc_auc = roc_auc_score(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes[0].plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC-AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Random (0.500)")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    axes[1].plot(recall, precision, color="darkorange", lw=2, label=f"PR-AUC = {pr_auc:.3f}")
    axes[1].axhline(y=prevalence, color="k", linestyle="--", lw=1,
                     label=f"Random ({prevalence:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()

    print(f"\nClass prevalence: {prevalence:.1%}")
    print(f"ROC-AUC: {roc_auc:.3f} (random baseline: 0.500)")
    print(f"PR-AUC:  {pr_auc:.3f} (random baseline: {prevalence:.3f})")
    if prevalence > 0:
        print(f"PR-AUC lift: {pr_auc / prevalence:.1f}x over random")

    return roc_auc, pr_auc


def simulated_credit_default_example():
    """Reproduces the chapter's 3%-prevalence credit default example (Figure 6.6)."""
    np.random.seed(42)
    n = 10_000
    prevalence = 0.03
    y_true = np.random.binomial(1, prevalence, n)
    signal = np.where(
        y_true == 1,
        np.random.normal(0.65, 0.2, n),   # defaults score higher
        np.random.normal(0.35, 0.2, n),   # non-defaults score lower
    )
    y_scores = np.clip(signal, 0, 1)
    return plot_roc_and_pr(y_true, y_scores,
                            title="Credit Default Model (3% Prevalence)",
                            filename="fig_6_6_credit_default_roc_pr.png")


def frontier_forecast_roc_pr(stats_df, top_frac=0.3):
    """
    Builds a (y_true, y_scores) pair from the frontier-forecasting setup:
    y_true = 1 if the stock is favorable in year_t+1; y_scores = the stock's
    mean-return RANK in year_t (higher rank score = model more confident it
    stays favorable). Reproduces Figure 6.7's close-to-random performance.
    """
    y_true_all, y_scores_all = [], []
    for y0, y1 in zip(YEARS[:-1], YEARS[1:]):
        s0 = stats_df[stats_df.year == y0].set_index("ticker")["mean"]
        s1 = stats_df[stats_df.year == y1].set_index("ticker")["mean"]
        n_favorable = max(1, int(round(top_frac * len(s1))))
        favorable_1 = set(s1.sort_values(ascending=False).head(n_favorable).index)

        # Score = rank of previous year's mean return (higher = better)
        scores = s0.rank(pct=True)
        for ticker in s0.index:
            y_true_all.append(1 if ticker in favorable_1 else 0)
            y_scores_all.append(scores[ticker])

    y_true_all = np.array(y_true_all)
    y_scores_all = np.array(y_scores_all)
    return plot_roc_and_pr(y_true_all, y_scores_all,
                            title="Frontier-Based Stock Forecast (2017-2021)",
                            filename="fig_6_7_frontier_roc_pr.png")


# ---------------------------------------------------------------------------
# 5. In-sample / out-of-sample: one-shot split and rolling window
# ---------------------------------------------------------------------------

def one_shot_train_test_split(returns, test_size=0.33):
    """
    Time-series-aware one-shot split: training precedes test in time
    (unlike sklearn's default random `train_test_split`, which the chapter
    explicitly warns against for serially dependent data).
    """
    n = len(returns)
    split_idx = int(round(n * (1 - test_size)))
    train, test = returns.iloc[:split_idx], returns.iloc[split_idx:]
    print(f"\nOne-shot split: {len(train)} train obs, {len(test)} test obs "
          f"({train.index[0].date()} - {train.index[-1].date()} train / "
          f"{test.index[0].date()} - {test.index[-1].date()} test)")
    return train, test


def rolling_window_backtest(returns, ticker_x, ticker_y,
                             estimation_window=252, prediction_window=252):
    """
    Implements the rolling-window process of Figure 6.9:
      1. Estimate a simple linear relationship (beta) of ticker_y on
         ticker_x over the estimation window.
      2. Apply that beta out-of-sample over the following prediction window.
      3. Record out-of-sample R^2 / correlation as the performance metric.
      4. Roll the window forward by the prediction window length and repeat.

    Returns a DataFrame of one row per rolling fold.
    """
    x = returns[ticker_x].values
    y = returns[ticker_y].values
    n = len(returns)
    results = []

    start = 0
    while start + estimation_window + prediction_window <= n:
        is_slice = slice(start, start + estimation_window)
        oos_slice = slice(start + estimation_window,
                           start + estimation_window + prediction_window)

        x_is, y_is = x[is_slice], y[is_slice]
        beta, alpha = np.polyfit(x_is, y_is, 1)

        x_oos, y_oos = x[oos_slice], y[oos_slice]
        y_pred_oos = alpha + beta * x_oos
        oos_mse = np.mean((y_oos - y_pred_oos) ** 2)
        oos_corr = np.corrcoef(y_oos, y_pred_oos)[0, 1] if len(y_oos) > 1 else np.nan

        results.append({
            "start": returns.index[start],
            "is_end": returns.index[start + estimation_window - 1],
            "oos_end": returns.index[start + estimation_window + prediction_window - 1],
            "beta": beta,
            "oos_mse": oos_mse,
            "oos_corr": oos_corr,
        })
        start += prediction_window

    results_df = pd.DataFrame(results)
    print(f"\n=== Rolling-window backtest: {ticker_y} ~ {ticker_x} "
          f"({estimation_window}d estimate / {prediction_window}d test) ===")
    display_df = results_df[["is_end", "oos_end", "beta", "oos_mse", "oos_corr"]].copy()
    display_df[["beta", "oos_mse", "oos_corr"]] = display_df[["beta", "oos_mse", "oos_corr"]].round(4)
    print(display_df.to_string(index=False))
    return results_df


# ---------------------------------------------------------------------------
# 6. Loss functions
# ---------------------------------------------------------------------------

def loss_raw(y_realized, y_predicted):
    """Loss = y - y_hat (simple, but positive/negative errors can cancel)."""
    return y_realized - y_predicted


def loss_mae(y_realized, y_predicted):
    """Mean Absolute Error (L1 / LASSO loss)."""
    return np.mean(np.abs(y_realized - y_predicted))


def loss_mse(y_realized, y_predicted):
    """Mean Squared Error (quadratic / L2 / Ridge loss)."""
    return np.mean((y_realized - y_predicted) ** 2)


def loss_rmse(y_realized, y_predicted):
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_realized - y_predicted) ** 2))


def loss_huber(y_realized, y_predicted, delta=1.0):
    """Huber loss: quadratic for small errors, linear for large errors."""
    err = y_realized - y_predicted
    abs_err = np.abs(err)
    quadratic = np.minimum(abs_err, delta)
    linear = abs_err - quadratic
    return np.mean(0.5 * quadratic ** 2 + delta * linear)


def loss_lp(y_realized, y_predicted, p):
    """Generalized Lp loss (the chapter's 'Wasserstein-style' Lp measure)."""
    return (np.mean(np.abs(y_realized - y_predicted) ** p)) ** (1 / p)


def empirical_cdf(data):
    """Returns (sorted_values, cdf) for the empirical CDF of `data`."""
    sorted_data = np.sort(data)
    n = len(sorted_data)
    cdf_values = np.arange(1, n + 1) / n
    return sorted_data, cdf_values


def plot_cdf_vs_normal(returns_series, title, filename):
    """Reproduces Figures 6.10 / 6.11: empirical CDF vs. a fitted Normal CDF."""
    data = returns_series.values
    sorted_data, cdf_values = empirical_cdf(data)

    mean, std_dev = np.mean(data), np.std(data)
    theoretical_cdf_values = norm.cdf(sorted_data, loc=mean, scale=std_dev)

    plt.figure(figsize=(7, 5))
    plt.plot(sorted_data, cdf_values, label="Empirical Return CDF")
    plt.plot(sorted_data, theoretical_cdf_values, label="Normal CDF")
    plt.title(title)
    plt.xlabel("Returns")
    plt.ylabel("Cumulative Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()

    diff = cdf_values - theoretical_cdf_values
    w1 = np.mean(np.abs(diff))
    ks = np.max(np.abs(diff))
    print(f"\n{title}")
    print(f"  W1 (mean |CDF diff|) = {w1:.4f}")
    print(f"  KS (max  |CDF diff|) = {ks:.4f}")
    return w1, ks


def demo_loss_functions():
    """Small worked example comparing all point-forecast loss functions."""
    np.random.seed(0)
    y_realized = np.random.normal(0.0005, 0.01, 500)
    y_predicted = y_realized + np.random.normal(0, 0.004, 500)  # noisy forecast

    print("\n=== Loss Function Comparison (synthetic forecast vs. realized) ===")
    print(f"  Mean raw error (bias): {np.mean(loss_raw(y_realized, y_predicted)):.6f}")
    print(f"  MAE  (L1):             {loss_mae(y_realized, y_predicted):.6f}")
    print(f"  MSE  (L2):             {loss_mse(y_realized, y_predicted):.6f}")
    print(f"  RMSE:                  {loss_rmse(y_realized, y_predicted):.6f}")
    print(f"  Huber (delta=1):       {loss_huber(y_realized, y_predicted):.6f}")
    print(f"  L3 (p=3):              {loss_lp(y_realized, y_predicted, 3):.6f}")


# ---------------------------------------------------------------------------
# 7. Loss-minimizing two-asset portfolio (AAL/SPY-style example)
# ---------------------------------------------------------------------------

def loss_minimizing_portfolio(df, ticker1="AAL", ticker2="SPY"):
    """
    Reproduces the chapter's error-minimizing allocation logic:
        w1 / w2 = -cov(R1, R2) / sigma1^2
    computed once per month using that month's realized covariance, then
    applied cumulatively. This is the "regression of one asset's return on
    the other's" interpretation the chapter derives from first principles.
    """
    def cum_ret(series):
        return (series + 1).cumprod().iloc[-1] - 1

    w1 = w2 = 0.0
    perf, aal_cum, spy_cum, rets, dates, w1s, w2s = [], [], [], [], [], [], []

    def compound(hist, r):
        return (hist[-1] + 1) * (1 + r) - 1 if hist else r

    for month, df1 in df.groupby("month"):
        if w1 or w2:
            r1, r2 = cum_ret(df1[ticker1]), cum_ret(df1[ticker2])
            yr_ret = w1 * r1 + w2 * r2
            perf.append(compound(perf, yr_ret))
            aal_cum.append(compound(aal_cum, r1))
            spy_cum.append(compound(spy_cum, r2))
            rets.append(yr_ret)
            dates.append(month)

        cov = np.cov(df1[ticker1], df1[ticker2])
        # w1/w2 = -cov(R1,R2) / sigma1^2  =>  solve for normalized weights
        beta1 = -cov[0, 1] / cov[0, 0]
        w2 = 1 / (1 + beta1) if (1 + beta1) != 0 else 0.5
        w1 = 1 - w2
        w1s.append(w1)
        w2s.append(w2)

    rets = np.array(rets)
    ann_return = np.mean(rets) * 12
    sharpe = np.mean(rets) / np.std(rets) * np.sqrt(12) if np.std(rets) > 0 else np.nan

    print(f"\n=== Loss-Minimizing {ticker1}/{ticker2} Portfolio ===")
    print(f"  Ann. Return : {ann_return:.2%}")
    print(f"  Sharpe Ratio: {sharpe:.2f}")

    fig, ax = plt.subplots(figsize=(10, 4))
    xticks = np.arange(0, len(dates) - 1, max(1, len(dates) // 20))
    ax.plot(perf, label="Portfolio")
    ax.set_xticks(xticks)
    ax.set_xticklabels(np.array(dates)[xticks], rotation=90)
    ax.set_title(f"Performance -- {ticker1} & {ticker2}")
    ax.legend()
    ax.grid(alpha=0.4)
    fig.tight_layout()
    fig.savefig(f"fig_6_12_performance_{ticker1}_{ticker2}.png", dpi=150)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(w1s, label=ticker1)
    ax.plot(w2s, label=ticker2)
    ax.set_xticks(xticks)
    ax.set_xticklabels(np.array(dates)[xticks], rotation=90)
    ax.set_title(f"Weights -- {ticker1} & {ticker2}")
    ax.legend()
    ax.grid(alpha=0.4)
    fig.tight_layout()
    fig.savefig(f"fig_6_13_weights_{ticker1}_{ticker2}.png", dpi=150)
    plt.show()

    return {"ann_return": ann_return, "sharpe": sharpe,
            "weights": pd.DataFrame({"date": dates, "w1": w1s[1:], "w2": w2s[1:]})}


# ---------------------------------------------------------------------------
# 8. Model performance metrics comparison table (Table 6.3-style helper)
# ---------------------------------------------------------------------------

METRICS_TABLE = pd.DataFrame([
    {"Metric": "Accuracy", "Type": "Classification",
     "Use when": "Balanced classes; symmetric error costs",
     "Avoid when": "Rare events / imbalanced classes"},
    {"Metric": "Precision", "Type": "Classification",
     "Use when": "False positives are expensive",
     "Avoid when": "Missing real events is costlier than false alarms"},
    {"Metric": "Recall", "Type": "Classification",
     "Use when": "False negatives are expensive",
     "Avoid when": "False alarms carry high operational cost"},
    {"Metric": "F1 score", "Type": "Classification",
     "Use when": "Single metric, no strong cost asymmetry",
     "Avoid when": "FP/FN costs are clearly asymmetric (use F-beta)"},
    {"Metric": "ROC-AUC", "Type": "Classification",
     "Use when": "Balanced classes; threshold-free comparison",
     "Avoid when": "Rare positive class (<10-15%)"},
    {"Metric": "PR-AUC", "Type": "Classification",
     "Use when": "Rare positive class; alert usefulness matters",
     "Avoid when": "Balanced classes (ROC-AUC is simpler)"},
    {"Metric": "MAE (L1)", "Type": "Regression",
     "Use when": "Outliers common; linear error penalty desired",
     "Avoid when": "Gradient-based optimization is required"},
    {"Metric": "MSE / RMSE (L2)", "Type": "Regression",
     "Use when": "Large errors disproportionately costly",
     "Avoid when": "Heavy-tailed data / outliers dominate"},
    {"Metric": "Huber loss", "Type": "Regression",
     "Use when": "Need outlier robustness + differentiability",
     "Avoid when": "Pure L1 or L2 already fits well"},
    {"Metric": "Wasserstein W1", "Type": "Distributional",
     "Use when": "Full distributional fit / tails matter",
     "Avoid when": "Only point-forecast accuracy matters"},
    {"Metric": "KS statistic", "Type": "Distributional",
     "Use when": "Worst-case distributional deviation matters",
     "Avoid when": "Average distributional fit is the goal"},
    {"Metric": "Quadratic loss (optimization)", "Type": "Optimization",
     "Use when": "Deriving closed-form weights / OLS regression",
     "Avoid when": "Outlier-sensitive objective is undesirable"},
])


def print_metrics_table():
    print("\n=== Model Performance Metrics Compared (Table 6.3) ===")
    print(METRICS_TABLE.to_string(index=False))


# ---------------------------------------------------------------------------
# Main: run the whole chapter end to end
# ---------------------------------------------------------------------------

def main():
    # 1-2. Yearly frontiers + Type I/II error framework
    returns_by_year = load_price_data()
    stats_df = plot_yearly_frontiers(returns_by_year)
    frontier_forecast_confusion_matrix(stats_df)

    # 3. ROC-AUC vs PR-AUC: the textbook imbalanced-default example ...
    simulated_credit_default_example()
    # ... and the same lens applied to the frontier-forecasting rule
    frontier_forecast_roc_pr(stats_df)

    # 4. In-sample / out-of-sample evaluation
    two_asset = load_two_asset_data()
    one_shot_train_test_split(two_asset[["AAL", "SPY"]])
    rolling_window_backtest(two_asset, "SPY", "AAL",
                             estimation_window=252, prediction_window=63)

    # 5. Loss functions
    demo_loss_functions()
    plot_cdf_vs_normal(two_asset["SPY"], "Cumulative Distribution of Daily SPY-like Returns",
                        "fig_6_10_6_11_cdf_vs_normal.png")

    # 6. Loss-minimizing two-asset portfolio
    loss_minimizing_portfolio(two_asset, "AAL", "SPY")

    # 7. Metrics comparison reference table
    print_metrics_table()


if __name__ == "__main__":
    main()
