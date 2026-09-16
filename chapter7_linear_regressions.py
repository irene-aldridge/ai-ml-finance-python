"""
Chapter 7 Companion Code
Linear Regressions and Market Dependencies
Financial Engineering with AI, ML and Python -- Irene Aldridge

This script reproduces every quantitative idea covered in the chapter:

    1. Deriving the OLS estimator from quadratic-loss minimization,
       beta_hat = (X'X)^-1 X'y = cov(X,y)/var(X)                     (7.1)
    2. Running the SPY -> AAL regression three ways (from scratch,
       statsmodels, sklearn) and plotting the fitted line              (7.2)
    3. Adding an intercept to recover the full CAPM equation and
       diagnosing anomalously large t-statistics                       (7.3)
    4. Estimating CAPM, building a market-neutral AAL/UAL portfolio,
       and comparing a fixed vs. a rolling hedge ratio                 (7.3.x)
    5. Standard errors & 95% confidence intervals for rolling betas,
       a Chow-style z-test for year-on-year beta changes               (7.4)
    6. Decomposing the R^2 collapse into beta^2, var(market),
       1/var(asset)                                                    (7.4.2)
    7. Visualizing beta instability with confidence bands               (7.4.3)
    8. Fama-French three-factor regression                             (7.5)
    9. Testing the pairs trade: Engle-Granger cointegration, ADF test,
       and Ornstein-Uhlenbeck half-life of mean reversion              (7.6)

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from sklearn import linear_model

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------

TRADING_DAYS = 251
START_YEAR, END_YEAR = 2010, 2025


# ---------------------------------------------------------------------------
# 1. Data: SPY, AAL, UAL daily returns (2010-2025)
# ---------------------------------------------------------------------------

def load_price_data():
    """
    Returns a DataFrame indexed by trading date with columns SPY, AAL, UAL
    (daily simple returns).

    """
    path = "Data_Path/"
    spy = pd.read_csv(path + "SPY_Daily.csv")
    spy["SPY"] = (spy["Close"] / spy["Close"].shift(1) - 1).fillna(0)
    aal = pd.read_csv(path + "AAL_Daily.csv")
    aal["AAL"] = (aal["Close"] / aal["Close"].shift(1) - 1).fillna(0)
    ual = pd.read_csv(path + "UAL_Daily.csv")
    ual["UAL"] = (ual["Close"] / ual["Close"].shift(1) - 1).fillna(0)
    df = spy.merge(aal[["Date", "AAL"]], on="Date", how="inner")
    df = df.merge(ual[["Date", "UAL"]], on="Date", how="inner")
    df["Date"] = pd.to_datetime(df["Date"])
    # Duplication check the chapter recommends before trusting a merge:
    assert df.duplicated(subset="Date").sum() == 0, "Duplicate dates found in merge!"
    return df.set_index("Date")[["SPY", "AAL", "UAL"]]

    
def load_fama_french_factors(index):
    """
    Download F-F_Research_Data_Factors_daily.csv from Ken French's data library and
    divide by 100 (the file is in percentage points).
    """
    df = pf.read_csv('F-F_Research_Data_Factors_daily.csv')
    df = df.rename(columns={"SMB": "smb", "HML": "hml", "RF": "rf"})
    return df


# ---------------------------------------------------------------------------
# 2. Section 7.1 / 7.2: OLS from first principles, three ways
# ---------------------------------------------------------------------------

def ols_from_scratch(x, y):
    """
    Closed-form OLS through the origin: beta_hat = (X'X)^-1 X'y = cov(x,y)/var(x)
    (uses the *population*-style dot-product form the chapter derives).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    beta_hat = (x @ y) / (x @ x)   # (X'X)^-1 X'y for a single-column X
    return beta_hat


def compare_ols_implementations(df, x_col="SPY", y_col="AAL"):
    """
    Reproduces the three equivalent ways of computing beta shown in Section
    7.2: the closed-form matrix formula, statsmodels (no intercept), and
    sklearn (no intercept).
    """
    beta_scratch = ols_from_scratch(df[x_col], df[y_col])

    mod = sm.OLS(df[y_col], df[x_col])
    result = mod.fit()
    beta_sm = result.params[x_col]

    reg = linear_model.LinearRegression(fit_intercept=False)
    reg.fit(df[[x_col]], df[y_col])
    beta_sklearn = reg.coef_[0]

    print(f"\n=== OLS beta ({y_col} ~ {x_col}, no intercept), three methods ===")
    print(f"  From scratch (X'X)^-1 X'y : {beta_scratch:.4f}")
    print(f"  statsmodels sm.OLS         : {beta_sm:.4f}")
    print(f"  sklearn LinearRegression   : {beta_sklearn:.4f}")

    return beta_scratch, result


def plot_scatter_with_fit(df, x_col="SPY", y_col="AAL", beta=None, filename="fig_7_1_7_3_scatter.png"):
    """Reproduces Figures 7.1 / 7.3: scatter of returns with the fitted line."""
    plt.figure(figsize=(7, 6))
    plt.scatter(df[x_col], df[y_col], s=2, alpha=0.5)
    if beta is not None:
        x_line = np.linspace(df[x_col].min(), df[x_col].max(), 100)
        plt.plot(x_line, beta * x_line, color="red", linewidth=2,
                  label=f"fitted: y = {beta:.4f}x")
        plt.legend()
    plt.title(f"Daily {y_col} Returns vs. Daily {x_col} Returns")
    plt.xlabel(f"Daily {x_col} Returns")
    plt.ylabel(f"Daily {y_col} Returns")
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


def check_regression_health(df, x_col="SPY", y_col="AAL"):
    """
    The chapter's checklist for anomalously large t-statistics: duplicated
    rows, unit mismatch, near-perfect collinearity. Run this before trusting
    any regression output.
    """
    n_dupes = df.duplicated().sum()
    corr = df[x_col].corr(df[y_col])
    scale_ratio = df[x_col].abs().mean() / df[y_col].abs().mean()

    print(f"\n=== Regression Health Check: {y_col} ~ {x_col} ===")
    print(f"  Duplicated rows:              {n_dupes} (expect 0)")
    print(f"  Correlation({x_col},{y_col}):        {corr:.3f} (expect well below 1.0)")
    print(f"  Mean |{x_col}| / mean |{y_col}| ratio: {scale_ratio:.3f} "
          "(expect roughly comparable scale)")


# ---------------------------------------------------------------------------
# 3. Section 7.3: Adding an intercept -- CAPM
# ---------------------------------------------------------------------------

def estimate_capm(y, x, print_summary=True):
    """
    Estimates y = alpha + beta * x + epsilon via statsmodels OLS with an
    intercept -- the CAPM regression. Returns the fitted results object.
    """
    X = sm.add_constant(x)
    result = sm.OLS(y, X).fit()
    if print_summary:
        print(result.summary())
    return result


def capm_beta(asset_returns, market_returns):
    """Convenience helper matching the chapter's `capm_beta` function."""
    result = sm.OLS(asset_returns, sm.add_constant(market_returns)).fit()
    x_name = market_returns.name
    return result.params[x_name], result.bse[x_name]


# ---------------------------------------------------------------------------
# 4. Market-neutral portfolio: fixed vs. rolling hedge ratio
# ---------------------------------------------------------------------------

def market_neutral_fixed_hedge(df):
    """
    Reproduces the four-step static market-neutral construction and Figure
    7.6 (cumulative P&L of a portfolio long AAL, short h*UAL, h fixed at
    the full-sample CAPM ratio).
    """
    beta_aal, se_aal = capm_beta(df["AAL"], df["SPY"])
    beta_ual, se_ual = capm_beta(df["UAL"], df["SPY"])
    h = beta_aal / beta_ual

    df = df.copy()
    df["R_portfolio"] = df["AAL"] - h * df["UAL"]
    df["cumulative_pnl"] = (1 + df["R_portfolio"]).cumprod() - 1

    fit_check = sm.OLS(df["R_portfolio"], sm.add_constant(df["SPY"])).fit()

    print("\n=== Market-Neutral Portfolio (fixed, full-sample hedge ratio) ===")
    print(f"  beta_AAL = {beta_aal:.4f} (SE={se_aal:.4f}), "
          f"beta_UAL = {beta_ual:.4f} (SE={se_ual:.4f})")
    print(f"  Hedge ratio h = beta_AAL / beta_UAL = {h:.4f}")
    print(f"  Portfolio beta  : {fit_check.params['SPY']:.4f} (target: ~0)")
    print(f"  Portfolio alpha : {fit_check.params['const'] * TRADING_DAYS * 100:.2f}% (annualized)")
    print(f"  Portfolio R^2   : {fit_check.rsquared:.4f}")

    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["cumulative_pnl"], color="seagreen", linewidth=1.5,
              label="Market-neutral (AAL - h*UAL)")
    plt.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    plt.title("Market-Neutral Portfolio vs. Components (Fixed Hedge)")
    plt.ylabel("Cumulative return")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_7_6_market_neutral_fixed_hedge.png", dpi=150)
    plt.show()

    return h, df


def market_neutral_rolling_hedge(df, window=63):
    """
    Reproduces the rolling 63-day hedge ratio of Figure 7.8 / 7.9: at each
    step, re-estimate beta_AAL and beta_UAL on the trailing `window` days,
    form h_t = beta_AAL,t / beta_UAL,t, and compute both the rolling-hedge
    and fixed-hedge portfolio returns for comparison.
    """
    beta_aal_full, _ = capm_beta(df["AAL"], df["SPY"])
    beta_ual_full, _ = capm_beta(df["UAL"], df["SPY"])
    h_fixed = beta_aal_full / beta_ual_full

    records = []
    for t in range(window, len(df)):
        win = df.iloc[t - window:t]
        X = sm.add_constant(win["SPY"])
        fit_a = sm.OLS(win["AAL"], X).fit()
        fit_u = sm.OLS(win["UAL"], X).fit()
        b_a, b_u = fit_a.params["SPY"], fit_u.params["SPY"]
        h = b_a / b_u if b_u != 0 else np.nan
        records.append({
            "date": df.index[t],
            "beta_aal": b_a,
            "beta_ual": b_u,
            "hedge_ratio": h,
            "r_port_roll": df["AAL"].iloc[t] - h * df["UAL"].iloc[t] if np.isfinite(h) else np.nan,
            "r_port_fixed": df["AAL"].iloc[t] - h_fixed * df["UAL"].iloc[t],
        })
    roll_df = pd.DataFrame(records).set_index("date")

    # Plot hedge ratio drift (Figure 7.8), clipped for readability
    plt.figure(figsize=(10, 4))
    plt.plot(roll_df.index, roll_df["hedge_ratio"].clip(-10, 10),
              color="#1D9E75", linewidth=1.2, label=f"Rolling hedge ratio ({window}-day)")
    plt.axhline(h_fixed, color="#888780", linestyle="--", linewidth=1,
                label=f"Fixed hedge ratio ({h_fixed:.2f})")
    plt.ylabel("Hedge ratio (UAL per $1 AAL), clipped to [-10, 10]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_7_8_rolling_hedge_ratio.png", dpi=150)
    plt.show()

    # Plot cumulative performance: rolling vs fixed hedge (Figure 7.9)
    cum_roll = (1 + roll_df["r_port_roll"].fillna(0)).cumprod() - 1
    cum_fixed = (1 + roll_df["r_port_fixed"]).cumprod() - 1

    plt.figure(figsize=(10, 4))
    plt.plot(roll_df.index, cum_roll, color="#1D9E75", linewidth=1.5, label="Rolling hedge")
    plt.plot(roll_df.index, cum_fixed, color="#888780", linewidth=1.5,
              linestyle="--", label="Fixed hedge")
    plt.axhline(0, color="#B4B2A9", linewidth=0.8)
    plt.ylabel("Cumulative return")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_7_9_rolling_vs_fixed_performance.png", dpi=150)
    plt.show()

    print(f"\n=== Rolling ({window}-day) vs. fixed hedge: final cumulative return ===")
    print(f"  Rolling hedge : {cum_roll.iloc[-1]:.2%}")
    print(f"  Fixed hedge   : {cum_fixed.iloc[-1]:.2%}")

    return roll_df


# ---------------------------------------------------------------------------
# 5. Section 7.4: Rolling annual CAPM with standard errors, CIs, z-tests
# ---------------------------------------------------------------------------

def rolling_annual_capm(df):
    """
    Reproduces Table 7.2/7.3: one CAPM regression per calendar year for
    both AAL and UAL, with SE, 95% CI, and a year-on-year z-test for
    whether the change in beta_AAL is statistically significant (Section
    7.4.1's Chow-style test).
    """
    results = []
    for start in pd.date_range(f"{START_YEAR}-01-01", f"{END_YEAR + 1}-01-01", freq="YS"):
        end = start + pd.DateOffset(years=1)
        mask = (df.index >= start) & (df.index < end)
        sub = df[mask]
        if len(sub) < 30:
            continue

        X = sm.add_constant(sub["SPY"])
        fit_aal = sm.OLS(sub["AAL"], X).fit()
        fit_ual = sm.OLS(sub["UAL"], X).fit()

        results.append({
            "year": start.year,
            "n_obs": len(sub),
            "r2": fit_aal.rsquared,
            "beta_aal": fit_aal.params["SPY"],
            "se_aal": fit_aal.bse["SPY"],
            "ci_aal_lo": fit_aal.conf_int().loc["SPY", 0],
            "ci_aal_hi": fit_aal.conf_int().loc["SPY", 1],
            "beta_ual": fit_ual.params["SPY"],
            "se_ual": fit_ual.bse["SPY"],
            "ci_ual_lo": fit_ual.conf_int().loc["SPY", 0],
            "ci_ual_hi": fit_ual.conf_int().loc["SPY", 1],
            "var_spy": sub["SPY"].var() * TRADING_DAYS,
            "var_aal": sub["AAL"].var() * TRADING_DAYS,
            "var_ual": sub["UAL"].var() * TRADING_DAYS,
        })

    res = pd.DataFrame(results)

    # Year-on-year significance test for beta_AAL (Section 7.4.1)
    res["z_aal"] = res["beta_aal"].diff() / np.sqrt(
        res["se_aal"] ** 2 + res["se_aal"].shift(1) ** 2
    )
    res["significant_change"] = res["z_aal"].abs() > 1.96

    print("\n=== Rolling Annual CAPM Estimates (Table 7.2 / 7.3) ===")
    print(res[["year", "n_obs", "r2", "beta_aal", "se_aal", "ci_aal_lo", "ci_aal_hi",
               "beta_ual", "se_ual", "z_aal", "significant_change"]].round(3).to_string(index=False))

    return res


# ---------------------------------------------------------------------------
# 6. Section 7.4.2: Decomposing the R^2 collapse
# ---------------------------------------------------------------------------

def decompose_r_squared(res):
    """
    R^2 = beta^2 * var(market) / var(asset). Reproduces Table 7.4 and the
    three-panel bar chart isolating each driver's year-by-year contribution.
    """
    res = res.copy()
    res["r2_implied"] = (res["beta_aal"] ** 2 * res["var_spy"]) / res["var_aal"]

    print("\n=== R^2 Decomposition: beta^2 * var(SPY) / var(AAL) (Table 7.4) ===")
    print(res[["year", "beta_aal", "var_spy", "var_aal", "r2_implied", "r2"]]
          .round(3).to_string(index=False))

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].bar(res["year"], res["beta_aal"] ** 2, color="#1565C0", alpha=0.7, label=r"$\hat{\beta}^2_{AAL}$")
    axes[0].set_ylabel("beta^2 (systematic sensitivity)")
    axes[0].legend()
    axes[0].grid(alpha=0.4)

    axes[1].bar(res["year"], res["var_spy"], color="#2E7D32", alpha=0.7, label="var(SPY), annualized")
    axes[1].set_ylabel("Market variance")
    axes[1].legend()
    axes[1].grid(alpha=0.4)

    axes[2].bar(res["year"], res["var_aal"], color="#B71C1C", alpha=0.7, label="var(AAL), annualized")
    axes[2].set_ylabel("AAL idiosyncratic variance")
    axes[2].set_xlabel("Year")
    axes[2].legend()
    axes[2].grid(alpha=0.4)

    fig.suptitle("R^2 Decomposition: beta^2 * var(SPY) / var(AAL)")
    plt.tight_layout()
    plt.savefig("fig_7_r2_decomposition.png", dpi=150)
    plt.show()

    return res


# ---------------------------------------------------------------------------
# 7. Section 7.4.3: Visualizing beta instability with confidence bands
# ---------------------------------------------------------------------------

def plot_rolling_beta_confidence_bands(res):
    """Reproduces Figure 7.10: rolling betas with shaded 95% CIs and an R^2 panel."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    for ax, asset, color in zip(axes[:2], ["aal", "ual"], ["#1565C0", "#7B3F00"]):
        ax.plot(res["year"], res[f"beta_{asset}"], marker="o", color=color,
                  linewidth=2, label=f"beta_{asset.upper()}")
        ax.fill_between(res["year"], res[f"ci_{asset}_lo"], res[f"ci_{asset}_hi"],
                          alpha=0.2, color=color, label="95% CI")
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=1, label="beta = 1 (market)")
        ax.set_ylabel(f"beta_{asset.upper()}")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.4)

        # Annotate year-on-year change for AAL, as in Figure 7.10
        if asset == "aal":
            for i in range(1, len(res)):
                delta = res["beta_aal"].iloc[i] - res["beta_aal"].iloc[i - 1]
                sig = res["significant_change"].iloc[i]
                marker = "significant" if sig else ""
                ax.annotate(f"{delta:+.2f}{' *' if sig else ''}",
                             (res["year"].iloc[i], res["beta_aal"].iloc[i]),
                             textcoords="offset points", xytext=(0, 12), fontsize=8,
                             color="red" if sig else "grey")

    axes[2].bar(res["year"], res["r2"], color="grey", alpha=0.7, label="R^2 (model fit)")
    axes[2].axhline(0.10, color="red", linestyle="--", linewidth=1, label="Caution threshold (R^2=0.10)")
    axes[2].set_ylabel("R^2")
    axes[2].set_xlabel("Year")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.4)

    fig.suptitle("Rolling Annual CAPM Betas with 95% Confidence Intervals\n"
                  "(* marks year-on-year changes significant at 5%)")
    plt.tight_layout()
    plt.savefig("fig_7_10_beta_confidence_bands.png", dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# 8. Section 7.5: Fama-French three-factor model
# ---------------------------------------------------------------------------

def fama_french_three_factor(df, ff):
    """
    Estimates the Fama-French three-factor regression for AAL and compares
    it against the single-factor CAPM (Table 7.6).
    """
    merged = df.join(ff, how="inner")
    merged["AAL_excess"] = merged["AAL"] - merged["RF"]
    merged["Mkt-RF"] = merged["SPY"] - merged["RF"]

    X3 = sm.add_constant(merged[["Mkt-RF", "SMB", "HML"]])
    ff3 = sm.OLS(merged["AAL_excess"], X3).fit()

    capm = estimate_capm(merged["AAL"], merged["SPY"], print_summary=False)

    comparison = pd.DataFrame({
        "CAPM": {
            "alpha_daily_pct": capm.params["const"] * 100,
            "beta_Mkt": capm.params["SPY"],
            "beta_SMB": np.nan,
            "beta_HML": np.nan,
            "R2": capm.rsquared,
        },
        "Fama-French 3-Factor": {
            "alpha_daily_pct": ff3.params["const"] * 100,
            "beta_Mkt": ff3.params["Mkt-RF"],
            "beta_SMB": ff3.params["SMB"],
            "beta_HML": ff3.params["HML"],
            "R2": ff3.rsquared,
        },
    }).T

    print("\n=== CAPM vs. Fama-French 3-Factor for AAL (Table 7.6) ===")
    print(comparison.round(4))

    return ff3, comparison


# ---------------------------------------------------------------------------
# 9. Section 7.6: Cointegration and mean-reversion (Engle-Granger, ADF, OU)
# ---------------------------------------------------------------------------

def test_cointegration(df):
    """
    Reproduces Section 7.6's Engle-Granger two-step test on log(AAL price)
    vs. log(UAL price): (1) cointegrating OLS regression, (2) ADF test on
    the residual spread, plus the statsmodels `coint` convenience wrapper.
    """
    price_aal = df["AAL_Price"]
    price_ual = df["UAL_Price"]
    log_aal = np.log(price_aal)
    log_ual = np.log(price_ual)

    # Step 1: cointegrating regression
    X = sm.add_constant(log_ual)
    coint_reg = sm.OLS(log_aal, X).fit()
    h_coint = coint_reg.params[log_ual.name if hasattr(log_ual, "name") else "UAL"]
    spread = coint_reg.resid

    # Step 2: ADF test on the residual spread
    adf_stat, adf_pvalue, *_ = adfuller(spread, autolag="AIC")

    # Convenience wrapper (full Engle-Granger with MacKinnon critical values)
    score, pvalue, crit_values = coint(log_aal, log_ual)

    print("\n=== Engle-Granger Cointegration Test: log(AAL) ~ log(UAL) ===")
    print(f"  Cointegrating hedge ratio (h): {h_coint:.4f}")
    print(f"  ADF statistic on spread:       {adf_stat:.3f}, p-value: {adf_pvalue:.4f}")
    print(f"  Engle-Granger p-value:         {pvalue:.4f}")
    print(f"  Critical values (1%,5%,10%):   {np.round(crit_values, 3)}")

    plt.figure(figsize=(10, 4))
    plt.plot(df.index, spread, color="purple", linewidth=1)
    plt.axhline(spread.mean(), color="black", linestyle="--", linewidth=1, label="mean")
    plt.title("Cointegrating Spread: log(AAL) - h * log(UAL)")
    plt.ylabel("Spread")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_7_cointegrating_spread.png", dpi=150)
    plt.show()

    return h_coint, spread, adf_pvalue, pvalue


def estimate_mean_reversion_half_life(spread):
    """
    Fits the discrete-time Ornstein-Uhlenbeck analogue:
        delta_spread_t = a + b * spread_{t-1} + eta_t
    lambda_hat = -b_hat, half_life = ln(2) / lambda_hat (in trading days).
    """
    spread = pd.Series(spread).reset_index(drop=True)
    delta_spread = spread.diff().dropna()
    spread_lag = spread.shift(1).dropna()
    spread_lag = spread_lag.loc[delta_spread.index]

    X = sm.add_constant(spread_lag)
    ou_fit = sm.OLS(delta_spread, X).fit()
    b_hat = ou_fit.params.iloc[1]
    lam = -b_hat
    half_life = np.log(2) / lam if lam > 0 else np.inf

    print("\n=== Ornstein-Uhlenbeck Mean-Reversion Fit ===")
    print(f"  b_hat (mean-reversion coefficient): {b_hat:.5f}")
    print(f"  Implied lambda:                     {lam:.5f}")
    print(f"  Half-life of mean reversion:        {half_life:.1f} trading days")

    return half_life, ou_fit


# ---------------------------------------------------------------------------
# Main: run the whole chapter end to end
# ---------------------------------------------------------------------------

def main():
    # 1. Data
    df = load_price_data()

    # 2. Sections 7.1 / 7.2: OLS from scratch, statsmodels, sklearn
    beta_scratch, result_no_intercept = compare_ols_implementations(df, "SPY", "AAL")
    check_regression_health(df, "SPY", "AAL")
    plot_scatter_with_fit(df, "SPY", "AAL", beta=beta_scratch)

    # 3. Section 7.3: CAPM with intercept
    print("\n=== CAPM Regression: AAL ~ SPY (with intercept) ===")
    capm_aal = estimate_capm(df["AAL"], df["SPY"])
    capm_ual = estimate_capm(df["UAL"], df["SPY"], print_summary=False)
    print(f"\nDaily alpha (AAL): {capm_aal.params['const']:.5%}  "
          f"(~{capm_aal.params['const'] * TRADING_DAYS:.2%} annualized)")
    print(f"Daily alpha (UAL): {capm_ual.params['const']:.5%}  "
          f"(~{capm_ual.params['const'] * TRADING_DAYS:.2%} annualized)")

    # 4. Market-neutral portfolio: fixed vs rolling hedge
    h_fixed, mn_df = market_neutral_fixed_hedge(df)
    roll_df = market_neutral_rolling_hedge(df, window=63)

    # 5. Rolling annual CAPM with SE / CI / z-test (Table 7.2, 7.3, Section 7.4.1)
    res = rolling_annual_capm(df)

    # 6. R^2 decomposition (Table 7.4)
    res = decompose_r_squared(res)

    # 7. Beta instability with confidence bands (Figure 7.10)
    plot_rolling_beta_confidence_bands(res)

    # 8. Fama-French three-factor model (Table 7.6)
    ff = load_fama_french_factors(df.index)
    ff3_result, ff_comparison = fama_french_three_factor(df, ff)

    # 9. Cointegration and mean reversion (Section 7.6)
    h_coint, spread, adf_pvalue, eg_pvalue = test_cointegration(df)
    half_life, ou_fit = estimate_mean_reversion_half_life(spread)

    print(f"\nReconciling hedge ratios: CAPM-based h = {h_fixed:.2f}  "
          f"vs. cointegration-based h = {h_coint:.2f}")

    print("\nAll figures saved as PNG files in the working directory.")


if __name__ == "__main__":
    main()
