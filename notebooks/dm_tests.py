"""
Diebold-Mariano Tests for Forecast Comparison

Compares forecast accuracy between econometric and ML models using the
Diebold-Mariano (1995) test statistic.
"""

import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

np.random.seed(42)

DATA_DIR = "../data"


def diebold_mariano_test(actual, forecast_1, forecast_2, horizon=1):
    """
    Diebold-Mariano test for comparing forecast accuracy.

    Tests whether two forecasts have equal predictive accuracy.
    A positive statistic means forecast_2 outperforms forecast_1.

    Parameters
    ----------
    actual : array-like
        Realized values
    forecast_1 : array-like
        Predictions from benchmark model
    forecast_2 : array-like
        Predictions from challenger model
    horizon : int
        Forecast horizon (for autocorrelation correction)

    Returns
    -------
    tuple
        (DM statistic, p-value)
    """
    errors_1 = actual - forecast_1
    errors_2 = actual - forecast_2

    # Loss differential using squared errors
    loss_diff = errors_1**2 - errors_2**2
    loss_diff = loss_diff[~np.isnan(loss_diff)]
    n = len(loss_diff)

    if n < 10:
        return np.nan, np.nan

    mean_loss_diff = np.mean(loss_diff)
    variance = np.var(loss_diff, ddof=1)

    # Adjust variance for multi-step forecasts (autocorrelation correction)
    if horizon > 1:
        autocovariance_sum = 0
        for lag in range(1, horizon):
            centered = loss_diff - mean_loss_diff
            gamma = np.mean(centered[lag:] * centered[:-lag])
            autocovariance_sum += gamma
        variance = (variance + 2 * autocovariance_sum) / n
    else:
        variance = variance / n

    if variance <= 0:
        return np.nan, np.nan

    dm_statistic = mean_loss_diff / np.sqrt(variance)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_statistic)))

    return dm_statistic, p_value


def generate_forecasts(asset):
    """
    Generate forecasts from all models for a given asset.

    Returns actual values and a dictionary of predictions keyed by model name.
    """
    from sklearn.preprocessing import StandardScaler

    from volatility_analysis import (
        fit_lstm,
        fit_random_forest,
        fit_xgboost,
        garch_rolling_forecast,
        prepare_features,
    )

    # Load price data
    df = pd.read_csv(f'{DATA_DIR}/{asset}_data.csv', index_col=0, parse_dates=True)
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    df = df.dropna()

    # Prepare ML features
    df_features = prepare_features(df)
    train_size = int(len(df_features) * 0.8)

    feature_cols = [
        c for c in df_features.columns
        if c not in ['open', 'high', 'low', 'close', 'volume', 'adj_close', 'returns', 'target']
    ]

    X = df_features[feature_cols].values
    y = df_features['target'].values

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Generate predictions from each model
    predictions = {}
    garch_train_size = int(len(df) * 0.8)

    # GARCH models
    garch_preds, _ = garch_rolling_forecast(df['returns'], train_size=garch_train_size, model_type='GARCH')
    egarch_preds, _ = garch_rolling_forecast(df['returns'], train_size=garch_train_size, model_type='EGARCH')

    # Align prediction lengths
    n_test = min(len(garch_preds), len(y_test))
    predictions['GARCH'] = garch_preds[:n_test]
    predictions['EGARCH'] = egarch_preds[:n_test]

    # ML models
    _, rf_preds = fit_random_forest(X_train, y_train, X_test)
    _, xgb_preds = fit_xgboost(X_train, y_train, X_test)
    predictions['RF'] = rf_preds[:n_test]
    predictions['XGBoost'] = xgb_preds[:n_test]

    # LSTM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    _, lstm_preds, lookback = fit_lstm(X_train_scaled, y_train, X_test_scaled)

    # Pad LSTM predictions to match other models (LSTM needs lookback period)
    lstm_aligned = np.full(n_test, np.nan)
    if len(lstm_preds) > 0:
        end_idx = min(lookback + len(lstm_preds), n_test)
        lstm_aligned[lookback:end_idx] = lstm_preds[:end_idx - lookback]
    predictions['LSTM'] = lstm_aligned

    return y_test[:n_test], predictions


def significance_stars(p):
    """Convert p-value to standard significance notation."""
    if pd.isna(p):
        return "---"
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return "ns"


def generate_latex_table(results, comparisons, assets):
    """Generate LaTeX table from results."""
    table = r"""
\begin{table}[htbp]
\centering
\caption{Diebold-Mariano Test Results for Forecast Comparison}
\label{tab:dm_test}
\begin{tabular}{lcccc}
\toprule
Comparison & BTC & ETH & SPX & VIX \\
\midrule
"""

    for challenger, benchmark in comparisons:
        row = f"{challenger} vs {benchmark} "
        for asset in assets:
            if asset in results[(challenger, benchmark)]:
                stars = results[(challenger, benchmark)][asset]['stars']
                row += f"& {stars} "
            else:
                row += "& --- "
        row += "\\\\\n"
        table += row

    table += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: *** p<0.01, ** p<0.05, * p<0.10, ns = not significant.
\item Positive DM statistic indicates the first model outperforms the second.
\end{tablenotes}
\end{table}
"""
    return table


def main():
    """Run DM tests for all model comparisons across all assets."""
    print("=" * 60)
    print("DIEBOLD-MARIANO TEST RESULTS")
    print("=" * 60)

    assets = ['BTC', 'ETH', 'SPX', 'VIX']

    # Model pairs to compare: (challenger, benchmark)
    comparisons = [
        ('RF', 'GARCH'),
        ('XGBoost', 'GARCH'),
        ('EGARCH', 'GARCH'),
        ('XGBoost', 'RF'),
        ('LSTM', 'GARCH'),
    ]

    results = {comp: {} for comp in comparisons}

    for asset in assets:
        print(f"\nProcessing {asset}...")

        try:
            actual, predictions = generate_forecasts(asset)

            for challenger, benchmark in comparisons:
                dm_stat, p_val = diebold_mariano_test(
                    actual,
                    predictions[benchmark],
                    predictions[challenger],
                    horizon=5
                )

                results[(challenger, benchmark)][asset] = {
                    'dm_stat': dm_stat,
                    'p_value': p_val,
                    'stars': significance_stars(p_val)
                }

                print(f"  {challenger} vs {benchmark}: DM={dm_stat:.3f}, p={p_val:.4f} {significance_stars(p_val)}")

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    # Generate and save LaTeX table
    print("\n" + "=" * 60)
    print("LaTeX Table")
    print("=" * 60)

    latex_table = generate_latex_table(results, comparisons, assets)
    print(latex_table)

    with open(f'{DATA_DIR}/dm_results.tex', 'w') as f:
        f.write(latex_table)

    # Save detailed results as CSV
    rows = []
    for (challenger, benchmark), asset_results in results.items():
        for asset, vals in asset_results.items():
            rows.append({
                'Comparison': f"{challenger} vs {benchmark}",
                'Asset': asset,
                'DM_Statistic': vals['dm_stat'],
                'P_Value': vals['p_value'],
                'Significance': vals['stars']
            })

    df_results = pd.DataFrame(rows)
    df_results.to_csv(f'{DATA_DIR}/dm_results_detailed.csv', index=False)

    print(f"\nResults saved to {DATA_DIR}/dm_results.tex and {DATA_DIR}/dm_results_detailed.csv")


if __name__ == "__main__":
    main()
