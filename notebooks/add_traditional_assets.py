"""
Add traditional assets (S&P 500, VIX) to the analysis using Yahoo Finance.

This script extends the volatility forecasting analysis to include traditional
market benchmarks alongside cryptocurrency data.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

# Reproducibility
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

# Import core analysis functions
import sys
sys.path.insert(0, '.')

from volatility_analysis import (
    calculate_returns,
    prepare_features,
    garch_rolling_forecast,
    fit_random_forest,
    fit_xgboost,
    fit_lstm,
    calculate_metrics,
    create_figures,
    create_results_tables,
    DATA_DIR,
    FIGURES_DIR
)


def fetch_yahoo_data(ticker, start, end):
    """
    Download daily OHLCV data from Yahoo Finance.

    Returns a DataFrame with lowercase column names and adj_close renamed
    for consistency with the rest of the pipeline.
    """
    df = yf.download(ticker, start=start, end=end, progress=False)

    # yfinance sometimes returns MultiIndex columns - flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [col.lower() for col in df.columns]
    df = df.rename(columns={'adj close': 'adj_close'})

    return df


def run_volatility_analysis(df, asset_name):
    """
    Run the full volatility forecasting pipeline for a single asset.

    Fits GARCH, EGARCH, Random Forest, XGBoost, and LSTM models,
    then returns performance metrics and predictions for each.
    """
    print(f"\n{'='*50}")
    print(f"Analyzing {asset_name}")
    print(f"{'='*50}")

    # Compute log returns and realized volatility
    df = calculate_returns(df)
    print(f"Observations: {len(df)}")
    print(f"Period: {df.index[0].date()} to {df.index[-1].date()}")

    # Build feature matrix for ML models
    df_features = prepare_features(df)

    # 80/20 train-test split
    split_idx = int(len(df_features) * 0.8)

    # Identify feature columns (exclude price data and target)
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'adj_close', 'returns', 'target']
    feature_cols = [c for c in df_features.columns if c not in exclude_cols]

    X = df_features[feature_cols].values
    y = df_features['target'].values

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    results = {}
    predictions = {}

    # --- GARCH(1,1) ---
    print("Fitting GARCH(1,1)...")
    garch_preds, _ = garch_rolling_forecast(
        df['returns'],
        train_size=int(len(df) * 0.8),
        model_type='GARCH'
    )
    # Align predictions with test set length
    n_preds = min(len(garch_preds), len(y_test))
    results['GARCH'] = calculate_metrics(y_test[:n_preds], garch_preds[:n_preds])
    predictions['GARCH'] = garch_preds[:n_preds]
    print(f"  RMSE: {results['GARCH']['RMSE']:.4f}")

    # --- EGARCH ---
    print("Fitting EGARCH...")
    egarch_preds, _ = garch_rolling_forecast(
        df['returns'],
        train_size=int(len(df) * 0.8),
        model_type='EGARCH'
    )
    results['EGARCH'] = calculate_metrics(y_test[:n_preds], egarch_preds[:n_preds])
    predictions['EGARCH'] = egarch_preds[:n_preds]
    print(f"  RMSE: {results['EGARCH']['RMSE']:.4f}")

    # --- Random Forest ---
    print("Fitting Random Forest...")
    rf_model, rf_preds = fit_random_forest(X_train, y_train, X_test)
    results['RF'] = calculate_metrics(y_test, rf_preds)
    predictions['RF'] = rf_preds[:n_preds]
    print(f"  RMSE: {results['RF']['RMSE']:.4f}")

    # --- XGBoost ---
    print("Fitting XGBoost...")
    xgb_model, xgb_preds = fit_xgboost(X_train, y_train, X_test)
    results['XGBoost'] = calculate_metrics(y_test, xgb_preds)
    predictions['XGBoost'] = xgb_preds[:n_preds]
    print(f"  RMSE: {results['XGBoost']['RMSE']:.4f}")

    # --- LSTM ---
    print("Fitting LSTM...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lstm_model, lstm_preds, lookback = fit_lstm(X_train_scaled, y_train, X_test_scaled)

    # LSTM predictions are shorter due to lookback window
    lstm_actual = y_test[lookback:lookback + len(lstm_preds)]
    results['LSTM'] = calculate_metrics(lstm_actual, lstm_preds)

    if len(lstm_preds) > 0:
        predictions['LSTM'] = lstm_preds[:n_preds - lookback]
    else:
        predictions['LSTM'] = np.array([])
    print(f"  RMSE: {results['LSTM']['RMSE']:.4f}")

    return results, predictions, y_test[:n_preds], df


def load_existing_crypto_data():
    """Load previously saved crypto data and results."""
    data = {}
    results = {}

    # Load raw data files
    for asset in ['BTC', 'ETH']:
        try:
            df = pd.read_csv(f'{DATA_DIR}/{asset}_data.csv', index_col=0, parse_dates=True)
            data[asset] = df
            print(f"Loaded {asset}: {len(df)} observations")
        except FileNotFoundError:
            print(f"Warning: {asset} data file not found")
        except Exception as e:
            print(f"Error loading {asset}: {e}")

    # Load saved results
    try:
        results_df = pd.read_csv(f'{DATA_DIR}/full_results.csv', index_col=0)
        for asset in ['BTC', 'ETH']:
            if asset in results_df.index:
                results[asset] = {}
                for col in results_df.columns:
                    try:
                        results[asset][col] = eval(results_df.loc[asset, col])
                    except (SyntaxError, NameError):
                        pass
    except FileNotFoundError:
        print("No existing results file found - will compute fresh")

    return data, results


def main():
    """Main entry point for adding traditional assets to the analysis."""
    print("="*60)
    print("Adding Traditional Assets (S&P 500, VIX)")
    print("="*60)

    # Start with existing crypto data
    all_data, all_results = load_existing_crypto_data()

    # Define traditional assets to fetch
    tickers = {
        'SPX': '^GSPC',  # S&P 500 index
        'VIX': '^VIX'    # CBOE Volatility Index
    }

    start_date = "2019-01-01"
    end_date = "2024-12-31"

    # Fetch and analyze each traditional asset
    for name, ticker in tickers.items():
        print(f"\nFetching {name} from Yahoo Finance...")

        try:
            df = fetch_yahoo_data(ticker, start_date, end_date)

            if len(df) == 0:
                print(f"  {name}: No data returned")
                continue

            # Save raw data
            df.to_csv(f"{DATA_DIR}/{name}_data.csv")
            print(f"  {name}: {len(df)} observations")

            # Run full analysis
            results, predictions, actuals, processed_df = run_volatility_analysis(df, name)

            all_results[name] = results
            all_data[name] = processed_df

        except Exception as e:
            print(f"  {name}: Error - {e}")
            import traceback
            traceback.print_exc()

    # Generate updated figures and tables with all assets
    print("\nGenerating updated figures...")
    create_figures(all_results, all_data)

    print("Generating updated results tables...")
    create_results_tables(all_results)

    # Save consolidated results
    results_df = pd.DataFrame(all_results).T
    results_df.to_csv(f'{DATA_DIR}/full_results.csv')

    print("\n" + "="*60)
    print("Analysis complete - all assets processed")
    print("="*60)


if __name__ == "__main__":
    main()
