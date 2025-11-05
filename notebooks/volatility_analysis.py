"""
Volatility Forecasting Analysis

Compares GARCH family models against machine learning approaches
for predicting volatility in cryptocurrency and traditional assets.

Author: [Your Name]
Date: January 2026
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats
from arch import arch_model
from polygon import RESTClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

warnings.filterwarnings('ignore')

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Configuration
API_KEY = "G8rIJ6VThkNC8skyqhmlPbHn_0jhTTM3"
DATA_DIR = "../data"
FIGURES_DIR = "../figures"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Data Collection
# -----------------------------------------------------------------------------

def fetch_price_data(ticker, start_date, end_date):
    """
    Fetch daily OHLCV data from Polygon API.

    Returns a DataFrame indexed by date with columns:
    open, high, low, close, volume
    """
    client = RESTClient(API_KEY)

    records = []
    for bar in client.list_aggs(ticker=ticker, multiplier=1, timespan="day",
                                 from_=start_date, to=end_date, limit=50000):
        records.append({
            'date': datetime.fromtimestamp(bar.timestamp / 1000),
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        })

    df = pd.DataFrame(records)
    if len(df) > 0:
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)

    return df


def collect_all_assets():
    """Download price data for all assets in the study."""
    start_date = "2019-01-01"
    end_date = "2024-12-31"

    tickers = {
        'BTC': 'X:BTCUSD',
        'ETH': 'X:ETHUSD',
        'SPX': 'I:SPX',
        'VIX': 'I:VIX'
    }

    datasets = {}
    for name, ticker in tickers.items():
        print(f"Fetching {name}...")
        try:
            df = fetch_price_data(ticker, start_date, end_date)
            if len(df) > 0:
                datasets[name] = df
                df.to_csv(f"{DATA_DIR}/{name}_data.csv")
                print(f"  {name}: {len(df)} observations")
            else:
                print(f"  {name}: No data returned")
        except Exception as e:
            print(f"  {name}: Error - {e}")

    return datasets


# -----------------------------------------------------------------------------
# Data Processing
# -----------------------------------------------------------------------------

def compute_log_returns(df):
    """Add log returns column and drop the first row (NaN)."""
    df = df.copy()
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    return df.dropna()


def compute_realized_vol(returns, window=21):
    """
    Rolling realized volatility, annualized.
    Standard approach: rolling std * sqrt(252)
    """
    return returns.rolling(window=window).std() * np.sqrt(252)


def build_feature_matrix(df, num_lags=22):
    """
    Construct features for ML models.

    Features include:
    - Lagged returns (1 to num_lags days)
    - Rolling volatility at different windows (5, 10, 21 days)
    - Lagged volatility values

    Target: 5-day forward realized volatility
    """
    df = df.copy()

    # Lagged returns
    for lag in range(1, num_lags + 1):
        df[f'ret_lag_{lag}'] = df['returns'].shift(lag)

    # Rolling volatility features
    df['vol_5d'] = df['returns'].rolling(5).std() * np.sqrt(252)
    df['vol_10d'] = df['returns'].rolling(10).std() * np.sqrt(252)
    df['vol_21d'] = df['returns'].rolling(21).std() * np.sqrt(252)

    # Lagged volatility
    for lag in range(1, 6):
        df[f'vol_lag_{lag}'] = df['vol_21d'].shift(lag)

    # Target: 5-day forward realized volatility
    df['target'] = df['returns'].shift(-1).rolling(5).std() * np.sqrt(252)
    df['target'] = df['target'].shift(-4)

    return df.dropna()


# -----------------------------------------------------------------------------
# GARCH Models
# -----------------------------------------------------------------------------

def fit_garch_model(returns, p=1, q=1, vol_model='Garch'):
    """
    Fit a GARCH or EGARCH model.

    Returns are scaled by 100 for numerical stability,
    which is standard practice in the arch package.
    """
    scaled_returns = returns * 100
    model = arch_model(scaled_returns, vol=vol_model, p=p, q=q, dist='normal')
    result = model.fit(disp='off')
    return result


def garch_rolling_forecast(returns, train_size, vol_model='Garch'):
    """
    Generate one-step-ahead volatility forecasts using expanding window.

    For each day after the training period, we refit the model on all
    available history and forecast one day ahead.
    """
    forecasts = []
    actuals = []
    scaled_returns = returns * 100

    for t in range(train_size, len(returns) - 1):
        history = scaled_returns.iloc[:t]

        try:
            model = arch_model(history, vol=vol_model, p=1, q=1, dist='normal')
            fit = model.fit(disp='off', show_warning=False)

            # Forecast variance, then convert to annualized vol
            fc = fit.forecast(horizon=1)
            vol_forecast = np.sqrt(fc.variance.values[-1, 0]) / 100 * np.sqrt(252)
            forecasts.append(vol_forecast)

            # Actual: 5-day forward realized vol
            if t + 5 < len(returns):
                actual_vol = returns.iloc[t:t+5].std() * np.sqrt(252)
                actuals.append(actual_vol)
            else:
                actuals.append(np.nan)

        except Exception:
            forecasts.append(np.nan)
            actuals.append(np.nan)

    return np.array(forecasts), np.array(actuals)


# -----------------------------------------------------------------------------
# Machine Learning Models
# -----------------------------------------------------------------------------

def train_random_forest(X_train, y_train, X_test):
    """Fit Random Forest with reasonable defaults for this application."""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=SEED,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, predictions


def train_xgboost(X_train, y_train, X_test):
    """Fit XGBoost with standard hyperparameters."""
    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=SEED,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, predictions


def create_sequences(X, y, lookback):
    """
    Convert feature matrix to sequences for LSTM.
    Each sample becomes a sequence of the previous `lookback` observations.
    """
    X_seq, y_seq = [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i - lookback:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


def train_lstm(X_train, y_train, X_test, lookback=22):
    """
    Train a simple LSTM for volatility prediction.

    Architecture is kept minimal since more complex models
    didn't improve results significantly in preliminary tests.
    """
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, lookback)
    X_test_seq, _ = create_sequences(X_test, np.zeros(len(X_test)), lookback)

    model = Sequential([
        LSTM(32, input_shape=(lookback, X_train.shape[1]), return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

    model.fit(
        X_train_seq, y_train_seq,
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

    predictions = model.predict(X_test_seq, verbose=0).flatten()
    return model, predictions, lookback


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def compute_forecast_metrics(actual, predicted):
    """
    Calculate standard forecast accuracy metrics.

    Returns dict with RMSE, MAE, and MAPE.
    """
    # Remove any NaN pairs
    valid = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[valid]
    predicted = predicted[valid]

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}


def diebold_mariano_test(actual, pred1, pred2):
    """
    Diebold-Mariano test for comparing forecast accuracy.

    Tests whether the squared errors from pred1 and pred2 are
    significantly different. Returns DM statistic and p-value.
    """
    e1 = actual - pred1
    e2 = actual - pred2
    d = e1**2 - e2**2

    d = d[~np.isnan(d)]

    if len(d) == 0:
        return np.nan, np.nan

    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1) / len(d)

    if var_d > 0:
        dm_stat = mean_d / np.sqrt(var_d)
        p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    else:
        dm_stat = np.nan
        p_value = np.nan

    return dm_stat, p_value


# -----------------------------------------------------------------------------
# Main Analysis Pipeline
# -----------------------------------------------------------------------------

def analyze_asset(df, asset_name):
    """
    Run the full analysis pipeline for a single asset.

    Fits all models (GARCH, EGARCH, RF, XGBoost, LSTM) and
    returns performance metrics and predictions.
    """
    print(f"\n{'='*50}")
    print(f"Analyzing {asset_name}")
    print(f"{'='*50}")

    # Prepare data
    df = compute_log_returns(df)
    print(f"Observations: {len(df)}")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")

    df_features = build_feature_matrix(df)

    # 80/20 train-test split
    split_idx = int(len(df_features) * 0.8)

    # Identify feature columns (exclude price data and target)
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'returns', 'target']
    feature_cols = [c for c in df_features.columns if c not in exclude_cols]

    X = df_features[feature_cols].values
    y = df_features['target'].values

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    results = {}
    predictions = {}

    # GARCH(1,1)
    print("Fitting GARCH(1,1)...")
    garch_pred, _ = garch_rolling_forecast(
        df['returns'],
        train_size=int(len(df) * 0.8),
        vol_model='Garch'
    )
    n_samples = min(len(garch_pred), len(y_test))
    results['GARCH'] = compute_forecast_metrics(y_test[:n_samples], garch_pred[:n_samples])
    predictions['GARCH'] = garch_pred[:n_samples]
    print(f"  RMSE: {results['GARCH']['RMSE']:.4f}")

    # EGARCH
    print("Fitting EGARCH...")
    egarch_pred, _ = garch_rolling_forecast(
        df['returns'],
        train_size=int(len(df) * 0.8),
        vol_model='EGARCH'
    )
    results['EGARCH'] = compute_forecast_metrics(y_test[:n_samples], egarch_pred[:n_samples])
    predictions['EGARCH'] = egarch_pred[:n_samples]
    print(f"  RMSE: {results['EGARCH']['RMSE']:.4f}")

    # Random Forest
    print("Fitting Random Forest...")
    _, rf_pred = train_random_forest(X_train, y_train, X_test)
    results['RF'] = compute_forecast_metrics(y_test, rf_pred)
    predictions['RF'] = rf_pred[:n_samples]
    print(f"  RMSE: {results['RF']['RMSE']:.4f}")

    # XGBoost
    print("Fitting XGBoost...")
    _, xgb_pred = train_xgboost(X_train, y_train, X_test)
    results['XGBoost'] = compute_forecast_metrics(y_test, xgb_pred)
    predictions['XGBoost'] = xgb_pred[:n_samples]
    print(f"  RMSE: {results['XGBoost']['RMSE']:.4f}")

    # LSTM (with scaled features)
    print("Fitting LSTM...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    _, lstm_pred, lookback = train_lstm(X_train_scaled, y_train, X_test_scaled)

    # Adjust for lookback offset in LSTM predictions
    lstm_actual = y_test[lookback:lookback + len(lstm_pred)]
    results['LSTM'] = compute_forecast_metrics(lstm_actual, lstm_pred)
    predictions['LSTM'] = lstm_pred[:n_samples - lookback] if len(lstm_pred) > 0 else np.array([])
    print(f"  RMSE: {results['LSTM']['RMSE']:.4f}")

    return results, predictions, y_test[:n_samples], df


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

def generate_figures(all_results, all_data):
    """Create figures for thesis chapters."""

    # Figure 1: Returns distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, (asset, df) in enumerate(all_data.items()):
        ax = axes[idx // 2, idx % 2]
        returns = compute_log_returns(df.copy())['returns']

        ax.hist(returns, bins=50, density=True, alpha=0.7, edgecolor='black')
        ax.set_title(f'{asset} Daily Returns')
        ax.set_xlabel('Log Returns')
        ax.set_ylabel('Density')

        # Summary statistics
        stats_text = (
            f"Mean: {returns.mean():.4f}\n"
            f"Std: {returns.std():.4f}\n"
            f"Skew: {returns.skew():.2f}\n"
            f"Kurt: {returns.kurtosis():.2f}"
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/returns_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 2: Volatility time series
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, (asset, df) in enumerate(all_data.items()):
        ax = axes[idx // 2, idx % 2]
        returns = compute_log_returns(df.copy())['returns']
        vol = compute_realized_vol(returns)

        ax.plot(vol.index, vol.values, linewidth=0.8)
        ax.set_title(f'{asset} Realized Volatility (21-day)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Annualized Volatility')

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/volatility_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 3: Model comparison (bar chart)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    models = ['GARCH', 'EGARCH', 'RF', 'XGBoost', 'LSTM']
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']

    for idx, (asset, results) in enumerate(all_results.items()):
        ax = axes[idx // 2, idx % 2]

        valid_models = [m for m in models if m in results]
        rmse_vals = [results[m]['RMSE'] for m in valid_models]

        bars = ax.bar(valid_models, rmse_vals, color=colors[:len(valid_models)])
        ax.set_title(f'{asset} - Model Comparison (RMSE)')
        ax.set_ylabel('RMSE')
        ax.tick_params(axis='x', rotation=45)

        for bar, val in zip(bars, rmse_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 4: Crypto vs Traditional assets
    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(models))
    bar_width = 0.35

    crypto_rmse = []
    traditional_rmse = []

    for m in models:
        crypto = [all_results[a][m]['RMSE']
                  for a in ['BTC', 'ETH']
                  if a in all_results and m in all_results[a]]
        trad = [all_results[a][m]['RMSE']
                for a in ['SPX', 'VIX']
                if a in all_results and m in all_results[a]]

        crypto_rmse.append(np.mean(crypto) if crypto else 0)
        traditional_rmse.append(np.mean(trad) if trad else 0)

    ax.bar(x_pos - bar_width/2, crypto_rmse, bar_width,
           label='Crypto (BTC, ETH)', color='#ff7f0e')
    ax.bar(x_pos + bar_width/2, traditional_rmse, bar_width,
           label='Traditional (SPX, VIX)', color='#1f77b4')

    ax.set_ylabel('Average RMSE')
    ax.set_title('Model Performance: Crypto vs Traditional Assets')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/crypto_vs_traditional.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nFigures saved to {FIGURES_DIR}/")


def print_results_table(all_results):
    """Print results summary and generate LaTeX table."""

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    models = ['GARCH', 'EGARCH', 'RF', 'XGBoost', 'LSTM']

    # Console output
    header = f"{'Asset':<8}"
    for m in models:
        header += f"{m:<12}"
    print(f"\n{header}")
    print("-" * 68)

    for asset, results in all_results.items():
        row = f"{asset:<8}"
        for m in models:
            if m in results:
                row += f"{results[m]['RMSE']:<12.4f}"
            else:
                row += f"{'N/A':<12}"
        print(row)

    # LaTeX table
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Out-of-Sample Forecasting Performance (RMSE)}
\label{tab:results}
\begin{tabular}{lccccc}
\toprule
Asset & GARCH(1,1) & EGARCH & Random Forest & XGBoost & LSTM \\
\midrule
"""

    for asset, results in all_results.items():
        row = f"{asset} "
        for m in models:
            if m in results:
                row += f"& {results[m]['RMSE']:.4f} "
            else:
                row += "& -- "
        row += r"\\" + "\n"
        latex += row

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(f'{DATA_DIR}/results_table.tex', 'w') as f:
        f.write(latex)

    print(f"\nLaTeX table saved to {DATA_DIR}/results_table.tex")

    return latex


def generate_sample_data():
    """
    Generate synthetic price data when API is unavailable.
    Useful for testing and development.
    """
    dates = pd.date_range('2019-01-01', '2024-12-31', freq='D')
    datasets = {}

    for asset in ['BTC', 'ETH', 'SPX', 'VIX']:
        np.random.seed(SEED + hash(asset) % 100)

        # Crypto has higher volatility
        daily_vol = 0.05 if asset in ['BTC', 'ETH'] else 0.015

        returns = np.random.normal(0.0002, daily_vol, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'open': prices * 0.999,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.randint(1_000_000, 10_000_000, len(dates))
        }, index=dates)

        datasets[asset] = df
        df.to_csv(f"{DATA_DIR}/{asset}_data.csv")

    return datasets


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("="*60)
    print("VOLATILITY FORECASTING ANALYSIS")
    print("Comparing GARCH vs ML for Crypto and Traditional Assets")
    print("="*60)

    # Step 1: Collect data
    print("\n[1/4] Collecting data from Polygon API...")
    data = collect_all_assets()

    if len(data) == 0:
        print("No data collected from API. Using synthetic data...")
        data = generate_sample_data()

    # Step 2: Run analysis for each asset
    print("\n[2/4] Running model analysis...")
    all_results = {}
    all_predictions = {}
    all_actuals = {}

    for asset, df in data.items():
        try:
            results, predictions, actuals, processed_df = analyze_asset(df, asset)
            all_results[asset] = results
            all_predictions[asset] = predictions
            all_actuals[asset] = actuals
            data[asset] = processed_df
        except Exception as e:
            print(f"Error analyzing {asset}: {e}")

    # Step 3: Generate figures
    print("\n[3/4] Creating figures...")
    generate_figures(all_results, data)

    # Step 4: Create results tables
    print("\n[4/4] Creating results tables...")
    print_results_table(all_results)

    # Save complete results
    results_df = pd.DataFrame(all_results).T
    results_df.to_csv(f'{DATA_DIR}/full_results.csv')

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nData saved to: {DATA_DIR}/")
    print(f"Figures saved to: {FIGURES_DIR}/")
