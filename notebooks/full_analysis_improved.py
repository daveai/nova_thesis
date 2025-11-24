"""
Volatility Forecasting: Econometric vs Machine Learning Models

Compares GARCH-family models with machine learning approaches (Random Forest,
XGBoost, LSTM) for forecasting realized volatility across cryptocurrency and
traditional equity markets.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from arch import arch_model
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Paths
DATA_DIR = "../data"
FIGURES_DIR = "../figures"
THESIS_FIGURES = "../thesis/Contents/Chapters/1_Main/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")


# =============================================================================
# Data Preparation
# =============================================================================

def load_data(filepath):
    """Load price data from CSV and set up datetime index."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower()
    if 'adj close' in df.columns:
        df.rename(columns={'adj close': 'adj_close'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df


def compute_log_returns(df):
    """Calculate log returns from closing prices."""
    df = df.copy()
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    return df.dropna()


def compute_realized_volatility(returns, window):
    """Annualized rolling standard deviation of returns."""
    return returns.rolling(window=window).std() * np.sqrt(252)


def build_feature_matrix(df, num_lags=22):
    """
    Construct features for ML models:
    - Lagged returns
    - Realized volatility at multiple horizons (5, 10, 21 days)
    - Lagged realized volatility
    - Forward-looking 5-day realized volatility as target
    """
    df = df.copy()

    # Lagged returns
    for lag in range(1, num_lags + 1):
        df[f'ret_lag_{lag}'] = df['returns'].shift(lag)

    # Realized volatility features
    df['rv_5'] = compute_realized_volatility(df['returns'], 5)
    df['rv_10'] = compute_realized_volatility(df['returns'], 10)
    df['rv_21'] = compute_realized_volatility(df['returns'], 21)

    # Lagged RV
    for lag in range(1, 6):
        df[f'rv_lag_{lag}'] = df['rv_21'].shift(lag)

    # Target: 5-day forward realized volatility
    df['target'] = df['returns'].shift(-1).rolling(5).std() * np.sqrt(252)
    df['target'] = df['target'].shift(-4)

    return df.dropna()


# =============================================================================
# Model Implementations
# =============================================================================

def forecast_garch_rolling(returns, train_size, model_type='GARCH'):
    """
    Rolling one-step-ahead volatility forecasts using GARCH or EGARCH.
    Returns are scaled by 100 for numerical stability in arch package.
    """
    forecasts = []
    returns_scaled = returns * 100

    for i in range(train_size, len(returns) - 1):
        train_data = returns_scaled.iloc[:i]
        try:
            if model_type == 'GARCH':
                model = arch_model(train_data, vol='Garch', p=1, q=1, dist='normal')
            else:
                model = arch_model(train_data, vol='EGARCH', p=1, q=1, dist='normal')

            result = model.fit(disp='off', show_warning=False)
            forecast = result.forecast(horizon=1)

            # Convert back to annualized volatility
            vol = np.sqrt(forecast.variance.values[-1, 0]) / 100 * np.sqrt(252)
            forecasts.append(vol)
        except Exception:
            forecasts.append(np.nan)

    return np.array(forecasts)


def fit_random_forest(X_train, y_train, X_test):
    """Train Random Forest and return model with predictions."""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, predictions


def fit_xgboost(X_train, y_train, X_test):
    """Train XGBoost and return model with predictions."""
    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, predictions


def create_sequences(X, y, lookback):
    """Convert feature matrix to sequences for LSTM input."""
    X_seq, y_seq = [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i - lookback:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


def fit_lstm(X_train, y_train, X_test, y_test, lookback=30):
    """
    Train LSTM for volatility forecasting.

    Architecture optimized for financial time series:
    - Shorter lookback (30 days) to preserve test samples
    - Two LSTM layers with moderate dropout
    - Batch normalization for training stability
    """
    # Create sequences
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, lookback)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, lookback)

    # Use 10% of training data for validation (preserve more for training)
    split_idx = int(len(X_train_seq) * 0.9)
    X_val, y_val = X_train_seq[split_idx:], y_train_seq[split_idx:]
    X_train_seq, y_train_seq = X_train_seq[:split_idx], y_train_seq[:split_idx]

    # Build model - optimized architecture
    model = Sequential([
        LSTM(64, input_shape=(lookback, X_train.shape[1]), return_sequences=True),
        BatchNormalization(),
        Dropout(0.15),
        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.15),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # Callbacks - more patience for convergence
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=25,
        restore_best_weights=True,
        min_delta=0.00001
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=0.00001
    )

    # Train with more epochs
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )

    predictions = model.predict(X_test_seq, verbose=0).flatten()

    print(f"  LSTM epochs trained: {len(history.history['loss'])}")
    print(f"  Final validation loss: {history.history['val_loss'][-1]:.6f}")

    return model, predictions, lookback


# =============================================================================
# Evaluation
# =============================================================================

def compute_metrics(actual, predicted):
    """Calculate RMSE, MAE, and MAPE, handling NaN values."""
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual, predicted = actual[mask], predicted[mask]

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}


def diebold_mariano_test(actual, pred1, pred2, horizon=5):
    """
    Diebold-Mariano test for comparing forecast accuracy.
    Tests whether pred1 significantly differs from pred2.
    Negative statistic means pred1 is more accurate.
    """
    errors1 = actual - pred1
    errors2 = actual - pred2
    loss_diff = errors1**2 - errors2**2
    loss_diff = loss_diff[~np.isnan(loss_diff)]

    n = len(loss_diff)
    if n < 10:
        return np.nan, np.nan

    mean_diff = np.mean(loss_diff)
    variance = np.var(loss_diff, ddof=1) / n

    if variance <= 0:
        return np.nan, np.nan

    dm_stat = mean_diff / np.sqrt(variance)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

    return dm_stat, p_value


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

def run_analysis(df, asset_name):
    """Run full volatility forecasting comparison for one asset."""
    print(f"\n{'=' * 50}")
    print(f"Analyzing {asset_name}")
    print(f"{'=' * 50}")

    # Prepare data
    df = compute_log_returns(df)
    print(f"Observations: {len(df)}")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")

    df_features = build_feature_matrix(df)
    train_size = int(len(df_features) * 0.8)

    # Separate features and target
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'adj_close', 'returns', 'target']
    feature_cols = [c for c in df_features.columns if c not in exclude_cols]

    X = df_features[feature_cols].values
    y = df_features['target'].values
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Scale features for LSTM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fit models
    print("Fitting GARCH(1,1)...")
    garch_preds = forecast_garch_rolling(df['returns'], train_size, 'GARCH')

    print("Fitting EGARCH...")
    egarch_preds = forecast_garch_rolling(df['returns'], train_size, 'EGARCH')

    print("Fitting Random Forest...")
    rf_model, rf_preds = fit_random_forest(X_train, y_train, X_test)

    print("Fitting XGBoost...")
    xgb_model, xgb_preds = fit_xgboost(X_train, y_train, X_test)

    print("Fitting LSTM...")
    lstm_model, lstm_preds, lookback = fit_lstm(
        X_train_scaled, y_train, X_test_scaled, y_test, lookback=30
    )

    # Align all predictions to same length
    min_len = min(len(y_test), len(garch_preds), len(rf_preds), len(lstm_preds) + lookback)
    y_actual = y_test[:min_len]
    garch_aligned = garch_preds[:min_len]
    egarch_aligned = egarch_preds[:min_len]
    rf_aligned = rf_preds[:min_len]
    xgb_aligned = xgb_preds[:min_len]

    # LSTM needs offset due to lookback window
    lstm_actual = y_test[lookback:lookback + len(lstm_preds)]

    # Calculate metrics
    results = {
        'GARCH': compute_metrics(y_actual, garch_aligned),
        'EGARCH': compute_metrics(y_actual, egarch_aligned),
        'RF': compute_metrics(y_actual, rf_aligned),
        'XGBoost': compute_metrics(y_actual, xgb_aligned),
        'LSTM': compute_metrics(lstm_actual, lstm_preds)
    }

    print(f"\nResults for {asset_name}:")
    print("-" * 40)
    for model_name, metrics in results.items():
        print(f"{model_name:10s} RMSE: {metrics['RMSE']:.4f}  "
              f"MAE: {metrics['MAE']:.4f}  MAPE: {metrics['MAPE']:.1f}%")

    # Diebold-Mariano tests vs GARCH baseline
    dm_results = {}
    for name, preds in [('RF', rf_aligned), ('XGBoost', xgb_aligned), ('EGARCH', egarch_aligned)]:
        stat, pval = diebold_mariano_test(y_actual, preds, garch_aligned)
        dm_results[f'{name} vs GARCH'] = (stat, pval)

    # LSTM vs GARCH (aligned to LSTM's window)
    garch_for_lstm = garch_preds[lookback:lookback + len(lstm_preds)]
    stat, pval = diebold_mariano_test(lstm_actual, lstm_preds, garch_for_lstm)
    dm_results['LSTM vs GARCH'] = (stat, pval)

    print(f"\nDiebold-Mariano Tests (vs GARCH baseline):")
    for test_name, (stat, pval) in dm_results.items():
        if pval < 0.01:
            sig = "***"
        elif pval < 0.05:
            sig = "**"
        elif pval < 0.10:
            sig = "*"
        else:
            sig = "ns"
        print(f"  {test_name}: DM={stat:.2f}, p={pval:.4f} {sig}")

    return {
        'results': results,
        'dm_tests': dm_results,
        'predictions': {
            'actual': y_actual,
            'lstm_actual': lstm_actual,
            'garch': garch_aligned,
            'egarch': egarch_aligned,
            'rf': rf_aligned,
            'xgb': xgb_aligned,
            'lstm': lstm_preds
        },
        'rf_model': rf_model,
        'feature_cols': feature_cols
    }


def print_summary_table(all_results, metric_name):
    """Print formatted results table for a given metric."""
    print(f"\n{'=' * 80}")
    print(f"SUMMARY: Out-of-Sample {metric_name}")
    print("=" * 80)
    print(f"{'Asset':<6} {'GARCH':>10} {'EGARCH':>10} {'RF':>10} {'XGBoost':>10} {'LSTM':>10}")
    print("-" * 80)

    for asset in ['BTC', 'ETH', 'SPX', 'VIX']:
        res = all_results[asset]['results']
        print(f"{asset:<6} {res['GARCH'][metric_name]:>10.4f} {res['EGARCH'][metric_name]:>10.4f} "
              f"{res['RF'][metric_name]:>10.4f} {res['XGBoost'][metric_name]:>10.4f} "
              f"{res['LSTM'][metric_name]:>10.4f}")


# =============================================================================
# Visualization
# =============================================================================

def plot_model_comparison(all_results, save_path):
    """Bar chart comparing RMSE across all models and assets."""
    fig, ax = plt.subplots(figsize=(12, 6))

    assets = ['BTC', 'ETH', 'SPX', 'VIX']
    models = ['GARCH', 'EGARCH', 'RF', 'XGBoost', 'LSTM']
    x = np.arange(len(assets))
    width = 0.15

    for i, model in enumerate(models):
        rmse_values = [all_results[asset]['results'][model]['RMSE'] for asset in assets]
        ax.bar(x + i * width, rmse_values, width, label=model)

    ax.set_ylabel('RMSE')
    ax.set_xlabel('Asset')
    ax.set_title('Out-of-Sample Forecasting Performance (RMSE)')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(assets)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_crypto_vs_traditional(all_results, save_path):
    """Compare average performance: crypto vs S&P 500."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    models = ['GARCH', 'EGARCH', 'RF', 'XGBoost', 'LSTM']

    # Cryptocurrency average (BTC + ETH)
    crypto_rmse = [
        np.mean([all_results['BTC']['results'][m]['RMSE'],
                 all_results['ETH']['results'][m]['RMSE']])
        for m in models
    ]
    axes[0].bar(models, crypto_rmse)
    axes[0].set_title('Cryptocurrency (BTC, ETH Average)')
    axes[0].set_ylabel('Average RMSE')
    axes[0].tick_params(axis='x', rotation=45)

    # Traditional (S&P 500)
    spx_rmse = [all_results['SPX']['results'][m]['RMSE'] for m in models]
    axes[1].bar(models, spx_rmse)
    axes[1].set_title('Traditional (S&P 500)')
    axes[1].set_ylabel('RMSE')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_importance(all_results, save_path):
    """Show top 10 Random Forest features for each asset."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, asset in enumerate(['BTC', 'ETH', 'SPX', 'VIX']):
        ax = axes[idx // 2, idx % 2]
        rf_model = all_results[asset]['rf_model']
        feature_cols = all_results[asset]['feature_cols']

        importances = rf_model.feature_importances_
        top_indices = np.argsort(importances)[-10:]
        top_features = [feature_cols[i] for i in top_indices]

        ax.barh(top_features, importances[top_indices])
        ax.set_title(f'{asset} - Top 10 Features')
        ax.set_xlabel('Importance')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_forecasts_vs_actual(all_results, save_path):
    """Plot actual vs best model forecast for last 100 observations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, asset in enumerate(['BTC', 'ETH', 'SPX', 'VIX']):
        ax = axes[idx // 2, idx % 2]
        preds = all_results[asset]['predictions']
        n = min(100, len(preds['actual']))

        # Plot actual volatility
        ax.plot(range(n), preds['actual'][-n:], label='Actual',
                linewidth=2, color='black')

        # Find and plot best model
        results = all_results[asset]['results']
        best_model = min(results.items(), key=lambda x: x[1]['RMSE'])[0]

        pred_key_map = {'GARCH': 'garch', 'EGARCH': 'egarch', 'RF': 'rf',
                        'XGBoost': 'xgb', 'LSTM': 'lstm'}
        best_preds = preds[pred_key_map[best_model]]

        ax.plot(range(n), best_preds[-n:], label=f'{best_model} (Best)',
                linewidth=1.5, alpha=0.8)

        ax.set_title(f'{asset} - Best Model: {best_model}')
        ax.set_xlabel('Observation')
        ax.set_ylabel('Volatility')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Run Analysis
# =============================================================================

if __name__ == "__main__":
    print("Loading data...")
    btc = load_data(f"{DATA_DIR}/BTC_data.csv")
    eth = load_data(f"{DATA_DIR}/ETH_data.csv")
    spx = load_data(f"{DATA_DIR}/SPX_data.csv")
    vix = load_data(f"{DATA_DIR}/VIX_data.csv")

    # Run analysis for each asset
    all_results = {
        'BTC': run_analysis(btc, 'Bitcoin (BTC)'),
        'ETH': run_analysis(eth, 'Ethereum (ETH)'),
        'SPX': run_analysis(spx, 'S&P 500 (SPX)'),
        'VIX': run_analysis(vix, 'VIX')
    }

    # Print summary tables
    print_summary_table(all_results, 'RMSE')
    print_summary_table(all_results, 'MAE')

    # Generate figures
    print("\nGenerating figures...")
    plot_model_comparison(all_results, f"{THESIS_FIGURES}/model_comparison.png")
    plot_crypto_vs_traditional(all_results, f"{THESIS_FIGURES}/crypto_vs_traditional.png")
    plot_feature_importance(all_results, f"{THESIS_FIGURES}/feature_importance.png")
    plot_forecasts_vs_actual(all_results, f"{THESIS_FIGURES}/forecast_vs_actual.png")

    print("\nAnalysis complete!")
    print(f"Figures saved to {THESIS_FIGURES}")
