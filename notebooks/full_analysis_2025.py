"""
Volatility Forecasting Analysis (2019-2025)

Compares econometric models (GARCH, EGARCH) with machine learning approaches
(Random Forest, XGBoost, LSTM) for forecasting realized volatility across
cryptocurrency (BTC, ETH) and traditional (SPX, VIX) assets.

Author: [Your Name]
Date: January 2025
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from xgboost import XGBRegressor
import tensorflow as tf

warnings.filterwarnings('ignore')

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Paths
DATA_DIR = "../data"
FIGURES_DIR = "../figures"
THESIS_FIGURES = "../thesis/Contents/Chapters/1_Main/figures"

os.makedirs(FIGURES_DIR, exist_ok=True)

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")


# -----------------------------------------------------------------------------
# Data Preparation
# -----------------------------------------------------------------------------

def compute_log_returns(prices):
    """Calculate log returns from price series."""
    df = prices.copy()
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    return df.dropna()


def compute_realized_volatility(returns, window=21):
    """
    Calculate realized volatility as rolling standard deviation of returns,
    annualized using sqrt(252) trading days.
    """
    return returns.rolling(window=window).std() * np.sqrt(252)


def build_feature_set(df, num_lags=22):
    """
    Construct feature matrix for ML models.

    Features include:
    - Lagged returns (1 to num_lags days)
    - Short/medium/long realized volatility (5, 10, 21 days)
    - Lagged volatility values

    Target: 5-day forward realized volatility
    """
    df = df.copy()

    # Lagged returns
    for lag in range(1, num_lags + 1):
        df[f'ret_lag_{lag}'] = df['returns'].shift(lag)

    # Realized volatility at different horizons
    df['rv_5'] = df['returns'].rolling(5).std() * np.sqrt(252)
    df['rv_10'] = df['returns'].rolling(10).std() * np.sqrt(252)
    df['rv_21'] = df['returns'].rolling(21).std() * np.sqrt(252)

    # Lagged volatility
    for lag in range(1, 6):
        df[f'rv_lag_{lag}'] = df['rv_21'].shift(lag)

    # Target: 5-day forward volatility
    df['target'] = df['returns'].shift(-1).rolling(5).std() * np.sqrt(252)
    df['target'] = df['target'].shift(-4)

    return df.dropna()


# -----------------------------------------------------------------------------
# Model Fitting Functions
# -----------------------------------------------------------------------------

def forecast_garch_rolling(returns, train_size, model_type='GARCH'):
    """
    Generate rolling one-step-ahead volatility forecasts using GARCH or EGARCH.

    Returns are scaled by 100 for numerical stability (arch package convention).
    Forecasts are converted back to annualized volatility.
    """
    forecasts = []
    scaled_returns = returns * 100

    for t in range(train_size, len(returns) - 1):
        train_window = scaled_returns.iloc[:t]

        try:
            if model_type == 'GARCH':
                model = arch_model(train_window, vol='Garch', p=1, q=1, dist='normal')
            else:
                model = arch_model(train_window, vol='EGARCH', p=1, q=1, dist='normal')

            fit = model.fit(disp='off', show_warning=False)
            forecast = fit.forecast(horizon=1)

            # Convert variance forecast to annualized volatility
            vol = np.sqrt(forecast.variance.values[-1, 0]) / 100 * np.sqrt(252)
            forecasts.append(vol)
        except Exception:
            forecasts.append(np.nan)

    return np.array(forecasts)


def fit_random_forest(X_train, y_train, X_test):
    """Train Random Forest regressor and generate predictions."""
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


def fit_xgboost(X_train, y_train, X_test):
    """Train XGBoost regressor and generate predictions."""
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
    """Convert feature matrix to sequences for LSTM input."""
    X_seq, y_seq = [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i - lookback:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


def fit_lstm(X_train, y_train, X_test, lookback=22):
    """
    Train a simple LSTM network for volatility forecasting.

    Architecture: LSTM(32) -> Dropout(0.2) -> Dense(16) -> Dense(1)
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
# Evaluation Metrics
# -----------------------------------------------------------------------------

def compute_error_metrics(actual, predicted):
    """Calculate RMSE, MAE, and MAPE between actual and predicted values."""
    valid = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[valid]
    predicted = predicted[valid]

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}


def diebold_mariano_test(actual, pred1, pred2, horizon=5):
    """
    Diebold-Mariano test for comparing predictive accuracy.

    Tests whether the forecast errors from pred1 and pred2 are significantly
    different. A positive DM statistic indicates pred2 is more accurate.
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


# -----------------------------------------------------------------------------
# Main Analysis
# -----------------------------------------------------------------------------

def analyze_asset(df, asset_name):
    """
    Run full volatility forecasting analysis for a single asset.

    Fits all models (GARCH, EGARCH, RF, XGBoost, LSTM) and returns
    performance metrics and predictions.
    """
    print(f"\n{'='*50}")
    print(f"Analyzing {asset_name}")
    print(f"{'='*50}")

    df = compute_log_returns(df)
    print(f"Observations: {len(df)}")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")

    # Prepare features and train/test split
    df_features = build_feature_set(df)
    train_size = int(len(df_features) * 0.8)

    # Identify feature columns (exclude price data and target)
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'adj_close', 'returns', 'target']
    feature_cols = [c for c in df_features.columns if c not in exclude_cols]

    X = df_features[feature_cols].values
    y = df_features['target'].values
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    results = {}
    predictions = {}

    # GARCH(1,1)
    print("Fitting GARCH(1,1)...")
    garch_train_size = int(len(df) * 0.8)
    garch_preds = forecast_garch_rolling(df['returns'], garch_train_size, 'GARCH')
    min_len = min(len(garch_preds), len(y_test))
    results['GARCH'] = compute_error_metrics(y_test[:min_len], garch_preds[:min_len])
    predictions['GARCH'] = garch_preds[:min_len]
    print(f"  RMSE: {results['GARCH']['RMSE']:.4f}")

    # EGARCH
    print("Fitting EGARCH...")
    egarch_preds = forecast_garch_rolling(df['returns'], garch_train_size, 'EGARCH')
    results['EGARCH'] = compute_error_metrics(y_test[:min_len], egarch_preds[:min_len])
    predictions['EGARCH'] = egarch_preds[:min_len]
    print(f"  RMSE: {results['EGARCH']['RMSE']:.4f}")

    # Random Forest
    print("Fitting Random Forest...")
    rf_model, rf_preds = fit_random_forest(X_train, y_train, X_test)
    results['RF'] = compute_error_metrics(y_test, rf_preds)
    predictions['RF'] = rf_preds[:min_len]
    print(f"  RMSE: {results['RF']['RMSE']:.4f}")

    # XGBoost
    print("Fitting XGBoost...")
    xgb_model, xgb_preds = fit_xgboost(X_train, y_train, X_test)
    results['XGBoost'] = compute_error_metrics(y_test, xgb_preds)
    predictions['XGBoost'] = xgb_preds[:min_len]
    print(f"  RMSE: {results['XGBoost']['RMSE']:.4f}")

    # LSTM
    print("Fitting LSTM...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lstm_model, lstm_preds, lookback = fit_lstm(X_train_scaled, y_train, X_test_scaled)

    # Align LSTM predictions (first `lookback` values are unavailable)
    lstm_preds_aligned = np.full(len(y_test), np.nan)
    if len(lstm_preds) > 0:
        end_idx = min(lookback + len(lstm_preds), len(y_test))
        lstm_preds_aligned[lookback:end_idx] = lstm_preds[:end_idx - lookback]

    results['LSTM'] = compute_error_metrics(y_test[lookback:lookback + len(lstm_preds)], lstm_preds)
    predictions['LSTM'] = lstm_preds_aligned[:min_len]
    print(f"  RMSE: {results['LSTM']['RMSE']:.4f}")

    return results, predictions, y_test[:min_len], df, rf_model


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

def generate_figures(all_results, all_data, all_predictions, all_actuals, rf_models):
    """Generate all figures for the thesis."""

    model_names = ['GARCH', 'EGARCH', 'RF', 'XGBoost', 'LSTM']
    model_colors = ['#2ecc71', '#27ae60', '#3498db', '#e74c3c', '#9b59b6']

    # Figure 1: Returns distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for idx, (asset, df) in enumerate(all_data.items()):
        ax = axes[idx // 2, idx % 2]
        returns = compute_log_returns(df.copy())['returns']

        ax.hist(returns, bins=50, density=True, alpha=0.7, edgecolor='black', color='steelblue')
        ax.set_title(f'{asset} Daily Returns Distribution (2019-2025)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Log Returns')
        ax.set_ylabel('Density')

        summary = (
            f"Mean: {returns.mean():.4f}\n"
            f"Std: {returns.std():.4f}\n"
            f"Skew: {returns.skew():.2f}\n"
            f"Kurt: {returns.kurtosis():.2f}"
        )
        ax.text(0.02, 0.98, summary, transform=ax.transAxes, verticalalignment='top',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/returns_distribution.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{THESIS_FIGURES}/returns_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 2: Volatility time series
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, (asset, df) in enumerate(all_data.items()):
        ax = axes[idx // 2, idx % 2]
        returns = compute_log_returns(df.copy())['returns']
        rv = compute_realized_volatility(returns)

        ax.plot(rv.index, rv.values, linewidth=0.8, color='darkblue')
        ax.fill_between(rv.index, 0, rv.values, alpha=0.3)
        ax.axhline(y=rv.mean(), color='red', linestyle='--', linewidth=1, label=f'Mean: {rv.mean():.2f}')
        ax.set_title(f'{asset} Realized Volatility (21-day, Annualized)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Annualized Volatility')
        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/volatility_timeseries.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{THESIS_FIGURES}/volatility_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 3: Model comparison (RMSE bar charts)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, (asset, results) in enumerate(all_results.items()):
        ax = axes[idx // 2, idx % 2]

        available_models = [m for m in model_names if m in results]
        rmse_values = [results[m]['RMSE'] for m in available_models]
        colors = model_colors[:len(available_models)]

        bars = ax.bar(available_models, rmse_values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_title(f'{asset} - Model Comparison (RMSE)', fontsize=12, fontweight='bold')
        ax.set_ylabel('RMSE')
        ax.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, val in zip(bars, rmse_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Highlight best model
        best_idx = rmse_values.index(min(rmse_values))
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{THESIS_FIGURES}/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 4: Crypto vs Traditional assets
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(model_names))
    bar_width = 0.35

    # Average RMSE for crypto (BTC, ETH) and traditional (SPX only, excluding VIX)
    crypto_rmse = []
    trad_rmse = []
    for model in model_names:
        crypto_vals = [all_results[a][model]['RMSE'] for a in ['BTC', 'ETH']
                       if a in all_results and model in all_results[a]]
        trad_vals = [all_results[a][model]['RMSE'] for a in ['SPX']
                     if a in all_results and model in all_results[a]]
        crypto_rmse.append(np.mean(crypto_vals) if crypto_vals else 0)
        trad_rmse.append(np.mean(trad_vals) if trad_vals else 0)

    ax.bar(x_pos - bar_width/2, crypto_rmse, bar_width, label='Crypto (BTC, ETH)', color='#f39c12', edgecolor='black')
    ax.bar(x_pos + bar_width/2, trad_rmse, bar_width, label='Traditional (SPX)', color='#3498db', edgecolor='black')
    ax.set_ylabel('Average RMSE', fontsize=11)
    ax.set_title('Model Performance: Crypto vs Traditional Assets', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/crypto_vs_traditional.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{THESIS_FIGURES}/crypto_vs_traditional.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 5: Price and volatility overlay
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    for idx, (asset, df) in enumerate(all_data.items()):
        ax = axes[idx]
        ax_vol = ax.twinx()

        returns = compute_log_returns(df.copy())['returns']
        rv = compute_realized_volatility(returns)

        ax.plot(df['close'].index, df['close'].values, 'b-', linewidth=0.8, label='Price')
        ax_vol.fill_between(rv.index, 0, rv.values, alpha=0.3, color='red', label='Volatility')

        ax.set_ylabel('Price', color='blue')
        ax_vol.set_ylabel('Volatility', color='red')
        ax.set_title(f'{asset}: Price and Realized Volatility (2019-2025)', fontsize=12, fontweight='bold')
        ax.tick_params(axis='y', labelcolor='blue')
        ax_vol.tick_params(axis='y', labelcolor='red')

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/price_volatility_overlay.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{THESIS_FIGURES}/price_volatility_overlay.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 6: Random Forest feature importance
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    feature_names = (
        [f'ret_lag_{i}' for i in range(1, 23)] +
        ['rv_5', 'rv_10', 'rv_21'] +
        [f'rv_lag_{i}' for i in range(1, 6)]
    )

    for idx, (asset, model) in enumerate(rf_models.items()):
        ax = axes[idx // 2, idx % 2]
        importances = model.feature_importances_

        # Show top 15 features
        top_indices = np.argsort(importances)[-15:]
        top_names = [feature_names[i] if i < len(feature_names) else f'feature_{i}' for i in top_indices]
        top_values = importances[top_indices]

        ax.barh(range(len(top_names)), top_values, color='steelblue', edgecolor='black')
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names)
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'{asset} - Random Forest Feature Importance (Top 15)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{THESIS_FIGURES}/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 7: Forecast vs actual comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, asset in enumerate(['BTC', 'ETH', 'SPX', 'VIX']):
        if asset not in all_predictions:
            continue

        ax = axes[idx // 2, idx % 2]
        actual = all_actuals[asset]

        # Use best-performing model for each asset type
        if asset in ['BTC', 'ETH']:
            best_model = 'XGBoost'
        else:
            best_model = 'RF'

        pred = all_predictions[asset][best_model]
        n_points = min(len(actual), len(pred), 100)

        ax.plot(range(n_points), actual[-n_points:], 'b-', linewidth=1, label='Actual', alpha=0.8)
        ax.plot(range(n_points), pred[-n_points:], 'r--', linewidth=1, label=f'{best_model} Forecast', alpha=0.8)
        ax.fill_between(range(n_points), actual[-n_points:], pred[-n_points:], alpha=0.2, color='gray')
        ax.set_title(f'{asset}: {best_model} Forecast vs Actual (Last 100 obs)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Observation')
        ax.set_ylabel('Volatility')
        ax.legend()

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/forecast_vs_actual.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{THESIS_FIGURES}/forecast_vs_actual.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 8: VIX-specific analysis
    if 'VIX' in all_data:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        vix_data = all_data['VIX']

        # VIX level distribution
        ax = axes[0, 0]
        ax.hist(vix_data['close'], bins=50, density=True, alpha=0.7, color='purple', edgecolor='black')
        ax.axvline(vix_data['close'].mean(), color='red', linestyle='--',
                   label=f"Mean: {vix_data['close'].mean():.1f}")
        ax.axvline(20, color='green', linestyle=':', label='VIX=20 threshold')
        ax.set_title('VIX Level Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('VIX Level')
        ax.legend()

        # VIX time series with fear threshold
        ax = axes[0, 1]
        ax.plot(vix_data.index, vix_data['close'], linewidth=0.8, color='purple')
        ax.axhline(20, color='green', linestyle=':', alpha=0.7)
        ax.fill_between(vix_data.index, 20, vix_data['close'],
                        where=vix_data['close'] > 20, alpha=0.3, color='red', label='Elevated (>20)')
        ax.set_title('VIX Time Series (2019-2025)', fontsize=12, fontweight='bold')
        ax.set_ylabel('VIX Level')
        ax.legend()

        # VIX vs SPX returns scatter
        ax = axes[1, 0]
        spx_data = all_data.get('SPX')
        if spx_data is not None:
            vix_returns = compute_log_returns(vix_data.copy())
            spx_returns = compute_log_returns(spx_data.copy())

            merged = pd.merge(
                vix_returns[['returns']], spx_returns[['returns']],
                left_index=True, right_index=True,
                suffixes=('_vix', '_spx')
            )

            ax.scatter(merged['returns_spx'], merged['returns_vix'], alpha=0.3, s=10)
            ax.set_xlabel('SPX Returns')
            ax.set_ylabel('VIX Returns')
            ax.set_title('VIX vs SPX Returns (Negative Correlation)', fontsize=12, fontweight='bold')

            correlation = merged['returns_spx'].corr(merged['returns_vix'])
            ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax.transAxes,
                    fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))

        # VIX model comparison
        ax = axes[1, 1]
        if 'VIX' in all_results:
            rmse_values = [all_results['VIX'][m]['RMSE'] for m in model_names]
            bars = ax.bar(model_names, rmse_values, color=model_colors, edgecolor='black')
            ax.set_title('VIX Volatility-of-Volatility Forecasting (RMSE)', fontsize=12, fontweight='bold')
            ax.set_ylabel('RMSE')

            for bar, val in zip(bars, rmse_values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(f'{FIGURES_DIR}/vix_analysis.png', dpi=150, bbox_inches='tight')
        plt.savefig(f'{THESIS_FIGURES}/vix_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\nFigures saved to {FIGURES_DIR}/ and {THESIS_FIGURES}/")


def run_dm_tests(all_results, all_predictions, all_actuals):
    """
    Run Diebold-Mariano tests comparing each model against GARCH baseline.
    """
    print("\n" + "="*60)
    print("DIEBOLD-MARIANO TEST RESULTS")
    print("="*60)

    comparisons = [('RF', 'GARCH'), ('XGBoost', 'GARCH'), ('EGARCH', 'GARCH'), ('LSTM', 'GARCH')]
    dm_results = {}

    for asset in all_predictions.keys():
        print(f"\n{asset}:")
        dm_results[asset] = {}
        actual = all_actuals[asset]

        for model, baseline in comparisons:
            pred_baseline = all_predictions[asset].get(baseline)
            pred_model = all_predictions[asset].get(model)

            if pred_baseline is not None and pred_model is not None:
                dm_stat, p_val = diebold_mariano_test(actual, pred_baseline, pred_model)

                # Significance stars
                if p_val < 0.01:
                    significance = '***'
                elif p_val < 0.05:
                    significance = '**'
                elif p_val < 0.1:
                    significance = '*'
                else:
                    significance = 'ns'

                dm_results[asset][(model, baseline)] = {'dm': dm_stat, 'p': p_val, 'stars': significance}
                print(f"  {model} vs {baseline}: DM={dm_stat:.3f}, p={p_val:.4f} {significance}")

    return dm_results


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("="*60)
    print("FULL VOLATILITY ANALYSIS (2019-2025)")
    print("="*60)

    # Load data for all assets
    all_data = {}
    assets = ['BTC', 'ETH', 'SPX', 'VIX']

    for asset in assets:
        try:
            df = pd.read_csv(f'{DATA_DIR}/{asset}_data.csv', index_col=0, parse_dates=True)
            all_data[asset] = df
            print(f"Loaded {asset}: {len(df)} observations")
        except Exception as e:
            print(f"Error loading {asset}: {e}")

    # Run analysis for each asset
    all_results = {}
    all_predictions = {}
    all_actuals = {}
    rf_models = {}

    for asset, df in all_data.items():
        results, predictions, actuals, processed_df, rf_model = analyze_asset(df, asset)
        all_results[asset] = results
        all_predictions[asset] = predictions
        all_actuals[asset] = actuals
        all_data[asset] = processed_df
        rf_models[asset] = rf_model

    # Statistical tests
    dm_results = run_dm_tests(all_results, all_predictions, all_actuals)

    # Generate figures
    print("\nCreating figures...")
    generate_figures(all_results, all_data, all_predictions, all_actuals, rf_models)

    # Print summary table
    print("\n" + "="*60)
    print("RESULTS SUMMARY (2019-2025)")
    print("="*60)

    model_names = ['GARCH', 'EGARCH', 'RF', 'XGBoost', 'LSTM']
    header = f"{'Asset':<8}" + "".join(f"{m:<12}" for m in model_names)
    print(f"\n{header}")
    print("-" * 68)

    for asset, results in all_results.items():
        row = f"{asset:<8}"
        for model in model_names:
            if model in results:
                row += f"{results[model]['RMSE']:<12.4f}"
        print(row)

    # Save results to CSV
    results_df = pd.DataFrame({
        asset: {model: metrics['RMSE'] for model, metrics in results.items()}
        for asset, results in all_results.items()
    }).T
    results_df.to_csv(f'{DATA_DIR}/results_2025.csv')

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
