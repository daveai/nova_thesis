"""
Data Update Script - Fetch market data through end of 2025

This script downloads price data for crypto (BTC, ETH) and traditional
market assets (S&P 500, VIX) for use in volatility forecasting analysis.
"""

import os
import warnings
from datetime import datetime

import pandas as pd
import yfinance as yf
from polygon import RESTClient

warnings.filterwarnings('ignore')

# Configuration
API_KEY = "G8rIJ6VThkNC8skyqhmlPbHn_0jhTTM3"
DATA_DIR = "../data"

START_DATE = "2019-01-01"
END_DATE = "2025-12-31"


def fetch_crypto_from_polygon(ticker, start, end):
    """
    Download daily crypto prices from Polygon API.

    Returns a DataFrame indexed by date with OHLCV columns.
    """
    client = RESTClient(API_KEY)

    records = []
    for bar in client.list_aggs(ticker, 1, "day", start, end, limit=50000):
        records.append({
            'date': datetime.fromtimestamp(bar.timestamp / 1000),
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.set_index('date').sort_index()
    return df


def fetch_traditional_from_yahoo(ticker, start, end):
    """
    Download daily price data from Yahoo Finance.

    Returns a DataFrame with standardized lowercase column names.
    """
    df = yf.download(ticker, start=start, end=end, progress=False)

    # Handle multi-index columns that yfinance sometimes returns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Standardize column names
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    return df


def download_asset(name, ticker, source):
    """
    Download and save data for a single asset.

    Returns True if successful, False otherwise.
    """
    print(f"  Downloading {name}...")

    try:
        if source == 'polygon':
            df = fetch_crypto_from_polygon(ticker, START_DATE, END_DATE)
        else:
            df = fetch_traditional_from_yahoo(ticker, START_DATE, END_DATE)

        if df.empty:
            print(f"    Warning: No data returned for {name}")
            return False

        output_path = f"{DATA_DIR}/{name}_data.csv"
        df.to_csv(output_path)

        first_date = df.index[0].date()
        last_date = df.index[-1].date()
        print(f"    Saved {len(df)} days ({first_date} to {last_date})")
        return True

    except Exception as e:
        print(f"    Error: {e}")
        return False


def main():
    print("=" * 50)
    print("Updating market data through end of 2025")
    print("=" * 50)

    os.makedirs(DATA_DIR, exist_ok=True)

    # Define assets to download
    assets = [
        ('BTC', 'X:BTCUSD', 'polygon'),
        ('ETH', 'X:ETHUSD', 'polygon'),
        ('SPX', '^GSPC', 'yahoo'),
        ('VIX', '^VIX', 'yahoo'),
    ]

    print("\nCrypto assets (Polygon):")
    for name, ticker, source in assets[:2]:
        download_asset(name, ticker, source)

    print("\nTraditional assets (Yahoo Finance):")
    for name, ticker, source in assets[2:]:
        download_asset(name, ticker, source)

    print("\n" + "=" * 50)
    print("Data update complete")
    print("=" * 50)


if __name__ == "__main__":
    main()
