import yfinance as yf
import pandas as pd
from pathlib import Path

# Configuration
ASSETS = ["SPY", "TLT", "GLD", "QQQ", "IWM"]

START_DATE = "2010-01-01"
END_DATE = "2024-01-01"

OUTPUT_PATH = Path("data/raw/etf_prices.csv")

# Download Data
def download_prices():

    print("Downloading market data...\n")

    data = yf.download(
        ASSETS,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True  
    )

    print("Columns returned by yfinance:")
    print(data.columns)
    print()

    prices = data["Close"]

    return prices

# Save Data
def save_data(prices):

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    prices.to_csv(OUTPUT_PATH)

    print(f"\nData saved to: {OUTPUT_PATH}")
    print(f"Dataset shape: {prices.shape}")

# Main Execution
def main():

    prices = download_prices()

    print("\nPreview of downloaded data:\n")
    print(prices.head())

    save_data(prices)


if __name__ == "__main__":
    main()