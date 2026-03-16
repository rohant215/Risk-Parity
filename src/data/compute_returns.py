import pandas as pd
from pathlib import Path

# File paths
PRICE_PATH = Path("data/raw/etf_prices.csv")

RETURNS_PATH = Path("data/processed/returns.csv")
COV_PATH = Path("data/processed/cov_matrix.csv")

# Load price data
def load_prices():

    print("Loading price data...")

    prices = pd.read_csv(
        PRICE_PATH,
        index_col=0,
        parse_dates=True
    )

    return prices


# Compute returns
def compute_returns(prices):

    print("Computing daily returns...")

    returns = prices.pct_change().dropna()

    return returns

# Compute covariance matrix
def compute_covariance(returns):

    print("Computing covariance matrix...")

    cov_matrix = returns.cov()

    return cov_matrix

# Save outputs
def save_outputs(returns, cov_matrix):

    RETURNS_PATH.parent.mkdir(parents=True, exist_ok=True)

    returns.to_csv(RETURNS_PATH)
    cov_matrix.to_csv(COV_PATH)

    print("\nSaved files:")
    print(f"Returns → {RETURNS_PATH}")
    print(f"Covariance → {COV_PATH}")

# Main pipeline
def main():

    prices = load_prices()

    print("\nPreview of prices:")
    print(prices.head())

    returns = compute_returns(prices)

    print("\nPreview of returns:")
    print(returns.head())

    cov_matrix = compute_covariance(returns)

    print("\nCovariance matrix:")
    print(cov_matrix)

    save_outputs(returns, cov_matrix)


if __name__ == "__main__":
    main()