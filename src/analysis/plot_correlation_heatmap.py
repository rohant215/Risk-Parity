import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# File paths
RETURNS_PATH = Path("data/processed/returns.csv")
OUTPUT_PATH = Path("results/figures/correlation_heatmap.png")

# Load returns
def load_returns():

    returns = pd.read_csv(
        RETURNS_PATH,
        index_col=0,
        parse_dates=True
    )

    return returns

# Plot heatmap
def plot_heatmap(returns):
    corr_matrix = returns.corr()
    plt.figure(figsize=(8,6))

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    plt.title("Asset Correlation Matrix")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH)
    print(f"\nHeatmap saved to {OUTPUT_PATH}")
    plt.show()

# Main
def main():

    returns = load_returns()

    plot_heatmap(returns)

if __name__ == "__main__":
    main()