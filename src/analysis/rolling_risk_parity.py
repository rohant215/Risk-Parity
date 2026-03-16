import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.portfolio.risk_parity import risk_parity_weights

returns = pd.read_csv(
    "data/processed/returns.csv",
    index_col=0,
    parse_dates=True
)

window = 252

dates = []
weights_list = []

for i in range(window, len(returns)):

    window_returns = returns.iloc[i-window:i]

    cov = window_returns.cov().values

    try:

        weights = risk_parity_weights(cov)

        weights_list.append(weights)
        dates.append(returns.index[i])

    except:
        pass

weights_df = pd.DataFrame(
    weights_list,
    index=dates,
    columns=returns.columns
)

turnover = weights_df.diff().abs().sum(axis=1)

plt.figure(figsize=(10,6))

for col in weights_df.columns:
    plt.plot(weights_df.index, weights_df[col], label=col)

plt.title("Rolling Risk Parity Weights")

plt.xlabel("Date")
plt.ylabel("Portfolio Weight")

plt.legend()

Path("results/figures").mkdir(parents=True, exist_ok=True)

plt.savefig("results/figures/rolling_risk_parity_weights.png")

plt.show()

plt.figure(figsize=(10,6))

plt.plot(turnover.index, turnover)

plt.title("Risk Parity Portfolio Turnover")

plt.xlabel("Date")
plt.ylabel("Total Weight Change")

plt.grid(True)

plt.savefig("results/figures/risk_parity_turnover.png")

plt.show()