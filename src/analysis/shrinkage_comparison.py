import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.covariance import LedoitWolf

from src.portfolio.risk_parity import risk_parity_weights

returns = pd.read_csv(
    "data/processed/returns.csv",
    index_col=0,
    parse_dates=True
)

window = 252

dates = []

sample_weights = []
shrinkage_weights = []

for i in range(window, len(returns)):

    window_returns = returns.iloc[i-window:i]

    sample_cov = window_returns.cov().values

    lw = LedoitWolf().fit(window_returns.values)
    shrink_cov = lw.covariance_

    try:

        w_sample = risk_parity_weights(sample_cov)
        w_shrink = risk_parity_weights(shrink_cov)

        sample_weights.append(w_sample)
        shrinkage_weights.append(w_shrink)

        dates.append(returns.index[i])

    except:
        pass


sample_df = pd.DataFrame(
    sample_weights,
    index=dates,
    columns=returns.columns
)

shrink_df = pd.DataFrame(
    shrinkage_weights,
    index=dates,
    columns=returns.columns
)


sample_turnover = sample_df.diff().abs().sum(axis=1)
shrink_turnover = shrink_df.diff().abs().sum(axis=1)


plt.figure(figsize=(10,6))

plt.plot(sample_turnover.index, sample_turnover, label="Sample Covariance")
plt.plot(shrink_turnover.index, shrink_turnover, label="Ledoit-Wolf Shrinkage")

plt.title("Risk Parity Turnover: Sample vs Shrinkage Covariance")

plt.xlabel("Date")
plt.ylabel("Total Weight Change")

plt.legend()
plt.grid(True)

Path("results/figures").mkdir(parents=True, exist_ok=True)

plt.savefig("results/figures/shrinkage_vs_sample_turnover.png")

plt.show()