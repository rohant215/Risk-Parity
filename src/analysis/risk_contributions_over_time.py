import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.portfolio.risk_parity import risk_parity_weights, risk_contributions

returns = pd.read_csv(
    "data/processed/returns.csv",
    index_col=0,
    parse_dates=True
)

window = 252

dates = []
rc_list = []

for i in range(window, len(returns)):

    window_returns = returns.iloc[i-window:i]
    cov = window_returns.cov().values

    try:

        weights = risk_parity_weights(cov)
        rc = risk_contributions(weights, cov)
        rc = rc / rc.sum()
        rc_list.append(rc)
        dates.append(returns.index[i])

    except:
        pass

rc_df = pd.DataFrame(
    rc_list,
    index=dates,
    columns=returns.columns
)

plt.figure(figsize=(10,6))

for col in rc_df.columns:
    plt.plot(rc_df.index, rc_df[col], label=col)

plt.axhline(1/len(rc_df.columns), linestyle="--")

plt.title("Risk Contributions Over Time (Risk Parity Verification)")

plt.xlabel("Date")
plt.ylabel("Fraction of Total Risk")

plt.legend()

Path("results/figures").mkdir(parents=True, exist_ok=True)

plt.savefig("results/figures/risk_contributions.png")

plt.show()