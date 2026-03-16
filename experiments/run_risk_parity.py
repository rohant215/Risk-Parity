import pandas as pd
from src.portfolio.risk_parity import risk_parity_weights
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.portfolio.risk_parity import risk_parity_weights

cov = pd.read_csv(
    "data/processed/cov_matrix.csv",
    index_col=0
).values

weights = risk_parity_weights(cov)

print("\nRisk Parity Weights:")
print(weights)

from src.portfolio.risk_parity import risk_contributions

rc = risk_contributions(weights, cov)

print("\nRisk Contributions:")
print(rc / rc.sum())