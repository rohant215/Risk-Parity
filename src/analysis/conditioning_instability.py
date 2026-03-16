import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.portfolio.risk_parity import risk_parity_weights

cov = pd.read_csv(
    "data/processed/cov_matrix.csv",
    index_col=0
).values

baseline_weights = risk_parity_weights(cov)

scale = np.mean(np.diag(cov))

condition_numbers = []
instability = []

for noise in np.linspace(0.01,0.25,60):

    noise_matrix = np.random.normal(0, noise * scale, cov.shape)

    noise_matrix = (noise_matrix + noise_matrix.T) / 2

    perturbed_cov = cov + noise_matrix

    eigvals, eigvecs = np.linalg.eigh(perturbed_cov)

    eigvals = np.maximum(eigvals, 1e-8)

    perturbed_cov = eigvecs @ np.diag(eigvals) @ eigvecs.T

    cond_number = np.max(eigvals) / np.min(eigvals)

    try:

        new_weights = risk_parity_weights(perturbed_cov)

        change = np.linalg.norm(new_weights - baseline_weights)

        condition_numbers.append(cond_number)
        instability.append(change)

    except:
        pass

condition_numbers = np.array(condition_numbers)
instability = np.array(instability)

log_cond = np.log(condition_numbers)

z = np.polyfit(log_cond, instability, 1)
p = np.poly1d(z)

x_line = np.linspace(log_cond.min(), log_cond.max(), 100)

plt.figure(figsize=(8,6))

plt.scatter(log_cond, instability)

plt.plot(x_line, p(x_line), color="red")

plt.xlabel("log(Condition Number)")
plt.ylabel("Weight Instability")

plt.title("Risk Parity Instability vs Covariance Conditioning")

plt.grid(True)

Path("results/figures").mkdir(parents=True, exist_ok=True)

plt.savefig("results/figures/conditioning_instability.png")

plt.show()