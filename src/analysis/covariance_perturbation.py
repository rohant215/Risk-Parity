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

noise_levels = np.linspace(0, 0.2, 20)

instability = []

for noise in noise_levels:

    changes = []

    for _ in range(200):

        noise_matrix = np.random.normal(
            0,
            noise * scale,
            cov.shape
        )

        noise_matrix = (noise_matrix + noise_matrix.T) / 2

        perturbed_cov = cov + noise_matrix

        np.fill_diagonal(
            perturbed_cov,
            np.maximum(np.diag(perturbed_cov), 1e-8)
        )

        try:

            new_weights = risk_parity_weights(perturbed_cov)

            change = np.linalg.norm(
                new_weights - baseline_weights
            )

            changes.append(change)

        except:
            pass

    instability.append(np.mean(changes))

plt.figure(figsize=(8,6))

plt.plot(
    noise_levels,
    instability,
    marker="o"
)

plt.xlabel("Covariance Noise Level (relative scale)")
plt.ylabel("Weight Instability")

plt.title("Risk Parity Sensitivity to Covariance Perturbations")

plt.grid(True)

Path("results/figures").mkdir(parents=True, exist_ok=True)

plt.savefig("results/figures/covariance_instability.png")

plt.show()