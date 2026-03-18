import numpy as np
import matplotlib.pyplot as plt

from src.portfolio.risk_parity import risk_parity_weights

# Parameter ranges
correlations = np.linspace(0.0, 0.9, 10)
noise_levels = np.linspace(0.0, 0.05, 10)

instability = np.zeros((len(correlations), len(noise_levels)))

n_assets = 5
n_trials = 50

for i, rho in enumerate(correlations):
    for j, noise in enumerate(noise_levels):

        changes = []

        for _ in range(n_trials):

            # Build covariance matrix with constant correlation
            cov = np.full((n_assets, n_assets), rho)
            np.fill_diagonal(cov, 1)

            # Compute baseline weights
            w = risk_parity_weights(cov)

            # Add perturbation
            perturb = np.random.normal(0, noise, cov.shape)
            cov_perturbed = cov + perturb

            # Compute new weights
            w_hat = risk_parity_weights(cov_perturbed)

            # Measure instability
            change = np.linalg.norm(w_hat - w)
            changes.append(change)

        instability[i, j] = np.mean(changes)

# ---- Plot heatmap ----

plt.figure(figsize=(10,6))

im = plt.imshow(
    instability,
    origin='lower',
    aspect='auto',
    cmap='viridis'
)

plt.colorbar(im, label="Weight Instability ||ŵ − w||")

plt.title("Risk Parity Instability Heatmap", fontsize=16)

plt.xlabel("Covariance Noise (σ of perturbation)", fontsize=12)
plt.ylabel("Asset Correlation (ρ)", fontsize=12)

# Axis tick labels
plt.xticks(range(len(noise_levels)), np.round(noise_levels,3))
plt.yticks(range(len(correlations)), np.round(correlations,2))

# Flip y-axis so higher correlation appears higher
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig("results/risk_parity_instability_heatmap.png", dpi=300)
plt.show()