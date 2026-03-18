import numpy as np
import matplotlib.pyplot as plt

from src.portfolio.risk_parity import risk_parity_weights

instability = []
condition_numbers = []

for i in range(100):

    # random covariance matrix
    A = np.random.randn(6,6)
    cov = np.dot(A, A.T)

    # condition number
    kappa = np.linalg.cond(cov)

    # risk parity weights
    w = risk_parity_weights(cov)

    # perturb covariance
    noise = np.random.normal(0, 0.01, cov.shape)
    cov_perturbed = cov + noise

    w_hat = risk_parity_weights(cov_perturbed)

    change = np.linalg.norm(w_hat - w)

    instability.append(change)
    condition_numbers.append(kappa)

plt.scatter(condition_numbers, instability)
plt.xlabel("Condition Number κ(Σ)")
plt.ylabel("Weight Change ||ŵ - w||")
plt.title("Risk Parity Instability vs Covariance Conditioning")
plt.show()