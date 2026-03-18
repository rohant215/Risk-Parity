import numpy as np
import matplotlib.pyplot as plt

from src.portfolio.risk_parity import risk_parity_weights, risk_contributions


sigma1 = 0.20
sigma2 = 0.10

rho_values = np.linspace(-0.9, 0.9, 50)

w1_theoretical = sigma2 / (sigma1 + sigma2)
w2_theoretical = sigma1 / (sigma1 + sigma2)

print("THEORETICAL WEIGHTS")
print("w1:", w1_theoretical)
print("w2:", w2_theoretical)

numerical_w1 = []
numerical_w2 = []
rc1_list = []
rc2_list = []

for rho in rho_values:

    cov = np.array([
        [sigma1**2, rho * sigma1 * sigma2],
        [rho * sigma1 * sigma2, sigma2**2]
    ])

    weights = risk_parity_weights(cov)

    rc = risk_contributions(weights, cov)
    rc = rc / rc.sum()

    numerical_w1.append(weights[0])
    numerical_w2.append(weights[1])

    rc1_list.append(rc[0])
    rc2_list.append(rc[1])


numerical_w1 = np.array(numerical_w1)
numerical_w2 = np.array(numerical_w2)
rc1_list = np.array(rc1_list)
rc2_list = np.array(rc2_list)


print("\nNUMERICAL WEIGHTS (example)")
print("w1:", numerical_w1[0])
print("w2:", numerical_w2[0])

print("\nRISK CONTRIBUTIONS (example)")
print("RC1:", rc1_list[0])
print("RC2:", rc2_list[0])


plt.figure(figsize=(8,6))

plt.plot(rho_values, numerical_w1, label="Numerical w1")
plt.plot(rho_values, numerical_w2, label="Numerical w2")

plt.axhline(w1_theoretical, linestyle="--", label="Theoretical w1")
plt.axhline(w2_theoretical, linestyle="--", label="Theoretical w2")

plt.xlabel("Correlation (rho)")
plt.ylabel("Weights")

plt.title("Two-Asset Risk Parity: Theory vs Numerical Solution")

plt.legend()
plt.grid(True)

plt.show()


plt.figure(figsize=(8,6))

plt.plot(rho_values, rc1_list, label="RC1")
plt.plot(rho_values, rc2_list, label="RC2")

plt.axhline(0.5, linestyle="--")

plt.xlabel("Correlation (rho)")
plt.ylabel("Risk Contribution")

import numpy as np
import matplotlib.pyplot as plt

from src.portfolio.risk_parity import risk_parity_weights, risk_contributions


sigma1 = 0.20
sigma2 = 0.10

rho_values = np.linspace(-0.9, 0.9, 50)

w1_theoretical = sigma2 / (sigma1 + sigma2)
w2_theoretical = sigma1 / (sigma1 + sigma2)

print("THEORETICAL WEIGHTS")
print("w1:", w1_theoretical)
print("w2:", w2_theoretical)

numerical_w1 = []
numerical_w2 = []
rc1_list = []
rc2_list = []

for rho in rho_values:

    cov = np.array([
        [sigma1**2, rho * sigma1 * sigma2],
        [rho * sigma1 * sigma2, sigma2**2]
    ])

    weights = risk_parity_weights(cov)

    rc = risk_contributions(weights, cov)
    rc = rc / rc.sum()

    numerical_w1.append(weights[0])
    numerical_w2.append(weights[1])

    rc1_list.append(rc[0])
    rc2_list.append(rc[1])


numerical_w1 = np.array(numerical_w1)
numerical_w2 = np.array(numerical_w2)
rc1_list = np.array(rc1_list)
rc2_list = np.array(rc2_list)


print("\nNUMERICAL WEIGHTS (example)")
print("w1:", numerical_w1[0])
print("w2:", numerical_w2[0])

print("\nRISK CONTRIBUTIONS (example)")
print("RC1:", rc1_list[0])
print("RC2:", rc2_list[0])


plt.figure(figsize=(8,6))

plt.plot(rho_values, numerical_w1, label="Numerical w1")
plt.plot(rho_values, numerical_w2, label="Numerical w2")

plt.axhline(w1_theoretical, linestyle="--", label="Theoretical w1")
plt.axhline(w2_theoretical, linestyle="--", label="Theoretical w2")

plt.xlabel("Correlation (rho)")
plt.ylabel("Weights")

plt.title("Two-Asset Risk Parity: Theory vs Numerical Solution")

plt.legend()
plt.grid(True)

plt.show()


plt.figure(figsize=(8,6))

plt.plot(rho_values, rc1_list, label="RC1")
plt.plot(rho_values, rc2_list, label="RC2")

plt.axhline(0.5, linestyle="--")

plt.xlabel("Correlation (rho)")
plt.ylabel("Risk Contribution")

plt.title("Risk Contributions (Should be Equal)")

plt.legend()
plt.grid(True)

plt.show()

