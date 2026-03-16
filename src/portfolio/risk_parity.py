import numpy as np
from scipy.optimize import minimize


def portfolio_volatility(w, cov):
    return np.sqrt(w @ cov @ w)


def risk_contributions(w, cov):

    port_vol = portfolio_volatility(w, cov)

    marginal = cov @ w

    return w * marginal / port_vol


def objective(w, cov):

    rc = risk_contributions(w, cov)

    target = np.mean(rc)

    return np.sum((rc - target) ** 2)


def risk_parity_weights(cov):

    n = cov.shape[0]

    init = np.ones(n) / n

    bounds = [(0, None)] * n

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    result = minimize(
        objective,
        init,
        args=(cov,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol":1e-12, "maxiter":2000}
    )

    return result.x