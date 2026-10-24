"""
Weight simulated clusters so their projected masses follow the A2744 measurement.

Each cluster gets

    w(M) = N(log M; mu, sigma) / n(log M)

where N is a Gaussian in log10 M(<750 kpc) with the mean and 1-sigma width of the
A2744 substructure mass, and n is the density of the clusters being averaged
(a KDE of their masses). Dividing by n removes the simulation's own mass function,
so abundant low-mass clusters are not over-weighted and the weighted sample follows
the Gaussian whatever the simulation.
"""

import numpy as np
from scipy.stats import gaussian_kde, norm

# Core + NW + N substructure masses [1e13 Msun] and their (lower, upper) errors
A2744_MASSES = {
    "harvey": {"core": 16.0, "nw": 10.8, "n": 6.5},
    "jauzac": {"core": 27.7, "nw": 18.0, "n": 8.6},
}
A2744_ERRORS = {
    "harvey": {"core": (0.9, 0.6), "nw": (1.0, 0.3), "n": (0.9, 0.7)},
    "jauzac": {"core": (0.1, 0.1), "nw": (1.0, 1.0), "n": (2.2, 2.2)},
}


def a2744_log_mass(nsigma=2, study="harvey"):
    """log10 of the total A2744 substructure mass and its nsigma (lower, upper) bounds."""
    mass = A2744_MASSES[study]
    err = A2744_ERRORS[study]
    est = sum(mass.values())
    lo, hi = (np.sqrt(sum((err[k][i] / mass[k]) ** 2 for k in mass)) * est for i in (0, 1))
    return np.log10(est * 1e13), np.log10(np.array([est - nsigma * lo, est + nsigma * hi]) * 1e13)


def a2744_mass_gaussian(study="harvey"):
    """(mu, sigma) in log10 Msun: the measurement and the mean of its 1-sigma log errors."""
    mu, (lo, hi) = a2744_log_mass(nsigma=1, study=study)
    return mu, 0.5 * (hi - lo)


def gaussian_mass_weights(log_mass, mu, sigma):
    """
    Weights N(log_mass; mu, sigma) / n(log_mass) for one sample of clusters.

    The density n is estimated from log_mass itself, so pass exactly the clusters
    being averaged together (one simulation, one cross-section, after any other
    selection). Non-finite masses get zero weight.
    """
    log_mass = np.asarray(log_mass, dtype=float)
    weights = np.zeros(log_mass.shape)
    ok = np.isfinite(log_mass)
    if ok.sum() < 2:
        return weights
    density = gaussian_kde(log_mass[ok])(log_mass[ok])
    weights[ok] = norm.pdf(log_mass[ok], mu, sigma) / density
    return weights


def effective_sample_size(weights):
    """Kish effective number of clusters, sum(w)^2 / sum(w^2)."""
    weights = np.asarray(weights)
    return weights.sum() ** 2 / np.sum(weights ** 2)
