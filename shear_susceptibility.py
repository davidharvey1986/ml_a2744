'''
Shear susceptibility G1 as a smooth function of galaxy size and detection S/N.

pyRRG's calc_shear divides the ellipticities by a single G1 or by an arctan in S/N. On the JWST
A2744 catalogues G1 depends mainly on size (from ~1.45 at the 2.75 px cut to ~1.1 for large
faint galaxies) and, at fixed size, on S/N; the arctan fit also collapses to a constant for F115W.
Here G1 is the local-linear Gaussian-kernel regression of the per-galaxy g1_gal (pyRRG's
estimator) in the standardised (log gal_size, log snr) plane, with the two bandwidths chosen by
two-fold cross-validation, tabulated on a 60x60 grid and interpolated (clipped to the grid edges).

    correct_shears(filtered_name)   # after calc_shear: gamma = e / G1(size, snr) / rhodes_factor
'''
import numpy as np
from astropy.table import Table
from scipy.interpolate import RegularGridInterpolator

RHODES_FACTOR = 0.86            # pyRRG calc_shear default (not overridden by SHEAR_CUTS)
H_SIZE = (0.06, 0.1, 0.15, 0.22)
H_SNR = (0.12, 0.2, 0.3, 0.45)
NGRID = 60


def local_linear(X, y, Q, h):
    '''
    Local-linear Gaussian-kernel regression of y at the points Q (X, Q standardised; h one bandwidth
    per axis): a weighted linear fit around each Q, whose intercept is the estimate. Unlike the kernel
    mean it has no first-order bias at the edges of the sample or on slopes.
    '''
    out = np.empty(len(Q))
    for s in range(0, len(Q), 400):
        dx = (X[None, :, :] - Q[s:s+400, None, :]) / h            # (q, n, 2)
        w = np.exp(-0.5 * (dx ** 2).sum(-1))
        A = np.concatenate([np.ones_like(dx[..., :1]), dx], -1)   # (q, n, 3)
        AtW = A.transpose(0, 2, 1) * w[:, None, :]
        beta = np.linalg.solve(AtW @ A + 1e-9 * np.eye(3), AtW @ y)
        out[s:s+400] = beta[:, 0]
    return out


def fit_g1(cat, verbose=True, seed=0):
    '''G1(gal_size, snr) from a calc_shear catalogue (g1_gal, gal_size, snr) -> (callable, info).'''
    snr, size, g = (np.asarray(cat[k], dtype=float) for k in ("snr", "gal_size", "g1_gal"))
    ok = np.isfinite(snr * size * g) & (snr > 0) & (size > 0)
    X = np.c_[np.log10(size[ok]), np.log10(snr[ok])]
    mu, sd = X.mean(0), X.std(0)
    Xs, y = (X - mu) / sd, g[ok]
    half = np.random.default_rng(seed).random(len(y)) < 0.5
    cv = {(hz, hs): np.mean([np.mean((y[~a] - local_linear(Xs[a], y[a], Xs[~a], np.array([hz, hs]))) ** 2)
                             for a in (half, ~half)])
          for hz in H_SIZE for hs in H_SNR}
    h = min(cv, key=cv.get)
    gz = np.linspace(X[:, 0].min(), X[:, 0].max(), NGRID)
    gs = np.linspace(X[:, 1].min(), X[:, 1].max(), NGRID)
    Q = np.array(np.meshgrid(gz, gs, indexing="ij")).reshape(2, -1).T
    grid = local_linear(Xs, y, (Q - mu) / sd, np.array(h)).reshape(NGRID, NGRID)
    interp = RegularGridInterpolator((gz, gs), grid, bounds_error=False, fill_value=None)
    lo, hi = np.array([gz[0], gs[0]]), np.array([gz[-1], gs[-1]])
    if verbose:
        print(f"G1(size, snr): {ok.sum()} galaxies, bandwidths (size, snr) = {h} (std units), CV MSE {cv[h]:.5f}")
    return (lambda sz, sn: interp(np.clip(np.c_[np.log10(sz), np.log10(sn)], lo, hi)),
            dict(h=h, grid=grid, log_size=gz, log_snr=gs, cv=cv))


def correct_shears(filtered_name, verbose=True):
    '''Rewrite a calc_shear output with gamma = e / G1(size, snr) / RHODES_FACTOR; g1_model holds the new G1.'''
    t = Table.read(filtered_name)
    model, _ = fit_g1(t, verbose=verbose)
    G1 = model(np.asarray(t["gal_size"]), np.asarray(t["snr"]))
    if verbose:
        print(f"   {filtered_name}: G1 1/50/99% {np.percentile(G1, [1, 50, 99]).round(3)} "
              f"(pyRRG {np.percentile(t['g1_model'], [1, 50, 99]).round(3)})")
    t["gamma1"] = t["e1"] / G1 / RHODES_FACTOR
    t["gamma2"] = t["e2"] / G1 / RHODES_FACTOR
    t["g1_model"] = G1
    t.write(filtered_name, overwrite=True)
    return G1
