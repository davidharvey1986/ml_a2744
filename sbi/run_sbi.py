#!/usr/bin/env python
"""
Simulation-based inference of sigma/m, mass and merger state for A2744.

Inputs are the CNN summaries from make_summaries.py. For each simulation
(bahamas, darkskies) the parameters are

    theta = (log1p sigma/m, log10 M(<750 kpc), n_components)      n_comp capped at NCOMP_MAX

and two estimators are trained:

    NLE  p(x | theta), x = logit of the ensemble-mean P(SIDM) (1-d). The posterior is
         evaluated on a grid with flat priors in log1p sigma/m, log M and n_comp
         (or the A2744 mass measurement as the mass prior, --mass-prior a2744).
    NPE  p(theta | x), x = the 30 per-model logit P(SIDM) plus, with --latent-pcs k,
         the first k principal components of each model's 512-d latent space (PCA
         per model, fitted on the simulated clusters). n_comp is dequantised
         (+U(-1/2, 1/2)). Its prior is the simulated population (the sigma/m grid,
         each simulation's mass function and merger fractions).

Discrepancy with CDM, per simulation:

    empirical   where A2744 sits in the distribution of CDM clusters that are merging
                (n_comp > 1) and weighted to the A2744 mass (mass_weighting.py):
                one-sided p-value and Gaussian-equivalent significance
    Bayes factor SIDM:CDM from each posterior, CDM := sigma/m < half the smallest
                simulated SIDM cross-section (posterior odds / prior odds); for the NLE
                also the Savage-Dickey ratio at sigma/m = 0 and the likelihood ratio
                of the best sigma/m to CDM.

Checks: NLE recovery from 10 stacked held-out clusters with known nuisance, NPE
simulation-based calibration (rank ECDFs) on held-out clusters, whether A2744 lies
inside the simulated summary distribution, and how much lower the CNN outputs are
for clusters a model was trained on.

Outputs in sbi/results/<summaries dir name>[_pc<k>]/: results.json, posteriors.npz,
posterior.pdf/png, discrepancy.pdf/png, validation.pdf/png and the trained
estimators (*.pt, not in git).

Usage (needs the sbi package, e.g. the Dark_Matter_DA/sbi environment):
    cd ml_check
    ../Dark_Matter_DA/sbi/bin/python sbi/run_sbi.py --summaries sbi/summaries_ta
    ../Dark_Matter_DA/sbi/bin/python sbi/run_sbi.py --summaries sbi/summaries_ta --latent-pcs 0   # softmax only
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import logsumexp
from scipy.stats import gaussian_kde, norm

from sbi.inference import NLE, NPE
from sbi.neural_nets import likelihood_nn, posterior_nn
from sbi.utils import BoxUniform

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from mass_weighting import a2744_mass_gaussian, effective_sample_size, gaussian_mass_weights  # noqa: E402

DOMAINS = ["bahamas", "darkskies"]
NCOMP_MAX = 6          # n_comp >= 6 grouped together
EPS = 1e-4             # clip on P(SIDM) before the logit
TEST_FRAC = 0.2
N_STACK = 10
N_POST = 20_000
N_SBC = 300

INK, MUTED = "#222222", "#6b6b6b"
NLE_COLOUR, NPE_COLOUR, PRIOR_COLOUR = "#222222", "#D55E00", "#8a8a8a"
RAMP = {"bahamas": plt.cm.Blues, "darkskies": plt.cm.Oranges}
trapezoid = getattr(np, "trapezoid", None) or np.trapz


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summaries", default=str(REPO / "sbi" / "summaries_ta"))
    p.add_argument("--out-dir", default=None, help="default sbi/results/<summaries dir name>")
    p.add_argument("--mass-prior", choices=["flat", "a2744"], default="flat",
                   help="NLE mass prior: flat over the simulations or the A2744 measurement")
    p.add_argument("--noise", default="0",
                   help="noise realisation of the summaries, or 'all' to train on every cluster-draw "
                        "pair (split by cluster; checks use one draw per cluster)")
    p.add_argument("--latent-pcs", type=int, default=0,
                   help="principal components of each model's latent space added to the NPE input")
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p)).astype(np.float32)


# ── data ───────────────────────────────────────────────────────────────────────
def latent_features(S, d, k, noise, rows):
    '''First k principal components of each model's latents, for the simulations and A2744.'''
    Z = np.load(S / f"latents_{d}.npy", mmap_mode="r")[:, noise]        # (n_models, N, 512)
    Z_obs = np.load(S / "obs.npz")["latents"][:, 0]                     # (n_models, 512)
    sims, obs, explained = [], [], []
    for m in range(Z.shape[0]):
        z = np.asarray(Z[m])[rows]
        mean = z.mean(0)
        _, sv, vt = np.linalg.svd(z - mean, full_matrices=False)
        sims.append((z - mean) @ vt[:k].T)
        obs.append((Z_obs[m] - mean) @ vt[:k].T)
        explained.append(np.sum(sv[:k]**2) / np.sum(sv**2))
    return (np.concatenate(sims, axis=1).astype(np.float32), np.concatenate(obs).astype(np.float32),
            float(np.mean(explained)))


def load_domain(summaries, d, noise, latent_pcs=0):
    '''
    Rows are cluster-draw pairs: one per cluster for a single noise realisation, or
    every realisation stacked (noise="all"), with "cluster" and "draw" indices.
    '''
    S = Path(summaries)
    P = np.load(S / f"params_{d}.npz")
    soft_all = np.load(S / f"softmax_{d}.npy")                         # (n_models, n_noise, N)
    draws = list(range(soft_all.shape[1])) if noise == "all" else [int(noise)]
    obs = np.load(S / "obs.npz")["softmax"][:, 0]                      # (n_models,)

    ok = np.isfinite(P["log10_mass_750"])
    n_ok = int(ok.sum())
    soft = np.concatenate([soft_all[:, r][:, ok] for r in draws], axis=1)   # (n_models, R * N)
    rep = len(draws)
    sigma = np.tile(P["cross_section"][ok], rep)
    theta = np.tile(np.c_[np.log1p(P["cross_section"][ok]), P["log10_mass_750"][ok],
                          np.minimum(P["ncomponents"][ok], NCOMP_MAX)], (rep, 1)).astype(np.float32)
    held_out = np.tile(P["held_out"][:, ok], (1, rep))
    x_npe, x_obs_npe = logit(soft.T), logit(obs)
    explained = 0.
    if latent_pcs > 0:
        if rep > 1:
            raise ValueError("--latent-pcs needs a single noise realisation")
        pcs, pcs_obs, explained = latent_features(S, d, latent_pcs, draws[0], ok)
        x_npe, x_obs_npe = np.c_[x_npe, pcs], np.r_[x_obs_npe, pcs_obs]
    return {
        "cluster": np.tile(np.arange(n_ok), rep),
        "draw": np.repeat(np.arange(rep), n_ok),
        "n_clusters": n_ok,
        "n_draws": rep,
        "x_npe": x_npe,                                                # (N, n_models * (1 + k))
        "x_obs_npe": x_obs_npe,
        "latent_variance_explained": explained,
        "theta": theta,
        "sigma": sigma,
        "x_ens": logit(soft.mean(0))[:, None],                         # (rows, 1)
        "x_models": logit(soft.T),                                     # (rows, n_models)
        "x_obs_ens": logit(np.array([obs.mean()])),                    # (1,)
        "x_obs_models": logit(obs),                                    # (n_models,)
        "held_out": held_out,
        "logit_models": logit(soft),                                   # (n_models, rows)
        "sigma_grid": np.unique(sigma),
        "n_dropped": int(np.sum(~ok)),
    }


def cdm_threshold(sigma_grid):
    '''CDM := sigma/m below half the smallest simulated SIDM cross-section.'''
    return 0.5 * sigma_grid[sigma_grid > 0].min()


def sigma_cells(sigma_grid):
    '''Edges (in sigma/m) of the cells around each simulated cross-section.'''
    mid = 0.5 * (sigma_grid[1:] + sigma_grid[:-1])
    return np.r_[-np.inf, mid, np.inf]


def snap_sigma(log1p_sigma, sigma_grid):
    '''Nearest simulated cross-section (cell membership) for log1p sigma/m values.'''
    idx = np.digitize(np.expm1(log1p_sigma), sigma_cells(sigma_grid)) - 1
    return sigma_grid[np.clip(idx, 0, len(sigma_grid) - 1)]


def discrete_quantiles(values, probs, q=(0.16, 0.5, 0.84)):
    c = np.cumsum(probs) / np.sum(probs)
    return [float(values[np.searchsorted(c, qi)]) for qi in q]


def bounds(D):
    th = D["theta"]
    lo = np.array([0., th[:, 1].min(), 0.5])
    hi = np.array([th[:, 0].max(), th[:, 1].max(), NCOMP_MAX + 0.5])
    pad = np.array([1e-3, 1e-3, 0.])
    return lo - pad, hi + pad


# ── empirical CDM test ─────────────────────────────────────────────────────────
def empirical_test(D, mu, sig):
    '''Weighted p-value of the observed summary under each simulated sigma/m.'''
    th, x, x_obs = D["theta"], D["x_ens"][:, 0], D["x_obs_ens"][0]
    merging = th[:, 2] > 1
    out = {}
    for s in D["sigma_grid"]:
        sel = np.isclose(D["sigma"], s) & merging
        w = gaussian_mass_weights(th[sel, 1], mu, sig)
        p = float(np.sum(w * (x[sel] >= x_obs)) / w.sum())
        n_eff = effective_sample_size(w) / D["n_draws"]          # clusters, not cluster-draw pairs
        out[f"{s:g}"] = {"p_ge_obs": p, "n_eff": float(n_eff), "x": x[sel], "w": w}
    cdm = out["0"]
    p_floor = 1. / cdm["n_eff"]
    p = max(cdm["p_ge_obs"], p_floor)
    raw = np.mean(x[D["sigma"] == 0] >= x_obs)
    return out, {
        "p_value": cdm["p_ge_obs"],
        "p_value_floor": p_floor,
        "significance_sigma": float(norm.isf(p)),
        "is_lower_limit": bool(cdm["p_ge_obs"] < p_floor),
        "n_eff": cdm["n_eff"],
        "p_value_all_cdm_unweighted": float(raw),
    }


def training_bias(D):
    '''Mean logit P(SIDM) of CDM clusters a model trained on minus those it did not.'''
    cdm = D["sigma"] == 0
    lg, ho = D["logit_models"][:, cdm], D["held_out"][:, cdm]
    return float(lg[~ho].mean() - lg[ho].mean())


# ── NLE ────────────────────────────────────────────────────────────────────────
def train_nle(theta, x, prior):
    inf = NLE(prior=prior, density_estimator=likelihood_nn(model="mdn", num_components=6, hidden_features=64),
              show_progress_bars=False)
    inf.append_simulations(torch.tensor(theta), torch.tensor(x))
    return inf.train(training_batch_size=512, learning_rate=5e-4, stop_after_epochs=20)


@torch.no_grad()
def nle_logp(est, x, cond, chunk=50_000):
    '''log p(x | cond) for a scalar x and conditions (K, 3).'''
    out = []
    for i in range(0, len(cond), chunk):
        c = torch.tensor(cond[i:i + chunk], dtype=torch.float32)
        xi = torch.full((1, len(c), 1), float(x))
        out.append(est.log_prob(xi, condition=c).numpy().ravel())
    return np.concatenate(out)


def nle_grids(D, n_sigma=200, n_mass=60):
    lo, hi = bounds(D)
    mass = np.percentile(D["theta"][:, 1], [1, 99])
    return (np.linspace(0, D["theta"][:, 0].max(), n_sigma).astype(np.float32),
            np.linspace(*mass, n_mass).astype(np.float32),
            np.arange(1, NCOMP_MAX + 1, dtype=np.float32))


def nle_posterior(est, x_obs, grids, mass_prior):
    '''Joint posterior on the (sigma, mass, ncomp) grid plus the marginal likelihood in sigma.'''
    gs, gm, gn = grids
    S, M, K = np.meshgrid(gs, gm, gn, indexing="ij")
    loglike = nle_logp(est, x_obs, np.c_[S.ravel(), M.ravel(), K.ravel()]).reshape(S.shape)
    log_prior = np.zeros_like(loglike) if mass_prior is None else norm.logpdf(M, *mass_prior)
    log_post = loglike + log_prior
    post = np.exp(log_post - log_post.max())
    post /= post.sum()
    # marginal likelihood of sigma, averaging the nuisance over its prior
    w = log_prior - logsumexp(log_prior[0])
    log_like_sigma = logsumexp(loglike + w, axis=(1, 2))
    return post, log_like_sigma


def nle_cell_probs(gs, p_sigma, sigma_grid):
    '''Posterior (and flat-prior) probability in the cell around each simulated sigma/m.'''
    cell = np.digitize(np.expm1(gs), sigma_cells(sigma_grid)) - 1
    post = np.array([p_sigma[cell == i].sum() for i in range(len(sigma_grid))])
    prior = np.array([np.mean(cell == i) for i in range(len(sigma_grid))])
    return post / post.sum(), prior / prior.sum()


def nle_summary(post, log_like_sigma, grids, sigma_grid):
    gs = grids[0]
    p_sigma = post.sum((1, 2))
    dens = p_sigma / np.gradient(gs)                              # density in log1p sigma
    prior_dens = 1. / gs[-1]
    cells, prior_cells = nle_cell_probs(gs, p_sigma, sigma_grid)
    p_cdm, prior_cdm = float(cells[0]), float(prior_cells[0])
    lr = float(np.exp(log_like_sigma.max() - log_like_sigma[0]))
    return {
        "sigma_quantiles": quantiles(np.expm1(gs), p_sigma),
        "sigma_probabilities": {f"{v:g}": float(p) for v, p in zip(sigma_grid, cells)},
        "posterior_p_cdm": p_cdm,
        "prior_p_cdm": prior_cdm,
        "bayes_factor_sidm_cdm": float(((1 - p_cdm) / p_cdm) / ((1 - prior_cdm) / prior_cdm)),
        "savage_dickey_bf_sidm_cdm": float(prior_dens / dens[0]),
        "likelihood_ratio_best_vs_cdm": lr,
        "lr_significance_sigma": float(np.sqrt(2 * np.log(lr))) if lr > 1 else 0.,
        "best_sigma": float(np.expm1(gs[np.argmax(log_like_sigma)])),
        "mass_quantiles": quantiles(grids[1], post.sum((0, 2))),
        "ncomp_probabilities": {str(int(k)): float(v) for k, v in zip(grids[2], post.sum((0, 1)))},
    }


def nle_recovery(est, th_te, x_te, gs, rng, n_rep=15):
    rec = []
    for s in np.unique(th_te[:, 0]):
        pool = np.where(np.isclose(th_te[:, 0], s))[0]
        for _ in range(n_rep):
            pick = rng.choice(pool, N_STACK, replace=False)
            ll = sum(nle_logp(est, x_te[i, 0], np.c_[gs, np.tile(th_te[i, 1:], (len(gs), 1))]) for i in pick)
            p = np.exp(ll - ll.max())
            rec.append([np.expm1(s), *quantiles(np.expm1(gs), p)])
    return np.array(rec)


# ── NPE ────────────────────────────────────────────────────────────────────────
def dequantise(theta, rng):
    th = theta.copy()
    th[:, 2] += rng.uniform(-0.5, 0.5, len(th)).astype(np.float32)
    return th


def train_npe(theta, x, prior):
    inf = NPE(prior=prior, density_estimator=posterior_nn(model="maf", hidden_features=64, num_transforms=5),
              show_progress_bars=False)
    inf.append_simulations(torch.tensor(theta), torch.tensor(x))
    est = inf.train(training_batch_size=256, learning_rate=5e-4, stop_after_epochs=20)
    return inf.build_posterior(est)


def npe_summary(samples, sigma_grid, theta_train):
    s, m, n = samples.T
    snapped = snap_sigma(s, sigma_grid)
    probs = np.array([np.mean(snapped == v) for v in sigma_grid])
    p_cdm = float(probs[0])
    prior_cdm = float(np.mean(np.isclose(theta_train[:, 0], 0)))
    nc = np.clip(np.round(n), 1, NCOMP_MAX).astype(int)
    floor = 1. / len(samples)
    return {
        "sigma_quantiles": discrete_quantiles(sigma_grid, probs),
        "sigma_probabilities": {f"{v:g}": float(p) for v, p in zip(sigma_grid, probs)},
        "posterior_p_cdm": p_cdm,
        "prior_p_cdm": prior_cdm,
        "bayes_factor_sidm_cdm": float(((1 - p_cdm) / max(p_cdm, floor)) / ((1 - prior_cdm) / prior_cdm)),
        "bayes_factor_is_lower_limit": bool(p_cdm < floor),
        "mass_quantiles": list(np.quantile(m, [0.16, 0.5, 0.84])),
        "ncomp_probabilities": {str(k): float(np.mean(nc == k)) for k in range(1, NCOMP_MAX + 1)},
    }


def npe_sbc(posterior, th_te, x_te, sigma_grid, rng, n_samples=500):
    '''
    Rank of the true value among posterior samples for held-out clusters. sigma/m
    and n_comp are discrete, so samples are snapped to the simulated grid / rounded
    and ties are broken at random (randomised PIT); mass is continuous.
    '''
    pick = rng.choice(len(th_te), min(N_SBC, len(th_te)), replace=False)
    ranks = []
    for i in pick:
        smp = posterior.sample((n_samples,), x=torch.tensor(x_te[i][None]), show_progress_bars=False).numpy()
        true_sigma = np.expm1(th_te[i, 0])
        disc = [(snap_sigma(smp[:, 0], sigma_grid), sigma_grid[np.argmin(abs(sigma_grid - true_sigma))]),
                (np.clip(np.round(smp[:, 2]), 1, NCOMP_MAX), th_te[i, 2])]
        r_sigma, r_ncomp = (np.mean(b < t) + rng.random() * np.mean(b == t) for b, t in disc)
        ranks.append([r_sigma, np.mean(smp[:, 1] < th_te[i, 1]), r_ncomp])
    return np.array(ranks)


def npe_test_metrics(posterior, th_te, x_te, sigma_grid, rng, n_clusters=1000, n_samples=400):
    '''
    How much a single cluster's posterior learns: probability on the true sigma/m
    (chance = 1 / number of simulated values), how often the most probable sigma/m is
    right, and the same for n_comp; mass as posterior std and error of the median.
    '''
    pick = rng.choice(len(th_te), min(n_clusters, len(th_te)), replace=False)
    smp = posterior.sample_batched((n_samples,), x=torch.tensor(x_te[pick]),
                                   show_progress_bars=False).numpy()      # (S, B, 3)
    true_sigma = np.expm1(th_te[pick, 0])
    snapped = snap_sigma(smp[..., 0], sigma_grid)
    probs = np.stack([np.mean(snapped == v, axis=0) for v in sigma_grid], axis=1)   # (B, n_sigma)
    true_idx = np.array([np.argmin(abs(sigma_grid - t)) for t in true_sigma])
    nc = np.clip(np.round(smp[..., 2]), 1, NCOMP_MAX)
    nc_probs = np.stack([np.mean(nc == k, axis=0) for k in range(1, NCOMP_MAX + 1)], axis=1)
    nc_true = th_te[pick, 2].astype(int) - 1
    rows = np.arange(len(pick))
    return {
        "n_clusters": int(len(pick)),
        "sigma_p_true": float(probs[rows, true_idx].mean()),
        "sigma_p_true_chance": float(1. / len(sigma_grid)),
        "sigma_map_correct": float(np.mean(probs.argmax(1) == true_idx)),
        "sigma_map_correct_by_sigma": {f"{v:g}": float(np.mean(probs.argmax(1)[true_idx == i] == i))
                                       for i, v in enumerate(sigma_grid)},
        "mass_posterior_std": float(np.mean(smp[..., 1].std(0))),
        "mass_prior_std": float(th_te[:, 1].std()),
        "mass_median_rms_error": float(np.sqrt(np.mean((np.median(smp[..., 1], 0) - th_te[pick, 1])**2))),
        "ncomp_p_true": float(nc_probs[rows, nc_true].mean()),
        "ncomp_map_correct": float(np.mean(nc_probs.argmax(1) == nc_true)),
    }


def misspecification(x_train, x_obs):
    '''Fraction of simulated summaries further from the mean (Mahalanobis) than A2744.'''
    mu = x_train.mean(0)
    icov = np.linalg.pinv(np.cov(x_train.T))
    d2 = np.einsum("ij,jk,ik->i", x_train - mu, icov, x_train - mu)
    d2_obs = (x_obs - mu) @ icov @ (x_obs - mu)
    return float(np.mean(d2 >= d2_obs))


def quantiles(values, weights, q=(0.16, 0.5, 0.84)):
    c = np.cumsum(weights)
    c /= c[-1]
    return [float(v) for v in np.interp(q, c, values)]


# ── plots ──────────────────────────────────────────────────────────────────────
def setup_style():
    plt.rcParams.update({
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.titlesize": 11,
        "axes.labelsize": 11, "legend.frameon": False, "font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot_posterior(R, mu, sig, out):
    fig, axes = plt.subplots(len(R), 4, figsize=(16, 4.2 * len(R)), constrained_layout=True, squeeze=False)
    for row, (d, r) in zip(axes, R.items()):
        gs, gm, gn = r["grids"]
        post = r["nle_post"]
        smp = r["npe_samples"]
        sig_x = np.expm1(gs)

        ax = row[0]
        grid_vals = r["sigma_grid"]
        res = r["results"]
        k = np.arange(len(grid_vals))
        nle_p = [res["nle"]["sigma_probabilities"][f"{v:g}"] for v in grid_vals]
        npe_p = [res["npe"]["sigma_probabilities"][f"{v:g}"] for v in grid_vals]
        _, nle_prior = nle_cell_probs(gs, post.sum((1, 2)), grid_vals)
        ax.bar(k - 0.2, nle_p, width=0.38, color=NLE_COLOUR, label="NLE")
        ax.bar(k + 0.2, npe_p, width=0.38, color=NPE_COLOUR, label="NPE")
        ax.plot(k - 0.2, nle_prior, "_", ms=22, mew=2, color=PRIOR_COLOUR, label="prior")
        ax.plot(k + 0.2, [res["npe"]["prior_p_cdm"]] * len(k), "_", ms=22, mew=2, color=PRIOR_COLOUR)
        ax.set_xticks(k)
        ax.set_xticklabels(["CDM"] + [f"{v:g}" for v in grid_vals[1:]])
        ax.set_xlabel(r"$\sigma/m$ [cm$^2$ g$^{-1}$] (nearest simulated)")
        ax.set_ylabel("Posterior probability")
        ax.set_title(f"{d.upper()}: cross-section", color=INK)
        ax.legend(loc="upper left", fontsize=8)

        ax = row[1]
        ax.plot(gm, post.sum((0, 2)) / trapezoid(post.sum((0, 2)), gm), color=NLE_COLOUR, lw=2, label="NLE")
        ax.plot(gm, gaussian_kde(smp[:, 1])(gm), color=NPE_COLOUR, lw=2, ls="--", label="NPE")
        ax.plot(gm, norm.pdf(gm, mu, sig), color=PRIOR_COLOUR, lw=1.5, ls=":", label="A2744 lensing mass")
        ax.set_xlabel(r"$\log_{10} M(<750\,\mathrm{kpc})$ [M$_\odot$]")
        ax.set_ylabel("Posterior density")
        ax.set_title(f"{d.upper()}: mass", color=INK)
        ax.legend(loc="upper right", fontsize=8)

        ax = row[2]
        k = np.arange(1, NCOMP_MAX + 1)
        nle_p = post.sum((0, 1))
        npe_p = np.array([np.mean(np.clip(np.round(smp[:, 2]), 1, NCOMP_MAX) == i) for i in k])
        ax.bar(k - 0.2, nle_p, width=0.38, color=NLE_COLOUR, label="NLE")
        ax.bar(k + 0.2, npe_p, width=0.38, color=NPE_COLOUR, label="NPE")
        ax.set_xticks(k)
        ax.set_xticklabels([str(i) for i in k[:-1]] + [f"{NCOMP_MAX}+"])
        ax.set_xlabel("Merging components")
        ax.set_ylabel("Posterior probability")
        ax.set_title(f"{d.upper()}: merger state", color=INK)
        ax.legend(loc="upper right")

        ax = row[3]
        joint = post.sum(2)
        levels = credible_levels(joint, [0.95, 0.68])
        ax.contour(sig_x, gm, joint.T, levels=levels, colors=NLE_COLOUR, linewidths=2)
        kde = gaussian_kde(np.c_[np.expm1(smp[::4, 0]), smp[::4, 1]].T)
        X, Y = np.meshgrid(sig_x, gm, indexing="ij")
        Z = kde(np.c_[X.ravel(), Y.ravel()].T).reshape(X.shape)
        ax.contour(sig_x, gm, Z.T, levels=credible_levels(Z, [0.95, 0.68]), colors=NPE_COLOUR,
                   linewidths=2, linestyles="--")
        ax.axhspan(mu - sig, mu + sig, color=PRIOR_COLOUR, alpha=0.15, lw=0)
        ax.set_xlabel(r"$\sigma/m$ [cm$^2$ g$^{-1}$]")
        ax.set_ylabel(r"$\log_{10} M(<750\,\mathrm{kpc})$")
        ax.set_title(f"{d.upper()}: 68/95% regions", color=INK)
        ax.legend(handles=[plt.Line2D([], [], color=NLE_COLOUR, lw=2, label="NLE"),
                           plt.Line2D([], [], color=NPE_COLOUR, lw=2, ls="--", label="NPE")], loc="upper right")
    fig.suptitle("A2744 posteriors", color=INK, fontsize=13)
    save(fig, out / "posterior")


def credible_levels(density, masses):
    flat = np.sort(density.ravel())[::-1]
    c = np.cumsum(flat) / flat.sum()
    return sorted(float(flat[np.searchsorted(c, m)]) for m in masses)


def plot_discrepancy(R, out):
    fig, axes = plt.subplots(1, len(R), figsize=(6.5 * len(R), 4.2), constrained_layout=True, squeeze=False)
    for ax, (d, r) in zip(axes[0], R.items()):
        emp = r["empirical"]
        sig_vals = list(emp)
        shades = np.linspace(0.35, 0.95, len(sig_vals))
        allx = np.concatenate([emp[s]["x"] for s in sig_vals])
        grid = np.linspace(*np.percentile(allx, [0.5, 99.5]), 300)
        for s, shade in zip(sig_vals, shades):
            kde = gaussian_kde(emp[s]["x"], weights=emp[s]["w"])
            ax.plot(grid, kde(grid), color=RAMP[d](shade), lw=2, label="CDM" if s == "0" else s)
        x_obs = r["x_obs"]
        ax.axvline(x_obs, color="k", lw=2)
        res = r["results"]
        e = res["empirical"]
        z = (">" if e["is_lower_limit"] else "") + f"{e['significance_sigma']:.1f}"
        text = (f"A2744 vs CDM\n"
                f"p = {e['p_value']:.2g}  ({z}$\\sigma$)\n"
                f"BF$_{{\\rm SIDM:CDM}}$  NLE {res['nle']['bayes_factor_sidm_cdm']:.1f},"
                f"  NPE {res['npe']['bayes_factor_sidm_cdm']:.1f}")
        ax.text(0.02, 0.97, text, transform=ax.transAxes, va="top", fontsize=9, color=INK)
        ax.annotate("A2744", (x_obs, 0.5), xycoords=("data", "axes fraction"), xytext=(4, 0),
                    textcoords="offset points", fontsize=9, color=INK)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], title=r"$\sigma/m$", loc="upper right", fontsize=8, title_fontsize=8)
        ax.set_xlabel("logit ensemble P(SIDM)")
        ax.set_ylabel("Density (merging, weighted to A2744 mass)")
        ax.set_title(d.upper(), color=INK)
    save(fig, out / "discrepancy")


def plot_validation(R, out):
    fig, axes = plt.subplots(2, len(R), figsize=(6 * len(R), 8.5), constrained_layout=True, squeeze=False)
    names = [r"$\sigma/m$", "mass", r"$n_{\rm comp}$"]
    colours = ["#0072B2", "#009E73", "#CC79A7"]
    for j, (d, r) in enumerate(R.items()):
        ax = axes[0, j]
        rec = r["recovery"]
        jit = rec[:, 0] * (1 + 0.04 * np.random.default_rng(1).standard_normal(len(rec)))
        ax.errorbar(jit, rec[:, 2], yerr=[rec[:, 2] - rec[:, 1], rec[:, 3] - rec[:, 2]], fmt="o", ms=5,
                    color=RAMP[d](0.8), alpha=0.6, capsize=0)
        top = max(rec[:, 0].max(), rec[:, 3].max()) * 1.05
        ax.plot([0, top], [0, top], "--", color=MUTED)
        ax.set_xlabel(r"true $\sigma/m$")
        ax.set_ylabel(rf"NLE median, {N_STACK} clusters (nuisance known)")
        ax.set_title(f"{d.upper()}: NLE recovery", color=INK)

        ax = axes[1, j]
        ranks = r["sbc_ranks"]
        u = np.linspace(0, 1, 101)
        band = 1.96 * np.sqrt(u * (1 - u) / len(ranks))
        ax.fill_between(u, -band, band, color="k", alpha=0.08, lw=0)
        for i in range(3):
            ecdf = np.array([np.mean(ranks[:, i] <= t) for t in u])
            ax.plot(u, ecdf - u, color=colours[i], lw=2, label=names[i])
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xlabel("Rank of true value")
        ax.set_ylabel("ECDF - uniform")
        ax.set_title(f"{d.upper()}: NPE calibration (SBC, 95% band)", color=INK)
        ax.legend(loc="upper right")
    save(fig, out / "validation")


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=180)
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────────
def run_domain(d, opts, out, rng, mu, sig):
    print(f"\n=== {d} ===")
    D = load_domain(opts.summaries, d, opts.noise, opts.latent_pcs)
    th, n = D["theta"], D["n_clusters"]
    test = (rng.random(n) < TEST_FRAC)[D["cluster"]]           # all draws of a cluster on one side
    check = test & (D["draw"] == 0)                             # one draw per held-out cluster for checks
    sigma_cut = cdm_threshold(D["sigma_grid"])
    print(f"  {n} clusters x {D['n_draws']} noise draws ({D['n_dropped']} without a valid mass dropped), "
          f"CDM := sigma/m < {sigma_cut:g}")

    empirical, emp_summary = empirical_test(D, mu, sig)
    print(f"  empirical: p(CDM >= A2744) = {emp_summary['p_value']:.3g} "
          f"({'>' if emp_summary['is_lower_limit'] else ''}{emp_summary['significance_sigma']:.2f} sigma, "
          f"N_eff = {emp_summary['n_eff']:.0f})")

    lo, hi = bounds(D)
    prior = BoxUniform(torch.tensor(lo, dtype=torch.float32), torch.tensor(hi, dtype=torch.float32))

    path = out / f"nle_{d}.pt"
    if path.exists() and not opts.retrain:
        nle = torch.load(path, weights_only=False)
    else:
        nle = train_nle(th[~test], D["x_ens"][~test], prior)
        torch.save(nle, path)
    grids = nle_grids(D)
    mass_prior = (mu, sig) if opts.mass_prior == "a2744" else None
    post, log_like_sigma = nle_posterior(nle, D["x_obs_ens"][0], grids, mass_prior)
    nle_res = nle_summary(post, log_like_sigma, grids, D["sigma_grid"])
    print(f"  NLE: sigma/m = {nle_res['sigma_quantiles'][1]:.3f} [{nle_res['sigma_quantiles'][0]:.3f}, "
          f"{nle_res['sigma_quantiles'][2]:.3f}], BF(SIDM:CDM) = {nle_res['bayes_factor_sidm_cdm']:.2f}, "
          f"LR = {nle_res['likelihood_ratio_best_vs_cdm']:.2f}")
    recovery = nle_recovery(nle, th[check], D["x_ens"][check], grids[0], rng)

    th_deq = dequantise(th, rng)
    path = out / f"npe_{d}.pt"
    if path.exists() and not opts.retrain:
        npe = torch.load(path, weights_only=False)
    else:
        npe = train_npe(th_deq[~test], D["x_npe"][~test], prior)
        torch.save(npe, path)
    samples = npe.sample((N_POST,), x=torch.tensor(D["x_obs_npe"][None]),
                         show_progress_bars=False).numpy()
    npe_res = npe_summary(samples, D["sigma_grid"], th[~test])
    print(f"  NPE: sigma/m = {npe_res['sigma_quantiles'][1]:.3f} [{npe_res['sigma_quantiles'][0]:.3f}, "
          f"{npe_res['sigma_quantiles'][2]:.3f}], BF(SIDM:CDM) = {npe_res['bayes_factor_sidm_cdm']:.2f}")
    sbc = npe_sbc(npe, th[check], D["x_npe"][check], D["sigma_grid"], rng)
    test_metrics = npe_test_metrics(npe, th[check], D["x_npe"][check], D["sigma_grid"], rng)
    print(f"  NPE on held-out single clusters: P(true sigma/m) = {test_metrics['sigma_p_true']:.2f} "
          f"(chance {test_metrics['sigma_p_true_chance']:.2f}), most probable right "
          f"{100 * test_metrics['sigma_map_correct']:.0f}%, mass std {test_metrics['mass_posterior_std']:.3f} "
          f"(prior {test_metrics['mass_prior_std']:.3f}), P(true n_comp) = {test_metrics['ncomp_p_true']:.2f}")

    checks = {
        "obs_summary_percentile_mahalanobis": misspecification(D["x_npe"], D["x_obs_npe"]),
        "cdm_training_bias_logit": training_bias(D),
    }
    print(f"  A2744 within simulations: {100 * checks['obs_summary_percentile_mahalanobis']:.1f}% of simulated "
          f"summaries are further out; trained-on CDM clusters are {checks['cdm_training_bias_logit']:+.3f} in logit")

    results = {"empirical": emp_summary, "nle": nle_res, "npe": npe_res, "checks": checks,
               "npe_test_metrics": test_metrics, "npe_input_dim": int(D["x_npe"].shape[1]),
               "latent_variance_explained": D["latent_variance_explained"],
               "cdm_definition_sigma_below": float(sigma_cut), "n_clusters": n,
               "empirical_p_by_sigma": {s: v["p_ge_obs"] for s, v in empirical.items()}}
    return results, {"grids": grids, "nle_post": post, "sigma_grid": D["sigma_grid"], "npe_samples": samples, "sigma_cut": sigma_cut,
                     "empirical": empirical, "x_obs": float(D["x_obs_ens"][0]), "recovery": recovery,
                     "sbc_ranks": sbc, "results": results}


def main():
    opts = parse()
    torch.manual_seed(opts.seed)
    rng = np.random.default_rng(opts.seed)
    warnings.filterwarnings("ignore", category=UserWarning)
    setup_style()
    summaries = Path(opts.summaries).resolve()
    opts.summaries = str(summaries)
    name = (summaries.name + (f"_pc{opts.latent_pcs}" if opts.latent_pcs > 0 else "")
            + ("_alldraws" if opts.noise == "all" else ""))
    out = Path(opts.out_dir) if opts.out_dir else REPO / "sbi" / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    mu, sig = a2744_mass_gaussian()

    results, R = {}, {}
    for d in DOMAINS:
        results[d], R[d] = run_domain(d, opts, out, rng, mu, sig)

    info = json.load(open(summaries / "info.json"))
    results["settings"] = {"summaries": summaries.name, "model": info["options"]["model"],
                           "n_models": len(info["seeds"]), "mass_prior_nle": opts.mass_prior,
                           "latent_pcs": opts.latent_pcs,
                           "noise": opts.noise,
                           "a2744_log_mass": [mu, sig], "ncomp_max": NCOMP_MAX, "seed": opts.seed}
    json.dump(results, open(out / "results.json", "w"), indent=2)
    np.savez(out / "posteriors.npz", **{f"{d}_{k}": v for d, r in R.items()
                                         for k, v in [("nle_post", r["nle_post"]), ("npe_samples", r["npe_samples"]),
                                                      ("sigma_grid", np.expm1(r["grids"][0])),
                                                      ("mass_grid", r["grids"][1])]})
    plot_posterior(R, mu, sig, out)
    plot_discrepancy(R, out)
    plot_validation(R, out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
