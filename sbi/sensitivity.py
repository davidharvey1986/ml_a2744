#!/usr/bin/env python
"""
Where in the map do the CNNs look? Occlusion sensitivity for A2744 and for simulated
clusters matched to it, processed identically (replaces data_sensitivity_map.ipynb).

For each map, a window (default 4x4 pixels, 80 kpc) is slid over the image and filled
with exactly what an empty pixel looks like after normalisation, i.e. the galaxies
there are removed. The sensitivity of a window is

    Delta = P(SIDM | full map) - P(SIDM | window removed)      (mean over models)

so Delta > 0 means that region pushes the output towards SIDM.

Like-for-like choices:
  * the same models, window, fill and reference (the unoccluded output) for A2744 and
    the simulations
  * simulated clusters drawn per sigma/m with probability proportional to the A2744
    mass and merger-class weights (mass_weighting.py, n_comp >= 3), forward-modelled
    as in the summaries: source-redshift rescaling, shape noise at the catalogue's
    galaxies, the same normalisation
  * kappa from the same Kaiser-Squires reconstruction of the binned noisy shear for
    both (ks93 on chi/2 ~ g); the true kappa of the simulations is shown alongside
  * binomial errors; maps drawn at the window centres

A2744 is also compared with the Lenstool excess map (model-subtracted, delensed
convergence, Dark_Matter_DA/lenstool/concat/delens_best/excess_kappa.fits): the stamp
is mapped onto the Lenstool frame with an affine fit to the catalogue galaxies, the
excess S/N (KAPPA_E / SIGMA) is averaged over each occlusion window and correlated with
the window sensitivity (Spearman; null from all circular shifts of the sensitivity map,
which keeps its spatial correlation).

Windows without galaxies (removing them changes nothing) are left out of all statistics.

Outputs (sbi/results/sensitivity_<model>/): sensitivity.npz / .json,
sensitivity_maps.pdf/png (A2744 with excess-S/N contours and Lenstool halos, matched
CDM / SIDM examples with kappa S/N contours), sensitivity_vs_kappa.pdf/png and
sensitivity_vs_excess.pdf/png.

Usage
-----
    cd ml_check && astro-env/bin/python sbi/sensitivity.py
    astro-env/bin/python sbi/sensitivity.py --model tweakedalign_k0p9_ba_best --clusters-per-sigma 30
"""

import argparse
import json
import pickle as pkl
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic, spearmanr
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
LAUNCH_DIR = Path.cwd()
sys.path[:0] = [str(REPO), str(REPO / "notebooks"), str(REPO / "sbi")]

import make_summaries as ms  # noqa: E402  (chdirs to notebooks/, loads the default args)
from add_shear_to_data import (  # noqa: E402
    SHEAR_CUTS, galaxy_distortion, get_obs_data, normalise_map, ra_dec_to_simulation_image_pos,
)
from dataset import get_test_transform  # noqa: E402
from lenspack.image.inversion import ks93  # noqa: E402
from lenspack.utils import bin2d  # noqa: E402
from mass_weighting import a2744_mass_gaussian, gaussian_mass_weights  # noqa: E402

NPIX = 100
SUBSET_DEX = 0.45                      # --all-clusters: clusters within this of the A2744 mass
A2744_NCOMP = (3, 4, 5, 6)
DOMAINS = ["bahamas", "darkskies"]
KAPPA_SMOOTH = 1.5                      # pixels, for the reconstructed kappa
INK, MUTED = "#222222", "#6b6b6b"
COLOUR = {"bahamas": "#56B4E9", "darkskies": "#E69F00"}

EXCESS_MAP = Path("/Users/davidharvey/Work/Dark_Matter_DA/lenstool/concat/delens_best/excess_kappa.fits")
LENSTOOL_REF = (3.494741, -30.339158)            # RA, Dec of the Lenstool frame (maps.par)
LENSTOOL_HALOS = {"O1": (-153.655514, -113.582824),   # Lenstool x, y [arcsec] (excess_mass.py)
                  "O2": (-290.369958, -228.819861),
                  "O3": (-266.190667, -64.959464)}


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="tweakedalign_ba_best")
    p.add_argument("--nmodels", type=int, default=None)
    p.add_argument("--window", type=int, default=4, help="occluded window size [pixels]")
    p.add_argument("--stride", type=int, default=4, help="step between windows [pixels]")
    p.add_argument("--clusters-per-sigma", type=int, default=20)
    p.add_argument("--all-clusters", action="store_true",
                   help="use every cluster in A2744's merger class within SUBSET_DEX of its mass, weighted to the "
                        "A2744 mass, instead of drawing --clusters-per-sigma")
    p.add_argument("--domains", nargs="+", default=DOMAINS, choices=DOMAINS)
    p.add_argument("--summaries", default=str(REPO / "sbi" / "summaries_ba_best_n5"),
                   help="any summaries dir: only cluster masses / n_comp / file indices are used")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--excess-map", default=str(EXCESS_MAP), help="Lenstool excess_kappa.fits ('' to skip)")
    p.add_argument("--replot", action="store_true", help="only redraw the correlation figure from sensitivity.npz")
    p.add_argument("--kappa-plot", action="store_true",
                   help="BAHAMAS-only mean Delta P against reconstructed kappa, with A2744 errors from "
                        "--rotations shear-rotated maps (uses sensitivity.npz; same --seed / --clusters-per-sigma)")
    p.add_argument("--rotations", type=int, default=100, help="maps for the A2744 error estimate")
    p.add_argument("--bin-factor", type=int, default=2,
                   help="--kappa-plot: multiply the number of bins (10 kappa quantile bins, 9 S/N bins) by this")
    p.add_argument("--errors", choices=["bootstrap", "rotation"], default="bootstrap",
                   help="A2744 errors: resample its galaxies with replacement (keeps the signal) or rotate "
                        "every shear (noise only; underestimates the errors, the networks see an empty field)")
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


# ── occlusion ──────────────────────────────────────────────────────────────────
def windows(window, stride):
    starts = np.arange(0, NPIX - window + 1, stride)
    return starts, starts + (window - 1) / 2. - (NPIX / 2 - 0.5)        # starts, centres in map coords


@torch.no_grad()
def model_outputs(models, images, device, batch_size):
    '''P(SIDM), (n_models, n_images).'''
    out = []
    for m in models:
        p = [ms.embed(m, images[i:i + batch_size].to(device))[1].cpu() for i in range(0, len(images), batch_size)]
        out.append(torch.cat(p))
    return torch.stack(out).numpy()


def occlusion_map(models, image, fill, window, stride, device, batch_size):
    '''Delta = P(full) - P(occluded) per window, (n_models, n, n), and P(full) per model.'''
    starts, _ = windows(window, stride)
    occluded = []
    for y in starts:                                                    # rows = dec
        for x in starts:                                                # columns = ra
            t = image.clone()
            t[:, y:y + window, x:x + window] = fill
            occluded.append(t)
    full = model_outputs(models, image[None], device, 1)[:, 0]
    p = model_outputs(models, torch.stack(occluded), device, batch_size)
    return (full[:, None] - p).reshape(len(models), len(starts), len(starts)), full


def window_mean(field, window, stride):
    '''Mean of a 100x100 field over each occlusion window, (n, n).'''
    starts, _ = windows(window, stride)
    return np.array([[np.nanmean(field[y:y + window, x:x + window]) for x in starts] for y in starts])


def reconstruct_kappa(raw):
    '''
    Kaiser-Squires from a binned distortion map chi (2, 100, 100); chi/2 ~ reduced shear.
    Returns the smoothed E-mode and its S/N (E / std of the B-mode).
    '''
    ke, kb = (gaussian_filter(k, KAPPA_SMOOTH) for k in ks93(raw[0] / 2., raw[1] / 2.))
    return ke, ke / np.std(kb)


def normalise(t):
    '''As BalancedClusterDataset.__getitem__, returning the value an empty (zero) pixel takes.'''
    mn, mx = torch.min(t), torch.max(t)
    return (t - mn) / mx, float((0 - mn) / mx)


# ── data and simulations ───────────────────────────────────────────────────────
def a2744_map():
    '''Raw (chi) and normalised A2744 map, built as add_shear_to_data.make_obs_maps map 0.'''
    cat = get_obs_data("concat", data_dir="../data/100/a2744/", cuts=SHEAR_CUTS)
    ra, dec = ra_dec_to_simulation_image_pos(cat)
    c1, c2 = galaxy_distortion(cat)
    raw = np.array(bin2d(ra, dec, v=(c1, c2), npix=NPIX, extent=[-50, 50, -50, 50]), dtype=np.float32)
    _, stored = pkl.load(open("../data/100/a2744/obs_data_concat.pkl", "rb"))
    assert np.allclose(np.array(normalise_map(raw[0], raw[1])), stored[0], atol=1e-5), "A2744 map differs from obs_data_concat.pkl"
    return raw, cat, ra, dec


def pick_clusters(summaries, domain, mu, sig, k, rng, all_clusters=False):
    '''
    Clusters in A2744's merger class, per sigma/m: either k drawn with probability proportional
    to the A2744 mass weights (equal weights afterwards), or all within SUBSET_DEX of its mass
    carrying those weights.
    '''
    P = dict(np.load(Path(summaries) / f"params_{domain}.npz"))
    cs, m, nc = P["cross_section"], P["log10_mass_750"], np.minimum(P["ncomponents"], 6)
    picks, weights = [], []
    for c in np.unique(cs):
        rows = np.where((cs == c) & np.isin(nc, A2744_NCOMP) & np.isfinite(m))[0]
        if all_clusters:
            rows = rows[np.abs(m[rows] - mu) < SUBSET_DEX]
            picks.append(rows)
            weights.append(gaussian_mass_weights(m[rows], mu, sig))
        else:
            w = gaussian_mass_weights(m[rows], mu, sig)
            picks.append(rng.choice(rows, min(k, len(rows)), replace=False, p=w / w.sum()))
            weights.append(np.ones(len(picks[-1])))
    rows = np.concatenate(picks)
    return {"filename": P["filename"][rows], "image_idx": P["image_idx"][rows], "cross_section": cs[rows],
            "log10_mass_750": m[rows], "ncomponents": P["ncomponents"][rows], "weight": np.concatenate(weights)}


def simulated_maps(sel, args, seed):
    '''Forward-modelled raw maps (after the test transform, before normalisation) and true kappa.'''
    transform = get_test_transform(args)
    raws, kappas = [], []
    np.random.seed(seed)
    torch.manual_seed(seed)
    for f in np.unique(sel["filename"]):
        rows = np.where(sel["filename"] == f)[0]
        _, obs = pkl.load(open(f"../data/100/obs/concat/{f}", "rb"))
        _, shear = pkl.load(open(f"../data/100/shear/{f}", "rb"))
        for r in rows:
            i = sel["image_idx"][r]
            raws.append((r, transform(torch.tensor(obs[i, :2], dtype=torch.float32))))
            kappas.append((r, shear[i, 2]))
    order = np.argsort([r for r, _ in raws])
    return [raws[j][1] for j in order], [kappas[j][1] for j in order]


def stamp_to_lenstool(cat, ra_pix, dec_pix):
    '''Affine map from stamp pixel coordinates to Lenstool x, y [arcsec], fitted on the catalogue.'''
    ra0, dec0 = LENSTOOL_REF
    x_lt = -(cat["RA"] - ra0) * np.cos(np.deg2rad(dec0)) * 3600.
    y_lt = (cat["DEC"] - dec0) * 3600.
    A = np.c_[ra_pix, dec_pix, np.ones_like(ra_pix)]
    cx, cy = (np.linalg.lstsq(A, v, rcond=None)[0] for v in (x_lt, y_lt))
    rms = np.sqrt(np.mean((A @ cx - x_lt)**2 + (A @ cy - y_lt)**2))
    M = np.array([cx[:2], cy[:2]])
    print(f"stamp -> Lenstool frame: {np.sqrt(abs(np.linalg.det(M))):.3f} arcsec per stamp pixel, rms residual {rms:.3f} arcsec")

    def forward(px, py):
        return cx[0] * px + cx[1] * py + cx[2], cy[0] * px + cy[1] * py + cy[2]

    def inverse(x, y):
        return np.linalg.solve(M, np.array([x - cx[2], y - cy[2]]))
    return forward, inverse


def excess_on_stamp(path, forward):
    '''Excess-kappa S/N (KAPPA_E / SIGMA) resampled onto the 100x100 stamp ([dec, ra]); nan outside.'''
    with fits.open(path) as h:
        snr = h["KAPPA_E"].data / h["SIGMA"].data
        hd = h["KAPPA_E"].header
    xg = hd["CRVAL1"] + hd["CDELT1"] * (np.arange(snr.shape[1]) + 1 - hd["CRPIX1"])
    yg = hd["CRVAL2"] + hd["CDELT2"] * (np.arange(snr.shape[0]) + 1 - hd["CRPIX2"])
    interp = RegularGridInterpolator((yg, xg), snr, bounds_error=False, fill_value=np.nan)
    pix = np.arange(NPIX) - (NPIX / 2 - 0.5)
    PX, PY = np.meshgrid(pix, pix)                                       # [dec, ra]
    x, y = forward(PX, PY)
    return interp(np.c_[y.ravel(), x.ravel()]).reshape(NPIX, NPIX)


def shift_null(delta, field, valid):
    '''Spearman rho of delta vs field, and its null from all circular shifts of the delta map.'''
    rho = spearmanr(delta[valid], field[valid]).statistic
    n = delta.shape[0]
    null = []
    for dy in range(n):
        for dx in range(n):
            if dx == 0 and dy == 0:
                continue
            d = np.roll(np.roll(delta, dy, 0), dx, 1)
            v = np.roll(np.roll(valid, dy, 0), dx, 1) & valid
            if v.sum() > 10:
                null.append(spearmanr(d[v], field[v]).statistic)
    null = np.array(null)
    return float(rho), float(np.mean(np.abs(null) >= abs(rho))), null


# ── plots ──────────────────────────────────────────────────────────────────────
def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
                         "legend.frameon": False, "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    return plt


def plot_maps(panels, centres, stride, path):
    plt = style()
    half = stride / 2.
    extent = [centres[0] - half, centres[-1] + half, centres[0] - half, centres[-1] + half]
    vmax = max(np.percentile(np.abs(p["delta"]), 99) for p in panels)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.6), constrained_layout=True)
    for ax, p in zip(np.atleast_1d(axes), panels):
        im = ax.imshow(np.ma.masked_equal(p["delta"], 0), origin="lower", extent=extent, cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax)
        ax.contour(p["kappa_snr"], origin="lower", extent=[-50, 50, -50, 50], levels=[2, 3, 4, 6, 8],
                   colors="k", linewidths=1)
        if p.get("kappa_true") is not None:
            ax.contour(gaussian_filter(p["kappa_true"], 1), origin="lower", extent=[-50, 50, -50, 50],
                       levels=np.linspace(0.1, 0.8, 8), colors="k", linewidths=0.8, linestyles=":")
        if p.get("excess_snr") is not None:
            ax.contour(p["excess_snr"], origin="lower", extent=[-50, 50, -50, 50], levels=[2, 3, 4],
                       colors="#2ca02c", linewidths=1.5)
            ax.contour(p["excess_snr"], origin="lower", extent=[-50, 50, -50, 50], levels=[-4, -3, -2],
                       colors="#2ca02c", linewidths=1, linestyles="--")
            for name, (hx, hy) in p["halos"].items():
                ax.plot(hx, hy, "x", color="#2ca02c", ms=9, mew=2)
                ax.text(hx + 2, hy + 2, name, color="#2ca02c", fontsize=8)
        ax.set_title(p["title"], color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.errorbar(-35, -45, xerr=10, capsize=2, color=INK, lw=1.5)
        ax.text(-35, -44, "400 kpc", ha="center", va="bottom", fontsize=8, color=INK)
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label(r"$\Delta P$(SIDM) when removed (red: pushes to SIDM)")
    fig.suptitle("Occlusion sensitivity. Black: reconstructed $\\kappa$ S/N 2,3,4,6,8 (dotted: true $\\kappa$); "
                 "green: Lenstool excess $\\kappa$ S/N $\\pm$2,3,4 and halos", color=INK, fontsize=10)
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=180)


def binned_fraction(kappa, delta, bins):
    keep = (delta != 0) & np.isfinite(kappa)            # windows without galaxies carry no information
    kappa, delta = kappa[keep], delta[keep]
    sens = (delta > 0).astype(float)
    frac, _, _ = binned_statistic(kappa, sens, statistic="mean", bins=bins)
    n, _, _ = binned_statistic(kappa, sens, statistic="count", bins=bins)
    mean, _, _ = binned_statistic(kappa, delta, statistic="mean", bins=bins)
    std, _, _ = binned_statistic(kappa, delta, statistic="std", bins=bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        return frac, np.sqrt(frac * (1 - frac) / n), mean, std / np.sqrt(n), n


SNR_BINS = np.array([-3, -2, -1, 0, 1, 2, 3, 4, 6, 10])
P_MATCH = 0.1          # simulated maps whose unoccluded P(SIDM) is within this of A2744's


def pooled(maps, relative):
    '''Concatenate window kappa S/N and Delta over maps; relative subtracts each map's median Delta.'''
    k = np.concatenate([m["snr"] for m in maps]) if maps else np.array([])
    d = np.concatenate([m["delta"] - (np.median(m["delta"][m["delta"] != 0]) if relative else 0) for m in maps]) \
        if maps else np.array([])
    return k, d


def plot_vs_kappa(obs, sims, path, bins=SNR_BINS):
    '''
    Rows: absolute Delta, and Delta relative to each map's median (removes the overall
    shift that a map's own output level gives). Columns: fraction of windows pushing
    towards SIDM, mean Delta. Simulations per sigma/m, and all simulated maps whose
    unoccluded output is within P_MATCH of A2744's (black dashed).
    '''
    plt = style()
    xc = 0.5 * (bins[1:] + bins[:-1])
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), constrained_layout=True, sharex=True)
    matched = [m for d in sims for m in sims[d] if abs(m["p_full"] - obs["p_full"]) < P_MATCH]
    for row, relative in enumerate([False, True]):
        for d in DOMAINS:
            cmap = plt.cm.Blues if d == "bahamas" else plt.cm.Oranges
            sig_vals = sorted({m["sigma"] for m in sims.get(d, [])})
            for s, sh in zip(sig_vals, np.linspace(0.35, 0.95, len(sig_vals))):
                k, dl = pooled([m for m in sims[d] if m["sigma"] == s], relative)
                frac, ferr, mean, merr, n = binned_fraction(k, dl, bins)
                ok = n >= 20
                lab = f"{d.upper()} " + ("CDM" if s == 0 else f"{s:g}")
                axes[row, 0].errorbar(xc[ok], frac[ok], ferr[ok], fmt="o-", ms=3, lw=1.2, capsize=2,
                                      color=cmap(sh), label=lab if row == 0 else None)
                axes[row, 1].errorbar(xc[ok], mean[ok], merr[ok], fmt="o-", ms=3, lw=1.2, capsize=2, color=cmap(sh))
        if matched:
            k, dl = pooled(matched, relative)
            frac, ferr, mean, merr, n = binned_fraction(k, dl, bins)
            ok = n >= 20
            lab = f"sims with P within {P_MATCH} of A2744 ({len(matched)} maps)"
            axes[row, 0].errorbar(xc[ok], frac[ok], ferr[ok], fmt="--", lw=2, color="k", capsize=2,
                                  label=lab if row == 0 else None)
            axes[row, 1].errorbar(xc[ok], mean[ok], merr[ok], fmt="--", lw=2, color="k", capsize=2)
        k, dl = pooled([obs], relative)
        frac, ferr, mean, merr, n = binned_fraction(k, dl, bins)
        ok = n >= 5
        axes[row, 0].errorbar(xc[ok], frac[ok], ferr[ok], fmt="s", ms=7, color="k", capsize=2,
                              label="A2744" if row == 0 else None, zorder=5)
        axes[row, 1].errorbar(xc[ok], mean[ok], merr[ok], fmt="s", ms=7, color="k", capsize=2, zorder=5)
        axes[row, 0].axhline(0.5, color=MUTED, lw=1)
        axes[row, 1].axhline(0, color=MUTED, lw=1)
        what = "relative to the map's median" if relative else "absolute"
        axes[row, 0].set_ylabel(f"Fraction of windows pushing to SIDM\n({what})")
        axes[row, 1].set_ylabel(f"Mean $\\Delta P$(SIDM) ({what})")
    for ax in axes[1]:
        ax.set_xlabel(r"Reconstructed $\kappa$ S/N (window mean)")
    axes[0, 0].legend(fontsize=7, ncols=2, loc="upper left")
    fig.suptitle("Where the CNNs find SIDM signal: A2744 vs matched simulations (same pipeline)",
                 color=INK, fontsize=11)
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=180)


def partial_spearman(a, b, c):
    '''Spearman correlation of a and b with c held fixed (ranks, linear residuals on c).'''
    from scipy.stats import rankdata
    ra, rb, rc = (rankdata(v) for v in (a, b, c))
    res = [r - np.polyval(np.polyfit(rc, r, 1), rc) for r in (ra, rb)]
    return float(np.corrcoef(*res)[0, 1])


def shift_null_partial(delta, field, control, valid):
    '''Partial Spearman of delta vs field given control, with the circular-shift null on delta.'''
    rho = partial_spearman(delta[valid], field[valid], control[valid])
    n, null = delta.shape[0], []
    for dy in range(n):
        for dx in range(n):
            if dx or dy:
                d = np.roll(np.roll(delta, dy, 0), dx, 1)
                v = np.roll(np.roll(valid, dy, 0), dx, 1) & valid
                null.append(partial_spearman(d[v], field[v], control[v]))
    null = np.array(null)
    return rho, float(np.mean(np.abs(null) >= abs(rho))), null


def plot_correlations(npz, window, stride, path):
    '''
    Correlation of the window sensitivity with the reconstructed kappa and with the Lenstool
    excess kappa: A2744 scatter + Spearman rho (circular-shift nulls), the excess correlation
    with kappa held fixed, and the per-cluster rho(sensitivity, kappa) of the simulations.
    '''
    plt = style()
    z = np.load(npz)
    n = len(z["centres"])
    delta = z["a2744_delta"].mean(0)
    k_w = z["a2744_snr_windows"].reshape(n, n)
    valid = (delta != 0) & np.isfinite(k_w)
    rho_k, p_k, null_k = shift_null(delta, k_w, valid)
    has_excess = "a2744_excess_snr" in z
    if has_excess:
        e_w = window_mean(z["a2744_excess_snr"], window, stride)
        valid &= np.isfinite(e_w)
        rho_e, p_e, null_e = shift_null(delta, e_w, valid)
        rho_ep, p_ep, null_ep = shift_null_partial(delta, e_w, k_w, valid)
        rho_ke = spearmanr(k_w[valid], e_w[valid]).statistic

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    ax = axes[0]
    ax.hexbin(k_w[valid], delta[valid], gridsize=40, mincnt=1, cmap="Greys", bins="log")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlabel(r"Reconstructed $\kappa$ S/N (window mean)")
    ax.set_ylabel(r"$\Delta P$(SIDM) when removed")
    ax.set_title(f"A2744 vs $\\kappa$: $\\rho$ = {rho_k:+.2f} (p = {p_k:.3f})", color=INK)

    ax = axes[1]
    if has_excess:
        ax.hexbin(e_w[valid], delta[valid], gridsize=40, mincnt=1, cmap="Greens", bins="log")
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xlabel(r"Lenstool excess $\kappa$ S/N (window mean)")
        ax.set_title(f"A2744 vs excess: $\\rho$ = {rho_e:+.2f} (p = {p_e:.3f})", color=INK)
        ax.text(0.02, 0.97, f"with $\\kappa$ held fixed: $\\rho$ = {rho_ep:+.2f} (p = {p_ep:.3f})\n"
                f"$\\rho(\\kappa$, excess) = {rho_ke:+.2f}", transform=ax.transAxes, va="top", fontsize=8,
                color=INK)
    else:
        ax.set_axis_off()

    ax = axes[2]
    xpos, labels = 0, []
    for d in DOMAINS:
        if f"{d}_delta" not in z:
            continue
        sig = z[f"{d}_sigma"]
        cmap = plt.cm.Blues if d == "bahamas" else plt.cm.Oranges
        for s, sh in zip(np.unique(sig), np.linspace(0.35, 0.95, len(np.unique(sig)))):
            rhos = []
            for dm, kw in zip(z[f"{d}_delta"][sig == s], z[f"{d}_snr_windows"][sig == s]):
                kw = kw.reshape(n, n)
                v = (dm != 0) & np.isfinite(kw)
                if v.sum() > 20:
                    rhos.append(spearmanr(dm[v], kw[v]).statistic)
            jit = xpos + 0.15 * np.random.default_rng(0).standard_normal(len(rhos))
            ax.plot(jit, rhos, "o", ms=4, alpha=0.6, color=cmap(sh))
            ax.errorbar(xpos, np.mean(rhos), np.std(rhos) / np.sqrt(len(rhos)), fmt="_", ms=18, mew=2,
                        color=INK, capsize=3)
            labels.append(f"{d[:4].upper()}\n" + ("CDM" if s == 0 else f"{s:g}"))
            xpos += 1
    ax.axhline(rho_k, color="k", lw=2, label=f"A2744 ({rho_k:+.2f})")
    lo, hi = np.percentile(null_k, [2.5, 97.5])
    ax.axhspan(lo, hi, color="k", alpha=0.08, lw=0, label="A2744 shift null, 95%")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(r"Spearman $\rho$(sensitivity, $\kappa$ S/N) per cluster")
    ax.set_title("Matched simulations vs A2744", color=INK)
    ax.legend(fontsize=8, loc="best")
    fig.suptitle("Does the CNN look where the mass is? Sensitivity vs reconstructed and excess convergence",
                 color=INK, fontsize=11)
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=180)
    out = {"a2744_rho_kappa": rho_k, "a2744_p_kappa": p_k}
    if has_excess:
        out.update(a2744_rho_excess=rho_e, a2744_p_excess=p_e, a2744_partial_rho_excess_given_kappa=rho_ep,
                   a2744_partial_p=p_ep, rho_kappa_excess=float(rho_ke))
    return out


def plot_vs_excess(excess_w, delta_w, valid, rho, p, null, path):
    plt = style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    x, d = excess_w[valid], delta_w[valid]
    bins = np.quantile(x, np.linspace(0, 1, 9))
    xc = 0.5 * (bins[1:] + bins[:-1])
    frac, ferr, mean, merr, n = binned_fraction(x, d, bins)
    axes[0].errorbar(xc, frac, ferr, fmt="s-", color="k", capsize=2, ms=6)
    axes[0].axhline(0.5, color=MUTED, lw=1)
    axes[0].set_ylabel("Fraction of windows pushing towards SIDM")
    axes[1].errorbar(xc, mean, merr, fmt="s-", color="k", capsize=2, ms=6)
    axes[1].axhline(0, color=MUTED, lw=1)
    axes[1].set_ylabel(r"Mean $\Delta P$(SIDM) when removed")
    for ax in axes[:2]:
        ax.set_xlabel(r"Lenstool excess $\kappa$ S/N (window mean)")
    axes[2].hist(null, bins=40, color=MUTED, alpha=0.6, label="circular shifts (null)")
    axes[2].axvline(rho, color="k", lw=2, label=f"A2744: $\\rho$ = {rho:+.2f}, p = {p:.3f}")
    axes[2].set_xlabel(r"Spearman $\rho$(sensitivity, excess $\kappa$)")
    axes[2].legend(fontsize=8)
    fig.suptitle("A2744: CNN sensitivity vs the Lenstool model-subtracted (excess) convergence", color=INK, fontsize=11)
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=180)


# ── main ───────────────────────────────────────────────────────────────────────
def rotated_a2744_deltas(models, cat, ra, dec, n_rot, window, stride, device, batch_size, cache, seed=1):
    '''
    Occlusion maps of A2744 with every galaxy's distortion rotated by a random angle (amplitude kept):
    shape noise only, same galaxies. Mean over models, (n_rot, n, n); cached.
    '''
    if cache.exists():
        z = np.load(cache)
        if len(z["deltas"]) >= n_rot:
            return z["deltas"][:n_rot]
    c1, c2 = galaxy_distortion(cat)
    amp = np.hypot(c1, c2)
    rng = np.random.default_rng(seed)
    out = []
    for _ in tqdm(range(n_rot), desc="rotated A2744 maps"):
        ang = rng.uniform(0, np.pi, len(amp))
        raw = np.array(bin2d(ra, dec, v=(amp * np.cos(2 * ang), amp * np.sin(2 * ang)), npix=NPIX,
                             extent=[-50, 50, -50, 50]), dtype=np.float32)
        img, fill = normalise(torch.tensor(raw))
        out.append(occlusion_map(models, img, fill, window, stride, device, batch_size)[0].mean(0))
    out = np.array(out)
    np.savez(cache, deltas=out)
    return out


def bootstrap_a2744_deltas(models, cat, ra, dec, n_boot, window, stride, device, batch_size, cache, seed=2):
    '''Occlusion maps of A2744 with its galaxies resampled with replacement (signal kept), (n_boot, n, n); cached.'''
    if cache.exists():
        z = np.load(cache)
        if len(z["deltas"]) >= n_boot:
            return z["deltas"][:n_boot]
    c1, c2 = galaxy_distortion(cat)
    rng = np.random.default_rng(seed)
    out = []
    for _ in tqdm(range(n_boot), desc="bootstrapped A2744 maps"):
        i = rng.integers(0, len(c1), len(c1))
        raw = np.array(bin2d(ra[i], dec[i], v=(c1[i], c2[i]), npix=NPIX, extent=[-50, 50, -50, 50]), dtype=np.float32)
        img, fill = normalise(torch.tensor(raw))
        out.append(occlusion_map(models, img, fill, window, stride, device, batch_size)[0].mean(0))
    out = np.array(out)
    np.savez(cache, deltas=out)
    return out


def bin_means(kw, delta, bins):
    '''Mean Delta P in kappa bins over windows with galaxies.'''
    keep = (delta != 0) & np.isfinite(kw)
    return binned_statistic(kw[keep], delta[keep], statistic="mean", bins=bins)[0]


def weighted_stats(per, w):
    '''Per bin: weighted mean, weighted single-cluster std and error of the mean (Kish effective N).'''
    ok = np.isfinite(per)
    W = np.where(ok, w[:, None], 0.)
    m = np.where(ok, per, 0.)
    sw = W.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (W * m).sum(0) / sw
        std = np.sqrt((W * (m - mean)**2).sum(0) / sw)
        neff = sw**2 / (W**2).sum(0)
        return mean, std, std / np.sqrt(neff)


def plot_delta_vs_kappa(a2744, rot_deltas, sims, path, n_bins=10, x="kappa", bins=None, errors="bootstrap"):
    '''
    Mean Delta P(SIDM) against reconstructed kappa: BAHAMAS per sigma/m (error: scatter between
    clusters / sqrt(N)) and output-matched BAHAMAS maps; A2744 with errors from the scatter over
    shear-rotated (noise-only) versions of the map, binned with A2744's own kappa windows.
    '''
    plt = style()
    kw, dl = a2744[x].ravel(), a2744["delta"].ravel()
    if bins is None:
        bins = np.quantile(kw[np.isfinite(kw)], np.linspace(0, 1, n_bins + 1))
    xc = 0.5 * (bins[1:] + bins[:-1])
    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    sig_vals = sorted({m["sigma"] for m in sims})
    for s, sh in zip(sig_vals, np.linspace(0.35, 0.95, len(sig_vals))):
        sub = [m for m in sims if m["sigma"] == s]
        per = np.array([bin_means(m[x], m["delta"], bins) for m in sub])
        mean, _, sem = weighted_stats(per, np.array([m.get("weight", 1.) for m in sub]))
        ax.errorbar(xc, mean, sem, fmt="o-", ms=4, lw=1.5, capsize=2, color=plt.cm.Blues(sh),
                    label=f"BAHAMAS " + ("CDM" if s == 0 else f"$\\sigma/m={s:g}$") + f" ({len(sub)})")
    sub = [m for m in sims if abs(m["p_full"] - a2744["p_full"]) < P_MATCH]
    if sub:
        per = np.array([bin_means(m[x], m["delta"], bins) for m in sub])
        mm, ms_, sem = weighted_stats(per, np.array([m.get("weight", 1.) for m in sub]))
        ax.fill_between(xc, mm - ms_, mm + ms_, color="k", alpha=0.1, lw=0,
                        label="single-cluster spread ($\\pm1\\sigma$) of those")
        ax.errorbar(xc, mm, sem, fmt="--", lw=2, color="k", capsize=2,
                    label=f"BAHAMAS with P within {P_MATCH} of A2744 ({len(sub)})")
    err = np.array([bin_means(kw, r.ravel(), bins) for r in rot_deltas])
    how = "bootstrapped galaxies" if errors == "bootstrap" else "shear-rotated maps"
    ax.errorbar(xc, bin_means(kw, dl, bins), np.nanstd(err, 0), fmt="o-", ms=4, lw=1.5, color="k", capsize=2,
                zorder=5, label=f"A2744 (errors: {len(err)} {how})")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlabel(r"Reconstructed convergence $\kappa$ (window mean)" if x == "kappa"
                  else r"Reconstructed $\kappa$ S/N (window mean)")
    ax.set_ylabel(r"Mean $\Delta P$(SIDM) when the window is removed")
    ax.set_title("Where the CNNs find SIDM signal: A2744 vs BAHAMAS", color=INK)
    ax.legend(fontsize=7, loc="lower left")
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=180)


def kappa_plot(opts, out):
    '''BAHAMAS-only Delta P vs kappa from sensitivity.npz, with rotated-shear errors for A2744.'''
    z = np.load(out / "sensitivity.npz")
    device = ms.get_device()
    args = ms.setup_args("concat")
    models = ms.load_models(SimpleNamespace(filter="concat", model=opts.model, include_str="nob1",
                                            nmodels=opts.nmodels), args, device)
    raw_np, cat, ra, dec = a2744_map()
    a2744 = {"kappa": window_mean(reconstruct_kappa(raw_np)[0], opts.window, opts.stride),
             "snr": z["a2744_snr_windows"].reshape(z["a2744_delta"].shape[1:]),
             "delta": z["a2744_delta"].mean(0), "p_full": json.load(open(out / "sensitivity.json"))["a2744_p_sidm"]}
    if opts.errors == "bootstrap":
        rot = bootstrap_a2744_deltas(models, cat, ra, dec, opts.rotations, opts.window, opts.stride, device,
                                     opts.batch_size, out / "bootstrap_a2744.npz")
    else:
        rot = rotated_a2744_deltas(models, cat, ra, dec, opts.rotations, opts.window, opts.stride, device,
                                   opts.batch_size, out / "rotated_a2744.npz")
    if "bahamas_kappa_windows" in z:                 # saved by the main run
        sims = [{"sigma": float(s), "p_full": float(p), "kappa": k, "delta": d.ravel(), "snr": n, "weight": float(w)}
                for s, p, k, d, n, w in zip(z["bahamas_sigma"], z["bahamas_p_full"], z["bahamas_kappa_windows"],
                                            z["bahamas_delta"], z["bahamas_snr_windows"], z["bahamas_weight"])]
        return finish_kappa_plot(opts, out, a2744, rot, sims)
    # older runs: regenerate the BAHAMAS maps (same seed, same draw order) to get their kappa
    mu, sig = a2744_mass_gaussian()
    sel = pick_clusters(opts.summaries, "bahamas", mu, sig, opts.clusters_per_sigma, np.random.default_rng(opts.seed))
    assert np.array_equal(sel["cross_section"], z["bahamas_sigma"]), "cluster draw differs from the saved run"
    raws, _ = simulated_maps(sel, args, opts.seed)
    sims = []
    for i, r in enumerate(tqdm(raws, desc="BAHAMAS kappa")):
        k, snr = reconstruct_kappa(r.numpy())
        if i == 0:
            assert np.allclose(window_mean(snr, opts.window, opts.stride).ravel(), z["bahamas_snr_windows"][0], atol=1e-4), \
                "regenerated maps differ from the saved run"
        sims.append({"sigma": float(sel["cross_section"][i]), "p_full": float(z["bahamas_p_full"][i]),
                     "kappa": window_mean(k, opts.window, opts.stride).ravel(), "delta": z["bahamas_delta"][i].ravel(),
                     "snr": z["bahamas_snr_windows"][i], "weight": 1.0})
    finish_kappa_plot(opts, out, a2744, rot, sims)


def finish_kappa_plot(opts, out, a2744, rot, sims):
    f = opts.bin_factor
    snr_bins = np.interp(np.linspace(0, len(SNR_BINS) - 1, f * (len(SNR_BINS) - 1) + 1),
                         np.arange(len(SNR_BINS)), SNR_BINS)          # split each S/N bin into f
    plot_delta_vs_kappa(a2744, rot, sims, out / "delta_vs_kappa_bahamas", n_bins=10 * f, errors=opts.errors)
    plot_delta_vs_kappa(a2744, rot, sims, out / "delta_vs_snr_bahamas", x="snr", bins=snr_bins, errors=opts.errors)
    print(f"wrote {out}/delta_vs_kappa_bahamas and delta_vs_snr_bahamas (.pdf/.png)")


def main():
    opts = parse()
    out = LAUNCH_DIR / (opts.out_dir or REPO / "sbi" / "results" / f"sensitivity_{opts.model}")
    out.mkdir(parents=True, exist_ok=True)
    if opts.kappa_plot:
        kappa_plot(opts, out)
        return
    if opts.replot:
        stats = plot_correlations(out / "sensitivity.npz", opts.window, opts.stride, out / "sensitivity_correlations")
        print(json.dumps(stats, indent=1))
        return
    rng = np.random.default_rng(opts.seed)
    device = ms.get_device()
    args = ms.setup_args("concat")
    models = ms.load_models(SimpleNamespace(filter="concat", model=opts.model, include_str="nob1",
                                            nmodels=opts.nmodels), args, device)
    print(f"{len(models)} models ({opts.model}); window {opts.window} px, stride {opts.stride} px")
    _, centres = windows(opts.window, opts.stride)
    mu, sig = a2744_mass_gaussian()

    # A2744
    raw_np, cat, ra_pix, dec_pix = a2744_map()
    raw = torch.tensor(raw_np)
    img, fill = normalise(raw)
    delta, full = occlusion_map(models, img, fill, opts.window, opts.stride, device, opts.batch_size)
    obs_kappa, obs_snr = reconstruct_kappa(raw_np)
    obs = {"snr": window_mean(obs_snr, opts.window, opts.stride).ravel(), "delta": delta.mean(0).ravel(),
           "p_full": float(full.mean())}
    used = obs["delta"] != 0
    print(f"A2744: P(SIDM) = {full.mean():.3f}; windows pushing to SIDM: {np.mean(obs['delta'][used] > 0):.2f} "
          f"of {used.sum()} with galaxies")
    panel = {"title": f"A2744  (P = {full.mean():.2f})", "delta": delta.mean(0), "kappa_snr": obs_snr}
    saved = {"a2744_delta": delta, "a2744_kappa": obs_kappa, "centres": centres}
    stats = {}
    if opts.excess_map:
        forward, inverse = stamp_to_lenstool(cat, ra_pix, dec_pix)
        excess = excess_on_stamp(opts.excess_map, forward)
        excess_w = window_mean(excess, opts.window, opts.stride)
        valid = (delta.mean(0) != 0) & np.isfinite(excess_w)
        rho, p, null = shift_null(delta.mean(0), excess_w, valid)
        print(f"A2744 sensitivity vs Lenstool excess kappa S/N: Spearman rho = {rho:+.3f}, "
              f"p = {p:.3f} (circular-shift null, {len(null)} shifts, {valid.sum()} windows)")
        panel.update(excess_snr=excess, halos={k: tuple(inverse(*v)) for k, v in LENSTOOL_HALOS.items()})
        saved.update(a2744_excess_snr=excess)
        stats.update(excess_spearman_rho=rho, excess_p_value=p, excess_windows=int(valid.sum()))
        plot_vs_excess(excess_w, delta.mean(0), valid, rho, p, null, out / "sensitivity_vs_excess")
    panels = [panel]

    # matched simulations
    sims = {}
    for d in opts.domains:
        sel = pick_clusters(opts.summaries, d, mu, sig, opts.clusters_per_sigma, rng, opts.all_clusters)
        raws, kappas_true = simulated_maps(sel, args, opts.seed)
        sims[d] = []
        deltas = []
        for i in tqdm(range(len(raws)), desc=f"{d} clusters"):
            img, fill = normalise(raws[i])
            dl, p_full = occlusion_map(models, img, fill, opts.window, opts.stride, device, opts.batch_size)
            k_rec, k_snr = reconstruct_kappa(raws[i].numpy())
            s = float(sel["cross_section"][i])
            sims[d].append({"sigma": s, "p_full": float(p_full.mean()), "delta": dl.mean(0).ravel(),
                            "kappa": window_mean(k_rec, opts.window, opts.stride).ravel(),
                            "weight": float(sel["weight"][i]),
                            "snr": window_mean(k_snr, opts.window, opts.stride).ravel()})
            deltas.append(dl.mean(0))
            # example panels: the first matched CDM and largest-sigma cluster of bahamas
            if d == "bahamas" and (s == 0 or s == sel["cross_section"].max()) and \
                    not any(pn.get("sigma") == s for pn in panels[1:]):
                panels.append({"title": f"BAHAMAS " + ("CDM" if s == 0 else f"$\\sigma/m={s:g}$")
                               + f"  (P = {p_full.mean():.2f})", "delta": dl.mean(0), "kappa_snr": k_snr,
                               "kappa_true": kappas_true[i], "sigma": s})
        saved[f"{d}_delta"] = np.array(deltas)
        saved[f"{d}_p_full"] = np.array([m["p_full"] for m in sims[d]])
        saved[f"{d}_snr_windows"] = np.array([m["snr"] for m in sims[d]])
        saved[f"{d}_kappa_windows"] = np.array([m["kappa"] for m in sims[d]])
        saved[f"{d}_weight"] = sel["weight"]
        saved[f"{d}_sigma"] = sel["cross_section"]
        saved[f"{d}_log10_mass_750"] = sel["log10_mass_750"]
        print(f"{d}: {len(raws)} matched clusters; fraction of windows pushing to SIDM by sigma/m: "
              + ", ".join(f"{s:g}: {np.mean(np.concatenate([m['delta'] for m in sims[d] if m['sigma'] == s]) > 0):.2f}"
                          for s in sorted({m['sigma'] for m in sims[d]})))

    json.dump({"model": opts.model, "n_models": len(models), "window": opts.window, "stride": opts.stride,
               "clusters_per_sigma": opts.clusters_per_sigma, "a2744_p_sidm": float(full.mean()), **stats},
              open(out / "sensitivity.json", "w"), indent=2)
    plot_maps(panels, centres, opts.stride, out / "sensitivity_maps")
    saved["a2744_snr_windows"] = obs["snr"]
    np.savez(out / "sensitivity.npz", **saved)
    n_match = sum(abs(m["p_full"] - obs["p_full"]) < P_MATCH for d in sims for m in sims[d])
    print(f"simulated maps with P(SIDM) within {P_MATCH} of A2744: {n_match}")
    plot_vs_kappa(obs, sims, out / "sensitivity_vs_kappa")
    corr = plot_correlations(out / "sensitivity.npz", opts.window, opts.stride, out / "sensitivity_correlations")
    print("correlations:", json.dumps(corr))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
