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
    p.add_argument("--summaries", default=str(REPO / "sbi" / "summaries_ba_best_n5"),
                   help="any summaries dir: only cluster masses / n_comp / file indices are used")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--excess-map", default=str(EXCESS_MAP), help="Lenstool excess_kappa.fits ('' to skip)")
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


def pick_clusters(summaries, domain, mu, sig, k, rng):
    '''k clusters per sigma/m, drawn with probability proportional to the A2744 mass / merger weights.'''
    P = dict(np.load(Path(summaries) / f"params_{domain}.npz"))
    cs, m, nc = P["cross_section"], P["log10_mass_750"], np.minimum(P["ncomponents"], 6)
    picks = []
    for c in np.unique(cs):
        rows = np.where((cs == c) & np.isin(nc, A2744_NCOMP) & np.isfinite(m))[0]
        w = gaussian_mass_weights(m[rows], mu, sig)
        picks.append(rng.choice(rows, min(k, len(rows)), replace=False, p=w / w.sum()))
    rows = np.concatenate(picks)
    return {"filename": P["filename"][rows], "image_idx": P["image_idx"][rows], "cross_section": cs[rows],
            "log10_mass_750": m[rows], "ncomponents": P["ncomponents"][rows]}


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


def plot_vs_kappa(obs, sims, path, bins):
    plt = style()
    xc = 0.5 * (bins[1:] + bins[:-1])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for d in DOMAINS:
        if d not in sims:
            continue
        sig_vals = sorted(sims[d])
        shades = np.linspace(0.35, 0.95, len(sig_vals))
        cmap = plt.cm.Blues if d == "bahamas" else plt.cm.Oranges
        for s, sh in zip(sig_vals, shades):
            frac, ferr, mean, merr, n = binned_fraction(*sims[d][s], bins)
            ok = n >= 20
            lab = f"{d.upper()} " + ("CDM" if s == 0 else f"{s:g}")
            axes[0].errorbar(xc[ok], frac[ok], ferr[ok], fmt="o-", ms=4, lw=1.5, capsize=2, color=cmap(sh), label=lab)
            axes[1].errorbar(xc[ok], mean[ok], merr[ok], fmt="o-", ms=4, lw=1.5, capsize=2, color=cmap(sh))
    frac, ferr, mean, merr, n = binned_fraction(*obs, bins)
    ok = n >= 5
    axes[0].errorbar(xc[ok], frac[ok], ferr[ok], fmt="s", ms=7, color="k", capsize=2, label="A2744", zorder=5)
    for ax in axes:
        ax.set_xscale("linear")
    axes[1].errorbar(xc[ok], mean[ok], merr[ok], fmt="s", ms=7, color="k", capsize=2, zorder=5)
    axes[0].axhline(0.5, color=MUTED, lw=1)
    axes[1].axhline(0, color=MUTED, lw=1)
    axes[0].set_ylabel("Fraction of windows pushing towards SIDM")
    axes[1].set_ylabel(r"Mean $\Delta P$(SIDM) when removed")
    for ax in axes:
        ax.set_xlabel(r"Reconstructed $\kappa$ (window mean)")
    axes[0].legend(fontsize=7, ncols=2, loc="best")
    fig.suptitle("Where the CNNs find SIDM signal: A2744 vs matched simulations (same pipeline)",
                 color=INK, fontsize=11)
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=180)


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
def main():
    opts = parse()
    out = LAUNCH_DIR / (opts.out_dir or REPO / "sbi" / "results" / f"sensitivity_{opts.model}")
    out.mkdir(parents=True, exist_ok=True)
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
    obs = (window_mean(obs_kappa, opts.window, opts.stride).ravel(), delta.mean(0).ravel())
    used = obs[1] != 0
    print(f"A2744: P(SIDM) = {full.mean():.3f}; windows pushing to SIDM: {np.mean(obs[1][used] > 0):.2f} "
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
    for d in DOMAINS:
        sel = pick_clusters(opts.summaries, d, mu, sig, opts.clusters_per_sigma, rng)
        raws, kappas_true = simulated_maps(sel, args, opts.seed)
        sims[d] = {}
        deltas = []
        for i in tqdm(range(len(raws)), desc=f"{d} clusters"):
            img, fill = normalise(raws[i])
            dl, p_full = occlusion_map(models, img, fill, opts.window, opts.stride, device, opts.batch_size)
            k_rec, k_snr = reconstruct_kappa(raws[i].numpy())
            s = float(sel["cross_section"][i])
            kw, dw = window_mean(k_rec, opts.window, opts.stride).ravel(), dl.mean(0).ravel()
            prev = sims[d].get(s, (np.array([]), np.array([])))
            sims[d][s] = (np.r_[prev[0], kw], np.r_[prev[1], dw])
            deltas.append(dl.mean(0))
            # example panels: the first matched CDM and largest-sigma cluster of bahamas
            if d == "bahamas" and (s == 0 or s == sel["cross_section"].max()) and \
                    not any(pn.get("sigma") == s for pn in panels[1:]):
                panels.append({"title": f"BAHAMAS " + ("CDM" if s == 0 else f"$\\sigma/m={s:g}$")
                               + f"  (P = {p_full.mean():.2f})", "delta": dl.mean(0), "kappa_snr": k_snr,
                               "kappa_true": kappas_true[i], "sigma": s})
        saved[f"{d}_delta"] = np.array(deltas)
        saved[f"{d}_sigma"] = sel["cross_section"]
        saved[f"{d}_log10_mass_750"] = sel["log10_mass_750"]
        print(f"{d}: {len(raws)} matched clusters; fraction of windows pushing to SIDM by sigma/m: "
              + ", ".join(f"{s:g}: {np.mean(v[1][v[1] != 0] > 0):.2f}" for s, v in sorted(sims[d].items())))

    np.savez(out / "sensitivity.npz", **saved)
    json.dump({"model": opts.model, "n_models": len(models), "window": opts.window, "stride": opts.stride,
               "clusters_per_sigma": opts.clusters_per_sigma, "a2744_p_sidm": float(full.mean()), **stats},
              open(out / "sensitivity.json", "w"), indent=2)
    plot_maps(panels, centres, opts.stride, out / "sensitivity_maps")
    allk = np.concatenate([obs[0][obs[1] != 0]] + [v[0][v[1] != 0] for d in sims for v in sims[d].values()])
    plot_vs_kappa(obs, sims, out / "sensitivity_vs_kappa", bins=np.quantile(allk[np.isfinite(allk)], np.linspace(0, 1, 11)))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
