#!/usr/bin/env python
'''
A2744 occlusion sensitivity marginalised over the uncertainty in the stamp centre.

Centres are drawn from the NFW-fit centre likelihood exp(-delta chi^2 / 2) as in
centre_uncertainty.py (offsets from Lenstool O2 on a half-pixel grid). For each draw the
A2744 map is rebuilt about that centre, the occlusion map (sensitivity.occlusion_map) is
computed in its own stamp frame and then put back on the sky: every window's Delta is
painted over its footprint on a grid at 0.5 stamp pixels in the frame of the adopted centre
(A2744_STAMP_CENTRE), so half-pixel centre offsets are exact.

Per sky position the outputs are the mean over draws of Delta / rms (rms of that draw's
occupied windows, as sensitivity.py --sigma-range), and the fraction of draws in which it is
>= 1 sigma and >= 2 sigma towards SIDM and towards CDM: a feature that survives most draws
does not depend on the centring.

    astro-env/bin/python sbi/sensitivity_centre.py --include-str spin2 --ndraws 20
'''
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from lenspack.utils import bin2d
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "notebooks"), str(REPO / "sbi")]

import make_summaries as ms  # noqa: E402  (chdirs to notebooks/)
import sensitivity as sens  # noqa: E402
from add_shear_to_data import (  # noqa: E402
    A2744_LENSTOOL_O2, SHEAR_CUTS, galaxy_distortion, get_obs_data, ra_dec_to_simulation_image_pos,
)

NPIX = 100
SUB = 2                    # sky grid samples per stamp pixel


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="tweakedalign_k0p9_ctr_ba_best")
    p.add_argument("--include-str", default="spin2")
    p.add_argument("--nmodels", type=int, default=None)
    p.add_argument("--ndraws", type=int, default=20)
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--sigma-range", type=float, nargs=2, default=[1, 5])
    p.add_argument("--chi2", default=str(REPO / "sbi" / "results" / "catalogue_comparison" / "nfw_centre_chi2.npz"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=str(REPO / "sbi" / "results" / "sensitivity_centre"))
    return p.parse_args()


def paint(nsig, starts, window, dx, dy):
    '''Window values of one draw on the sky grid (adopted-centre frame), NaN where not covered.'''
    sky = np.full((NPIX * SUB, NPIX * SUB), np.nan)
    for iy, y in enumerate(starts):
        for ix, x in enumerate(starts):
            if nsig[iy, ix] == 0:                               # no galaxies in the window
                continue
            x0, y0 = int(round((x + dx) * SUB)), int(round((y + dy) * SUB))
            xs, ys = slice(max(x0, 0), max(x0 + window * SUB, 0)), slice(max(y0, 0), max(y0 + window * SUB, 0))
            sky[ys, xs] = nsig[iy, ix]
    return sky


def main():
    opts = parse()
    out = Path(opts.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = ms.get_device()
    models = ms.load_models(SimpleNamespace(filter="concat", model=opts.model, include_str=opts.include_str,
                                            nmodels=opts.nmodels), ms.setup_args("concat"), device)
    print(f"{len(models)} models: {opts.model} containing '{opts.include_str}'")

    cat = get_obs_data("concat", data_dir="../data/100/a2744/", cuts=SHEAR_CUTS, drop_strong_lensing=True)
    x, y = ra_dec_to_simulation_image_pos(cat)                  # stamp pixels about A2744_STAMP_CENTRE
    c1, c2 = galaxy_distortion(cat)
    ox, oy = (float(v[0]) for v in ra_dec_to_simulation_image_pos(
        {"x": np.array([A2744_LENSTOOL_O2[0]]), "y": np.array([A2744_LENSTOOL_O2[1]])}))

    z = np.load(opts.chi2)
    gx, gy = np.meshgrid(z["grid"], z["grid"])
    dchi2 = z["chi2map"] - np.nanmin(z["chi2map"])
    w = np.where(np.isfinite(dchi2), np.exp(-0.5 * dchi2), 0.).ravel()
    pick = np.random.default_rng(opts.seed).choice(w.size, opts.ndraws, p=w / w.sum())
    # the adopted centre first, then the draws; offsets (stamp px) of each centre from the adopted one
    cx = np.concatenate([[0.], gx.ravel()[pick] + ox])
    cy = np.concatenate([[0.], gy.ravel()[pick] + oy])

    starts, _ = sens.windows(opts.window, opts.stride)
    skies, p_full = [], []
    for dx, dy in tqdm(list(zip(cx, cy)), desc="centres"):
        raw = np.array(bin2d(x - dx, y - dy, v=(c1, c2), npix=NPIX, extent=[-50, 50, -50, 50]), dtype=np.float32)
        img, fill = sens.normalise(torch.tensor(raw))
        delta, full = sens.occlusion_map(models, img, fill, opts.window, opts.stride, device, opts.batch_size)
        d = delta.mean(0)
        nsig = d / np.sqrt(np.mean(d[d != 0] ** 2))
        # stamp pixel i of this centre sits at i + (dx, dy) in the adopted frame (x along columns, y along rows)
        skies.append(paint(nsig, starts, opts.window, dx, dy))
        p_full.append(float(full.mean()))
    skies = np.array(skies)
    draws = skies[1:]
    n_cov = np.sum(np.isfinite(draws), 0)
    covered = n_cov >= 0.5 * len(draws)
    with np.errstate(invalid="ignore"):
        mean = np.where(covered, np.nanmean(draws, 0), np.nan)
        frac = {f"{lab}_{k:g}sigma": np.where(covered, np.nansum(sgn * draws >= k, 0) / np.maximum(n_cov, 1), np.nan)
                for lab, sgn in (("sidm", 1), ("cdm", -1)) for k in (1, 2)}
    np.savez(out / f"{opts.model}_{opts.include_str}.npz", skies=skies, centres_px=np.c_[cx, cy], p_full=p_full,
             mean_nsigma=mean, **frac)
    summary = {"model": opts.model, "include_str": opts.include_str, "n_models": len(models), "n_draws": opts.ndraws,
               "window": opts.window, "stride": opts.stride, "p_sidm_adopted": p_full[0],
               "p_sidm_draws_mean_std": [float(np.mean(p_full[1:])), float(np.std(p_full[1:]))],
               "centre_offsets_kpc_rms": float(20 * np.sqrt(np.mean(cx[1:] ** 2 + cy[1:] ** 2)))}
    for k, v in frac.items():
        summary[f"area_frac_ge_0.5_{k}"] = float(np.nanmean(v >= 0.5))
    json.dump(summary, open(out / f"{opts.model}_{opts.include_str}.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))

    # figure: adopted centre, centre-marginalised mean, and how often each position passes 1 sigma
    raw0 = np.array(bin2d(x, y, v=(c1, c2), npix=NPIX, extent=[-50, 50, -50, 50]))
    kappa_snr = sens.reconstruct_kappa(raw0)[1]
    vmin, vmax = opts.sigma_range
    plt = sens.style()
    ext = [-50, 50, -50, 50]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.6), constrained_layout=True)
    panels = [("adopted centre", skies[0]), (f"mean over {opts.ndraws} centre draws", mean)]
    for row, (sgn, cmap, name) in enumerate([(1, "Reds", "SIDM"), (-1, "Blues", "CDM")]):
        for col, (title, m) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(np.ma.masked_invalid(np.where(sgn * m >= vmin, sgn * m, np.nan)), origin="lower",
                           extent=ext, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"pushes to {name}: {title}", fontsize=10)
        fig.colorbar(im, ax=axes[row, :2], fraction=0.03, pad=0.01).set_label(r"$\pm\Delta P$ / rms ($\sigma$)")
        ax = axes[row, 2]
        f = frac[f"{name.lower()}_1sigma"]
        im = ax.imshow(np.ma.masked_invalid(np.where(f > 0, f, np.nan)), origin="lower", extent=ext, cmap=cmap,
                       vmin=0, vmax=1)
        ax.set_title(rf"pushes to {name}: fraction of draws $\geq$ {vmin:g}$\sigma$", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    for ax in axes.ravel():
        ax.contour(kappa_snr, origin="lower", extent=ext, levels=[2, 3, 4, 6, 8], colors="k", linewidths=0.8)
        ax.plot(0, 0, "+", color="k", ms=10, mew=1.5)
        ax.plot(ox, oy, "x", color="#2ca02c", ms=8, mew=2)
        ax.plot(cx[1:], cy[1:], ".", color="#2ca02c", ms=3)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].errorbar(-35, -45, xerr=10, capsize=2, color="k", lw=1.5)
    axes[0, 0].text(-35, -44, "400 kpc", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"A2744 occlusion sensitivity marginalised over the centre ({opts.model}, {opts.include_str}, "
                 f"{len(models)} models, {opts.window}x{opts.window} px windows, stride {opts.stride}). "
                 f"+: adopted centre; green x: Lenstool O2; dots: centre draws", fontsize=10)
    for e in ("png", "pdf"):
        fig.savefig(out / f"{opts.model}_{opts.include_str}.{e}", dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
