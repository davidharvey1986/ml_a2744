#!/usr/bin/env python
'''
A2744's ensemble P(SIDM) against the photo-z cut used to remove cluster members and
foreground galaxies (add_shear_to_data.get_obs_data redshift_cut: galaxies matched to an
UNCOVER photo-z below it are dropped; unmatched galaxies are always kept; fiducial
z_cl + 3 x 0.05 = 0.455). The map is rebuilt as make_obs_maps (SHEAR_CUTS, strong-lensing
kappa >= 0.9 dropped) for each cut. Only the data change: the simulations are not
re-processed, so a rising source redshift is not matched in them.

    astro-env/bin/python sbi/redshift_cut.py --include-str spin2
'''
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from lenspack.utils import bin2d

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "notebooks"), str(REPO / "sbi")]

import make_summaries as ms  # noqa: E402  (chdirs to notebooks/)
import sensitivity as sens  # noqa: E402
from add_shear_to_data import SHEAR_CUTS, galaxy_distortion, get_obs_data, normalise_map, ra_dec_to_simulation_image_pos  # noqa: E402

FIDUCIAL = 0.305 + 0.05 * 3


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="tweakedalign_k0p9_ctr_ba_best")
    p.add_argument("--include-str", default="spin2")
    p.add_argument("--nmodels", type=int, default=None)
    p.add_argument("--cuts", type=float, nargs="+",
                   default=[0., 0.35, FIDUCIAL, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0])
    p.add_argument("--out-dir", default=str(REPO / "sbi" / "results" / "redshift_cut"))
    return p.parse_args()


def main():
    opts = parse()
    out = Path(opts.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = ms.get_device()
    models = ms.load_models(SimpleNamespace(filter="concat", model=opts.model, include_str=opts.include_str,
                                            nmodels=opts.nmodels), ms.setup_args("concat"), device)
    rows = []
    for zc in opts.cuts:
        cat = get_obs_data("concat", data_dir="../data/100/a2744/", cuts=SHEAR_CUTS, drop_strong_lensing=True,
                           photoz=True, redshift_cut=zc)
        x, y = ra_dec_to_simulation_image_pos(cat)
        c1, c2 = galaxy_distortion(cat)
        raw = np.array(bin2d(x, y, v=(c1, c2), npix=100, extent=[-50, 50, -50, 50]))
        img = torch.tensor(np.array(normalise_map(*raw))[None], dtype=torch.float32)
        p = sens.model_outputs(models, img, device, 1)[:, 0]
        inside = (np.abs(x) < 50) & (np.abs(y) < 50)
        z = cat["redshift"][inside]
        rows.append({"redshift_cut": zc, "n_gal": int(inside.sum()), "median_z": float(np.median(z)),
                     "p_sidm": float(p.mean()), "p_sidm_model_std": float(p.std()),
                     "p_sidm_sem": float(p.std() / np.sqrt(len(p)))})
        print(f"z_cut {zc:5.3f}: {inside.sum():5d} galaxies in the stamp, median z {np.median(z):.2f}, "
              f"P(SIDM) {p.mean():.3f} ± {p.std() / np.sqrt(len(p)):.3f} (model std {p.std():.3f})")
    json.dump({"model": opts.model, "include_str": opts.include_str, "n_models": len(models), "rows": rows},
              open(out / f"{opts.model}_{opts.include_str}.json", "w"), indent=1)

    plt = sens.style()
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    zc = np.array([r["redshift_cut"] for r in rows])
    ax.errorbar(zc, [r["p_sidm"] for r in rows], [r["p_sidm_sem"] for r in rows], fmt="o-", color="k", capsize=2)
    ax.axvline(FIDUCIAL, color="#D55E00", ls="--", lw=1.2, label=f"fiducial cut {FIDUCIAL:.3f}")
    for r in rows:
        ax.annotate(f"{r['n_gal']}", (r["redshift_cut"], r["p_sidm"]), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7, color="0.4")
    ax.set_xlabel("photo-z cut (matched galaxies below it removed)")
    ax.set_ylabel("A2744 ensemble P(SIDM)")
    ax.set_title(f"{opts.model} ({opts.include_str}, {len(models)} models); grey: galaxies in the stamp", fontsize=9)
    ax.legend(fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{opts.model}_{opts.include_str}.{ext}", dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
