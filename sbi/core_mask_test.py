#!/usr/bin/env python
'''
Does A2744's high P(SIDM) come from its core? Remove the same central disc of galaxies
(pixels set empty before normalisation) from A2744 and from mass-matched BAHAMAS clusters,
and compare P(SIDM) as the disc grows. As a control the same disc is removed at off-centre
positions in A2744.

In the simulations galaxies with kappa >= KAPPA_THRESH are dropped, but A2744 has no such
cut, so its galaxies inside the critical curve (r < ~130 kpc) keep strong / flipped shear.

With --halo (for the median-centred stamp, where A2744's main halo is off-centre) the disc
in A2744 is centred on its main halo, Lenstool O2. The simulated clusters are masked both
on their own main halo (the map centre) and on the same pixels as A2744.

    astro-env/bin/python sbi/core_mask_test.py --model tweakedalign_k0p9_ctr_ba_best \
        --summaries $PWD/sbi/summaries_k0p9_ctr_best_n5
    astro-env/bin/python sbi/core_mask_test.py --halo --model tweakedalign_ba_best \
        --summaries $PWD/sbi/summaries_ba_best_n5
'''
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "notebooks"), str(REPO / "sbi")]

import make_summaries as ms  # noqa: E402  (chdirs to notebooks/)
import sensitivity as sens  # noqa: E402
from mass_weighting import a2744_mass_gaussian  # noqa: E402
from add_shear_to_data import ra_dec_to_simulation_image_pos  # noqa: E402
from obs_map_checks import RG, XG, YG  # noqa: E402

RADII = [0, 3, 5, 7, 10, 15]          # pixels (20 kpc)
OUTER_RADII = [71, 50, 40, 30, 20]    # --outer: keep only r < R
O2_PX = (12949.2, 8049.1)             # A2744 main halo (Lenstool O2) in catalogue pixels


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="tweakedalign_k0p9_ctr_ba_best")
    p.add_argument("--nmodels", type=int, default=10)
    p.add_argument("--n", type=int, default=100, help="matched clusters per cross-section")
    p.add_argument("--summaries", default=str(REPO / "sbi" / "summaries_k0p9_ctr_best_n5"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outer", action="store_true", help="remove everything outside a disc instead")
    p.add_argument("--halo", action="store_true", help="centre A2744's disc on its main halo (O2), not the map centre")
    p.add_argument("--out-dir", default=str(REPO / "sbi" / "results" / "core_mask_test"))
    return p.parse_args()


def masked(raw, mask):
    t = torch.as_tensor(raw, dtype=torch.float32).clone()
    t[:, torch.as_tensor(mask)] = 0.
    return sens.normalise(t)[0]


def main():
    opts = parse()
    out = Path(opts.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = ms.get_device()
    args = ms.setup_args("concat")
    models = ms.load_models(SimpleNamespace(filter="concat", model=opts.model, include_str="nob1",
                                            nmodels=opts.nmodels), args, device)
    raw_a, cat, _, _ = sens.a2744_map()
    if opts.halo:
        hx, hy = (float(v[0]) for v in ra_dec_to_simulation_image_pos(
            {"x": np.array([O2_PX[0]]), "y": np.array([O2_PX[1]])}, ra_0=np.median(cat["x"]), dec_0=np.median(cat["y"])))
        print(f"A2744 main halo at stamp ({hx:.1f}, {hy:.1f}) px, {20 * np.hypot(hx, hy):.0f} kpc from the centre")
    else:
        hx, hy = 0., 0.
    RH = np.hypot(XG - hx, YG - hy)
    mu, sig = a2744_mass_gaussian()
    sel = sens.pick_clusters(opts.summaries, "bahamas", mu, sig, opts.n, np.random.default_rng(opts.seed))
    raws, _ = sens.simulated_maps(sel, args, opts.seed)
    sigmas = np.unique(sel["cross_section"])

    radii = OUTER_RADII if opts.outer else RADII
    res = {"model": opts.model, "n_models": len(models), "outer": opts.outer, "halo_px": [hx, hy],
           "radii_kpc": [20 * r for r in radii], "rows": []}
    ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    for r in radii:
        mask = RG >= r if opts.outer else RG < r
        mask_a = RH >= r if opts.outer else RH < r
        p_a = float(sens.model_outputs(models, masked(raw_a, mask_a)[None], device, 1).mean())
        # control: the same disc 25 px (500 kpc) from the halo
        ctrl = [masked(raw_a, np.hypot(XG - hx - 25 * np.cos(a), YG - hy - 25 * np.sin(a)) < r) for a in ang]
        p_c = sens.model_outputs(models, torch.stack(ctrl), device, 8).mean(0) if r and not opts.outer else np.array([p_a])
        p_s = sens.model_outputs(models, torch.stack([masked(x, mask) for x in raws]), device, 64).mean(0)
        row = {"r_kpc": 20 * r, "a2744": p_a, "a2744_offcentre_disc": [float(p_c.mean()), float(p_c.std())]}
        for s in sigmas:
            v = p_s[sel["cross_section"] == s]
            row[f"bahamas_{s:g}"] = [float(v.mean()), float(v.std() / np.sqrt(len(v))), float((v >= p_a).mean())]
        if opts.halo:
            p_p = sens.model_outputs(models, torch.stack([masked(x, mask_a) for x in raws]), device, 64).mean(0)
            for s in sigmas:
                v = p_p[sel["cross_section"] == s]
                row[f"bahamas_{s:g}_a2744_pixels"] = [float(v.mean()), float(v.std() / np.sqrt(len(v))), float((v >= p_a).mean())]
        res["rows"].append(row)
        print(f"{'keep r<' if opts.outer else 'mask r<'}{20*r:4d} kpc  A2744 {p_a:.3f}  (off-centre disc {p_c.mean():.3f}±{p_c.std():.3f})  " +
              "  ".join(f"σ={s:g}: {row[f'bahamas_{s:g}'][0]:.3f} (≥A2744 {row[f'bahamas_{s:g}'][2]:.2f})" for s in sigmas))
        if opts.halo:
            print(" " * 22 + "sims masked on A2744's pixels: " + "  ".join(
                f"σ={s:g}: {row[f'bahamas_{s:g}_a2744_pixels'][0]:.3f} (≥A2744 {row[f'bahamas_{s:g}_a2744_pixels'][2]:.2f})" for s in sigmas))
    tag = opts.model + ("_outer" if opts.outer else "") + ("_halo" if opts.halo else "")
    json.dump(res, open(out / f"{tag}.json", "w"), indent=1)

    plt = sens.style()
    fig, ax = plt.subplots(figsize=(6, 4.2), constrained_layout=True)
    rk = [x["r_kpc"] for x in res["rows"]]
    for s, c in zip(sigmas, ["#56B4E9", "#3a8fc7", "#1f6aa5", "#0b3d6e"]):
        m = np.array([x[f"bahamas_{s:g}"][:2] for x in res["rows"]])
        ax.errorbar(rk, m[:, 0], m[:, 1], color=c, marker="o", ms=4, label=f"BAHAMAS σ/m={s:g}" if s else "BAHAMAS CDM")
        if opts.halo:
            m = np.array([x[f"bahamas_{s:g}_a2744_pixels"][:2] for x in res["rows"]])
            ax.errorbar(rk, m[:, 0], m[:, 1], color=c, marker="o", ms=4, ls=":", mfc="none",
                        label="  same pixels as A2744" if s == sigmas[-1] else None)
    ax.plot(rk, [x["a2744"] for x in res["rows"]], "o-", color="k", lw=2, label="A2744")
    oc = np.array([x["a2744_offcentre_disc"] for x in res["rows"]])
    if not opts.outer:
        ax.errorbar(rk, oc[:, 0], oc[:, 1], color="#6b6b6b", ls="--", marker="s", ms=4,
                    label="A2744, disc 500 kpc from " + ("the main halo" if opts.halo else "the centre"))
    ax.set_xlabel("radius kept [kpc]" if opts.outer else
                  "radius of disc removed around the main halo [kpc]" if opts.halo else "radius of central disc removed [kpc]")
    ax.set_ylabel("ensemble P(SIDM)")
    ax.set_title(f"{opts.model}: {len(models)} models, {opts.n} mass-matched BAHAMAS clusters "
                 f"per σ/m with n_comp ≥ {min(sens.A2744_NCOMP)}", fontsize=8)
    ax.legend(fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{tag}.{ext}", dpi=150)


if __name__ == "__main__":
    main()
