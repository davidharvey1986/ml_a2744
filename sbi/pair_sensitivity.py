#!/usr/bin/env python
'''
Occlusion sensitivity (sensitivity.occlusion_map) of one SIDM / CDM-twin pair from sbi/analogue_pairs.py: each map
forward-modelled with A2744's galaxies and noise (sensitivity.simulated_maps, --draws noise seeds, Delta averaged
over them and over the models), drawn as sensitivity_maps with the reconstructed kappa S/N (solid) and true kappa
(dotted).

    astro-env/bin/python sbi/pair_sensitivity.py --pair 10
'''
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "notebooks"), str(REPO / "sbi")]
import sensitivity as sens  # noqa: E402  (chdirs to notebooks/)
import make_summaries as ms  # noqa: E402


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs-json", default=str(REPO / "sbi" / "results" / "a2744_analogue_pairs_nc4_all.json"))
    p.add_argument("--pair", type=int, default=10, help="1-based rank in --pairs-json")
    p.add_argument("--model", default="tweakedalign_k0p7_ctr_wb1_ba_final")
    p.add_argument("--include-str", default="spin2_cx_pgc_ft_")
    p.add_argument("--nmodels", type=int, default=5)
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--draws", type=int, default=1, help="noise seeds averaged")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    opts = parse()
    J = json.load(open(opts.pairs_json))
    q = J["pairs"][opts.pair - 1]
    files = (J["settings"]["sidm_file"], J["settings"]["cdm_file"])
    device = ms.get_device()
    args = ms.setup_args("concat")
    models = ms.load_models(SimpleNamespace(filter="concat", model=opts.model, include_str=opts.include_str,
                                            nmodels=opts.nmodels), args, device)
    _, centres = sens.windows(opts.window, opts.stride)
    panels, out_np = [], {}
    for f, key, lab in ((files[0], "sidm_image", "SIDM σ/m=1"), (files[1], "cdm_image", "CDM twin")):
        sel = {"filename": np.array([f]), "image_idx": np.array([q[key]])}
        deltas, fulls = [], []
        for seed in range(opts.draws):
            raws, kappas = sens.simulated_maps(sel, args, seed)
            img, fill = sens.normalise(raws[0])
            d, full = sens.occlusion_map(models, img, fill, opts.window, opts.stride, device, opts.batch_size)
            deltas.append(d.mean(0))
            fulls.append(full)
        delta, full = np.mean(deltas, 0), np.concatenate([np.atleast_1d(x) for x in fulls])
        k_snr = sens.reconstruct_kappa(raws[0].numpy())[1]
        used = delta != 0
        print(f"{lab} image {q[key]}: P(SIDM) {full.mean():.3f} ({len(models)} models x {opts.draws} draws); windows "
              f"pushing to SIDM {np.mean(delta[used] > 0):.2f} of {used.sum()}")
        panels.append({"title": f"{lab}: image {q[key]}  (P = {full.mean():.2f}, n$_{{\\rm comp}}$ = "
                                f"{q['ncomp_sidm' if key == 'sidm_image' else 'ncomp_cdm']})",
                       "delta": delta, "kappa_snr": k_snr, "kappa_true": kappas[0]})
        out_np[f"{key}_delta"], out_np[f"{key}_p"] = delta, full

    out = Path(opts.out or REPO / "sbi" / "results" / f"pair{opts.pair}_sensitivity_w{opts.window}s{opts.stride}")
    np.savez(f"{out}.npz", centres=centres, **out_np)
    plt = sens.style()
    half = opts.stride / 2.
    extent = [centres[0] - half, centres[-1] + half, centres[0] - half, centres[-1] + half]
    vmax = max(np.percentile(np.abs(p["delta"]), 99) for p in panels)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
    for ax, p in zip(axes, panels):
        im = sens.draw_panel(ax, p, extent, vmax)
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label(r"$\Delta P$(SIDM) when removed (red: pushes to SIDM)")
    fig.suptitle(f"Pair {opts.pair}: occlusion sensitivity, {opts.window}×{opts.window} px windows, stride {opts.stride} "
                 f"({len(models)} models, {opts.draws} noise draw{'s' * (opts.draws > 1)}). Black: reconstructed κ S/N "
                 "2,3,4,6,8; dotted: true κ", fontsize=10)
    for e in ("png", "pdf"):
        fig.savefig(f"{out}.{e}", dpi=160)
    print(f"wrote {out}.png")


if __name__ == "__main__":
    main()
