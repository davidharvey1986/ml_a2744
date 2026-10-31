'''
Redraw sensitivity_maps with 20 matched BAHAMAS clusters (spread evenly over sigma/m) instead of 2 examples.
Reuses the occlusion deltas stored in sensitivity.npz and regenerates the same simulated maps (same seed / selection)
for the kappa contours.   Run from the repo root:  astro-env/bin/python sbi/sensitivity_maps20.py [results_dir]
'''
import json
import sys
from pathlib import Path

import numpy as np

import sensitivity as S

REPO = S.REPO
res = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "sbi" / "results" / "sensitivity_tweakedalign_ba_best"
npz = np.load(res / "sensitivity.npz")
cfg = json.load(open(res / "sensitivity.json"))
seed, k = 0, cfg["clusters_per_sigma"]

rng = np.random.default_rng(seed)
mu, sig = S.a2744_mass_gaussian()
args = S.ms.setup_args("concat")
sels = {d: S.pick_clusters(str(REPO / "sbi" / "summaries_ba_best_n5"), d, mu, sig, k, rng) for d in S.DOMAINS}
sel = sels["bahamas"]
assert np.allclose(sel["cross_section"], npz["bahamas_sigma"]), "selection differs from the stored run"
raws, kappas = S.simulated_maps(sel, args, seed)          # full set, so the random transform matches the stored run
deltas, pfull, sigma = npz["bahamas_delta"], npz["bahamas_p_full"], npz["bahamas_sigma"]

order = np.argsort(sigma, kind="stable")
pick = order[np.unique(np.round(np.linspace(0, len(order) - 1, 20)).astype(int))]
print("clusters:", [(int(i), float(sigma[i]), round(float(pfull[i]), 2)) for i in pick])

plt = S.style()
centres = npz["centres"]
stride = cfg["stride"]
half = stride / 2.
extent = [centres[0] - half, centres[-1] + half, centres[0] - half, centres[-1] + half]
vmax = np.percentile(np.abs(deltas[pick]), 99)
fig, axes = plt.subplots(4, 5, figsize=(4 * 5, 4 * 4 + 0.6), constrained_layout=True)
for ax, i in zip(axes.ravel(), pick):
    _, snr = S.reconstruct_kappa(raws[i].numpy())
    im = ax.imshow(np.ma.masked_equal(deltas[i], 0), origin="lower", extent=extent, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.contour(snr, origin="lower", extent=[-50, 50, -50, 50], levels=[2, 3, 4, 6, 8], colors="k", linewidths=0.9)
    ax.contour(S.gaussian_filter(kappas[i], 1), origin="lower", extent=[-50, 50, -50, 50],
               levels=np.linspace(0.1, 0.8, 8), colors="k", linewidths=0.7, linestyles=":")
    s = sigma[i]
    ax.set_title(("BAHAMAS CDM" if s == 0 else f"BAHAMAS $\\sigma/m={s:g}$") + f"  (P = {pfull[i]:.2f})", fontsize=9, color=S.INK)
    ax.set_xticks([]); ax.set_yticks([])
cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
cbar.set_label(r"$\Delta P$(SIDM) when removed (red: pushes to SIDM)")
fig.suptitle("Occlusion sensitivity, 20 matched BAHAMAS clusters. Black: reconstructed $\\kappa$ S/N 2,3,4,6,8 "
             "(dotted: true $\\kappa$)", color=S.INK, fontsize=11)
for ext in ("png", "pdf"):
    fig.savefig(res / f"sensitivity_maps_20clusters.{ext}", dpi=150)
print("wrote", res / "sensitivity_maps_20clusters.png")
