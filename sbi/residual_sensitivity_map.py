#!/usr/bin/env python
'''
A2744 occlusion-sensitivity map (sbi/sensitivity.py, mean over its models) with the data - Lenstool model
Kaiser-Squires residual of sbi/lenstool_model_map.py as contours: the model as is, and scaled by the fitted
catalogue/model shear slope.

    astro-env/bin/python sbi/residual_sensitivity_map.py
'''
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "notebooks"), str(REPO / "sbi")]
import sensitivity as sens  # noqa: E402

SENS = REPO / "sbi" / "results" / "sensitivity_wb1_final_w4s1"
MODEL = REPO / "sbi" / "results" / "main_halo_profile" / "lenstool_model_map.npz"
OUT = REPO / "sbi" / "results" / "main_halo_profile" / "residual_sensitivity_map"


def main():
    z, m = np.load(SENS / "sensitivity.npz"), np.load(MODEL)
    delta = z["a2744_delta"].mean(0)
    centres = z["centres"]
    st = centres[1] - centres[0]
    extent = [centres[0] - st / 2, centres[-1] + st / 2] * 2
    vmax = np.percentile(np.abs(delta), 99)
    halos = dict(zip(m["halo_names"], map(tuple, m["halo_xy"])))
    a = float(m["slope"])
    res = {"data − model": (m["ke_data"] - m["ke_model"]) / m["noise"],
           f"data − {a:.2f} × model": (m["ke_data"] - a * m["ke_model"]) / m["noise"]}
    plt = sens.style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), constrained_layout=True)
    E = [-50, 50, -50, 50]
    for ax, (name, r) in zip(axes, res.items()):
        im = ax.imshow(np.ma.masked_equal(delta, 0), origin="lower", extent=extent, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.contour(r, origin="lower", extent=E, levels=[2, 3, 4, 6], colors="k", linewidths=1.2)
        ax.contour(r, origin="lower", extent=E, levels=[-4, -3, -2], colors="k", linewidths=0.8, linestyles="--")
        ax.contour(m["hole"].astype(float), origin="lower", extent=E, levels=[0.5], colors="#2ca02c", linewidths=1)
        for nm, (hx, hy) in halos.items():
            ax.plot(hx, hy, "x", color="#2ca02c", ms=9, mew=2)
            ax.text(hx + 2, hy + 2, nm, color="#2ca02c", fontsize=8)
        ax.set_title(f"A2744: {name} (KS S/N contours)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.errorbar(-35, -45, xerr=10, capsize=2, color="k", lw=1.5)
        ax.text(-35, -44, "400 kpc", ha="center", va="bottom", fontsize=8)
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label(r"$\Delta P$(SIDM) when removed (red: pushes to SIDM)")
    fig.suptitle("Occlusion sensitivity (final, mean of the models), north up, east left. Black: data − Lenstool model "
                 "KS S/N ±2,3,4(,6); green: κ ≥ 0.7 pixels and halos", fontsize=10)
    for e in ("png", "pdf"):
        fig.savefig(f"{OUT}.{e}", dpi=170)
    print(f"wrote {OUT}.png")


if __name__ == "__main__":
    main()
