#!/usr/bin/env python
'''
Where do the min / max pixels that normalise the CNN input come from, (x - min) / max over both
channels, in A2744 and in the simulations, and does the shape-noise composition matter?

dataset.apply_intrinsic_ell used to combine (fixed since) the reduced shear g and the pixel-mean source distortion
chi_s component by component: the numerator has sample**2 * conj(chi_s) (component-wise squares,
not the complex g^2) and the denominator 1 + |g|^2 + 2 g1 chi_s1 (missing g2 chi_s2). The correct
composition (complex, as for real galaxies) is

    chi = (chi_s + 2 g + g^2 conj(chi_s)) / (1 + |g|^2 + 2 Re(g conj(chi_s)))

which keeps |chi| <= 1. Here every simulated map gets the same noise draw both ways ("as trained"
and "complex"), and the outputs are compared, with A2744 (whose pixels are means of measured
chi, so |chi| <= 1) for reference:

    1  extremes   value of min and max, |chi| there, galaxies in that pixel, distance from the
                  centre; fraction of pixels with a component beyond +-1
    2  CNN        P(SIDM) of mass-matched BAHAMAS clusters (n_comp >= 2) per cross-section with
                  the noise as trained and composed correctly; and A2744 normalised with the
                  simulations' typical (min, max) instead of its own

    astro-env/bin/python sbi/check_normalisation.py --include-str spin2_ft_
'''
import argparse
import json
import pickle as pkl
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torchvision import transforms

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "notebooks"), str(REPO / "sbi")]

import make_summaries as ms  # noqa: E402  (chdirs to notebooks/)
import sensitivity as sens  # noqa: E402
from dataset import apply_intrinsic_ell  # noqa: E402
from mass_weighting import a2744_mass_gaussian  # noqa: E402

NPIX = 100
_c = np.arange(NPIX) - (NPIX / 2 - 0.5)
RG = np.hypot(*np.meshgrid(_c, _c))


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="tweakedalign_k0p9_ctr_ba_best")
    p.add_argument("--include-str", default="spin2_ft_")
    p.add_argument("--nmodels", type=int, default=None)
    p.add_argument("--n", type=int, default=100, help="matched clusters per cross-section")
    p.add_argument("--summaries", default=str(REPO / "sbi" / "summaries_spin2_best_n5"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=str(REPO / "sbi" / "results" / "check_normalisation"))
    return p.parse_args()


def compose(g, chi_s):
    '''Complex composition of reduced shear g and source distortion chi_s, (2, n, n) tensors.'''
    gc = torch.complex(g[0].double(), g[1].double())
    sc = torch.complex(chi_s[0].double(), chi_s[1].double())
    out = (sc + 2 * gc + gc ** 2 * sc.conj()) / (1 + gc.abs() ** 2 + 2 * (gc * sc.conj()).real)
    out = torch.where((g != 0).any(0), out, torch.zeros_like(out))
    return torch.stack([out.real, out.imag]).float()


def extremes(raw, ngal):
    '''Min / max of the map over both channels and where they sit.'''
    raw = np.asarray(raw)
    out = {}
    for k, f in (("min", np.argmin), ("max", np.argmax)):
        c, iy, ix = np.unravel_index(f(raw), raw.shape)
        out[k] = {"value": float(raw[c, iy, ix]), "channel": int(c), "abs_chi": float(np.hypot(*raw[:, iy, ix])),
                  "ngal": int(ngal[iy, ix]), "r_kpc": float(20 * RG[iy, ix])}
    occ = np.any(raw != 0, axis=0)
    out["frac_pixels_beyond_1"] = float(np.mean(np.any(np.abs(raw[:, occ]) > 1, axis=0)))
    return out


def summarise(rows):
    keys = [("min", "value"), ("max", "value"), ("min", "ngal"), ("max", "ngal"), ("min", "abs_chi"),
            ("max", "abs_chi"), ("min", "r_kpc"), ("max", "r_kpc")]
    s = {f"{a}_{b}": np.percentile([r[a][b] for r in rows], [16, 50, 84]).round(3).tolist() for a, b in keys}
    s["frac_pixels_beyond_1"] = float(np.mean([r["frac_pixels_beyond_1"] for r in rows]))
    return s


def main():
    opts = parse()
    out = Path(opts.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(opts.seed)
    device = ms.get_device()
    args = ms.setup_args("concat")
    models = ms.load_models(SimpleNamespace(filter="concat", model=opts.model, include_str=opts.include_str,
                                            nmodels=opts.nmodels), args, device)
    print(f"{len(models)} models: {opts.model} containing '{opts.include_str}'")

    steps = ms.get_test_transform(args).transforms
    i_ell = next(i for i, t in enumerate(steps) if isinstance(t, apply_intrinsic_ell))
    pre, ell, post = transforms.Compose(steps[:i_ell]), steps[i_ell], transforms.Compose(steps[i_ell + 1:])
    flat = ell.binned_data["flat_bin"]
    ngal = np.bincount(flat, minlength=NPIX * NPIX).reshape(NPIX, NPIX)

    # A2744
    raw_a, _, _, _ = sens.a2744_map()
    res = {"a2744": extremes(raw_a, ngal)}
    p_a = float(sens.model_outputs(models, sens.normalise(torch.tensor(raw_a))[0][None], device, 1).mean())
    res["a2744"]["p_sidm"] = p_a

    # simulations: the same noise draw composed as trained and correctly
    sens.A2744_NCOMP = range(2, 7)
    mu, sig = a2744_mass_gaussian()
    sel = sens.pick_clusters(opts.summaries, "bahamas", mu, sig, opts.n, rng)
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    maps = {"trained": [], "complex": []}
    for f in np.unique(sel["filename"]):
        rows = np.where(sel["filename"] == f)[0]
        _, obs = pkl.load(open(f"../data/100/obs/concat/{f}", "rb"))
        for r in rows:
            g = pre(torch.tensor(obs[sel["image_idx"][r], :2], dtype=torch.float32))
            chi_s = ell(torch.full_like(g, 1e-12))                 # noise only (exact zeros are masked)
            if not maps["trained"]:
                torch.manual_seed(opts.seed + 1)
                np.random.seed(opts.seed + 1)
                ref = ell(g.clone())
                torch.manual_seed(opts.seed + 1)
                np.random.seed(opts.seed + 1)
                chk = ell_as_trained(g, ell(torch.full_like(g, 1e-12)))
                if not torch.allclose(ref, chk, atol=1e-5):         # dataset.py now composes correctly
                    torch.manual_seed(opts.seed + 1)
                    np.random.seed(opts.seed + 1)
                    assert torch.allclose(ref, compose(g, ell(torch.full_like(g, 1e-12))), atol=1e-5)
            maps["trained"].append((r, post(ell_as_trained(g, chi_s))))
            maps["complex"].append((r, post(compose(g, chi_s))))
    order = np.argsort([r for r, _ in maps["trained"]])
    cs = sel["cross_section"][np.sort([r for r, _ in maps["trained"]])]
    p = {}
    for k in maps:
        m = [maps[k][j][1] for j in order]
        res[f"sims_{k}"] = {f"{c:g}": summarise([extremes(m[i].numpy(), ngal) for i in np.where(cs == c)[0]])
                            for c in np.unique(cs)}
        p[k] = sens.model_outputs(models, torch.stack([sens.normalise(x)[0] for x in m]), device, 64).mean(0)
    res["cnn"] = {k: {f"{c:g}": [float(v[cs == c].mean()), float(v[cs == c].std() / np.sqrt((cs == c).sum()))]
                      for c in np.unique(cs)} for k, v in p.items()}
    res["cnn"]["a2744"] = p_a

    # A2744 normalised with the simulations' median (min, max)
    for k in maps:
        mn = np.median([s["min_value"][1] for s in res[f"sims_{k}"].values()])
        mx = np.median([s["max_value"][1] for s in res[f"sims_{k}"].values()])
        img = (torch.tensor(raw_a) - mn) / mx
        res["cnn"][f"a2744_with_sim_{k}_minmax"] = [float(mn), float(mx), float(
            sens.model_outputs(models, img[None], device, 1).mean())]

    json.dump(res, open(out / "check_normalisation.json", "w"), indent=1)
    report(res)


def ell_as_trained(g, chi_s):
    '''dataset.apply_intrinsic_ell's composition, given its noise map chi_s.'''
    conj = chi_s.clone()
    conj[1] *= -1
    gg = torch.sqrt(torch.sum(g ** 2, axis=0))
    out = (chi_s + 2. * g + g ** 2 * conj) / (1 + gg ** 2 + 2. * (g * conj)[0])
    out[g == 0] = 0
    return out


def report(res):
    a = res["a2744"]
    print("\nA2744: " + " | ".join(f"{k} {a[k]['value']:+.3f} (|chi| {a[k]['abs_chi']:.3f}, {a[k]['ngal']} gal, "
                                    f"r {a[k]['r_kpc']:.0f} kpc)" for k in ("min", "max"))
          + f" | pixels beyond ±1: {a['frac_pixels_beyond_1']:.4f}")
    for k in ("trained", "complex"):
        print(f"\nsimulations, noise composed {k} (16/50/84%):")
        for c, s in res[f"sims_{k}"].items():
            print(f"  σ={c:<4s} min {s['min_value'][1]:+.3f} [{s['min_value'][0]:+.3f}, {s['min_value'][2]:+.3f}] "
                  f"ngal {s['min_ngal'][1]:.0f} | max {s['max_value'][1]:+.3f} [{s['max_value'][0]:+.3f}, "
                  f"{s['max_value'][2]:+.3f}] ngal {s['max_ngal'][1]:.0f} | pixels beyond ±1: {s['frac_pixels_beyond_1']:.4f}")
    print("\nCNN P(SIDM):")
    for k in ("trained", "complex"):
        print(f"  BAHAMAS, noise {k:8s}: " + " ".join(f"σ={c} {v[0]:.3f}±{v[1]:.3f}" for c, v in res["cnn"][k].items()))
    print(f"  A2744 (own min/max): {res['cnn']['a2744']:.3f}")
    for k in ("trained", "complex"):
        mn, mx, pp = res["cnn"][f"a2744_with_sim_{k}_minmax"]
        print(f"  A2744 normalised with the simulations' ({k}) median min {mn:+.3f}, max {mx:+.3f}: {pp:.3f}")


if __name__ == "__main__":
    main()
