#!/usr/bin/env python
'''
Numbers quoted in the paper that need the trained models (final ensemble, 30 seeds):

  noise-floor  A2744 with every galaxy's distortion rotated by a random angle (amplitude and position kept), the
               ensemble-mean P(SIDM) per realisation: the output when the lensing signal is removed but the shape
               noise and galaxy distribution are the data's. The unrotated map is checked against obs.npz first.
               Reads the A2744 catalogue (rebuilds the shared filtered catalogue: run alone).
  rho          correlation between ensemble members (Eq. 8 of the paper, Fig. A2): the fit of
               sigma sqrt((1 - rho)/N + rho) to the RMSE of the mean of N random models (BAHAMAS CDM clusters, every
               shape-noise draw), the mean pairwise correlation of the models' errors, and the spread of both over a
               bootstrap of the clusters. Reads the summaries only.
  a-distance   proxy A-distance between BAHAMAS and DARKSKIES, d_A = 2 (1 - 2 eps), with eps the 5-fold
               cross-validated error of a logistic regression separating the two domains on one model's latent
               features (512-d, standardised); clusters that model never trained on, one shape-noise draw.

    astro-env/bin/python sbi/paper_numbers.py noise-floor --n 1000
    astro-env/bin/python sbi/paper_numbers.py a-distance --per-domain 1500
    astro-env/bin/python sbi/paper_numbers.py rho --n-boot 200
'''
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "notebooks"), str(REPO / "sbi")]
import make_summaries as ms  # noqa: E402  (chdirs to notebooks/)

MODEL, INCLUDE = "tweakedalign_k0p7_ctr_wb1_ba_final", "spin2_cx_pgc_ft_"
OUT = REPO / "sbi" / "results" / "paper_numbers"


def models_and_args():
    device = ms.get_device()
    args = ms.setup_args("concat")
    models = ms.load_models(SimpleNamespace(filter="concat", model=MODEL, include_str=INCLUDE, nmodels=None), args, device)
    print(f"{len(models)} models ({MODEL}), device {device}")
    return models, args, device


def noise_floor(opts):
    import sensitivity as sens
    from add_shear_to_data import galaxy_distortion
    from lenspack.utils import bin2d
    models, _, device = models_and_args()
    raw, cat, ra, dec = sens.a2744_map()
    img, _ = sens.normalise(torch.tensor(raw))
    p0 = sens.model_outputs(models, img[None], device, 1)[:, 0]
    p_obs = np.load(REPO / "sbi" / "summaries_wb1_final_n5" / "obs.npz")["softmax"][:, 0]
    print(f"A2744 unrotated: {p0.mean():.4f} (obs.npz {p_obs.mean():.4f})")
    assert np.allclose(p0, p_obs, atol=2e-3), "A2744 map does not reproduce obs.npz"
    c1, c2 = galaxy_distortion(cat)
    amp = np.hypot(c1, c2)
    rng = np.random.default_rng(opts.seed)
    imgs = []
    for _ in tqdm(range(opts.n), desc="rotated maps"):
        ang = rng.uniform(0, np.pi, len(amp))
        r = np.array(bin2d(ra, dec, v=(amp * np.cos(2 * ang), amp * np.sin(2 * ang)), npix=sens.NPIX,
                           extent=[-50, 50, -50, 50]), dtype=np.float32)
        imgs.append(sens.normalise(torch.tensor(r))[0])
    p = sens.model_outputs(models, torch.stack(imgs), device, opts.batch_size)       # (n_models, n)
    ens = p.mean(0)
    res = {"n_rotations": opts.n, "a2744": float(p0.mean()), "rotated_mean": float(ens.mean()),
           "rotated_std": float(ens.std()), "rotated_16_50_84": np.percentile(ens, [16, 50, 84]).tolist(),
           "rotated_max": float(ens.max()), "fraction_at_or_above_a2744": float(np.mean(ens >= p0.mean())),
           "a2744_sigma_above_floor": float((p0.mean() - ens.mean()) / ens.std())}
    np.save(OUT / "noise_floor_ensemble.npy", ens)
    return res


def a_distance(opts):
    from dataset import BalancedClusterDataset, get_test_transform
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    models, args, device = models_and_args()
    rng = np.random.default_rng(opts.seed)
    lat, held, cs = {}, {}, {}
    for d in ms.DOMAINS:
        ds = BalancedClusterDataset(args.data_dir, f"{args.image_size}/obs/concat/{d}*", transform=get_test_transform(args),
                                    args=args, dtype="image")
        fn = np.array([ds.files_data[f]["filename"] for f, _ in ds.file_indices])
        ii = np.array([i for _, i in ds.file_indices])
        h = ms.held_out_mask(models, d, fn, ii, "concat")                                 # (n_models, N)
        pick = rng.choice(np.flatnonzero(h.any(0)), min(opts.per_domain, int(h.any(0).sum())), replace=False)
        np.random.seed(opts.seed); torch.manual_seed(opts.seed)
        x = torch.stack([ds[i][0] for i in tqdm(pick, desc=f"{d} images", leave=False)])
        lat[d] = ms.run_models(models, x, device, opts.batch_size)[0]                     # (n_models, n, 512)
        held[d], cs[d] = h[:, pick], np.asarray(ds.cross_sections)[pick]
        print(f"{d}: {len(pick)} clusters")
    res = {}
    for name, keep in (("all", lambda c: np.ones(len(c), bool)), ("cdm", lambda c: c == 0)):
        dA = []
        for m in range(len(models)):
            X, y = [], []
            for k, d in enumerate(ms.DOMAINS):
                s = held[d][m] & keep(cs[d])
                X.append(lat[d][m][s]); y.append(np.full(s.sum(), k))
            n = min(len(a) for a in X)                                                    # balanced classes
            X = np.concatenate([a[rng.choice(len(a), n, replace=False)] for a in X]); y = np.repeat([0, 1], n)
            acc = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(C=opts.C, max_iter=2000)),
                                  X, y, cv=5).mean()
            dA.append(2 * (1 - 2 * (1 - acc)))
        dA = np.array(dA)
        res[name] = {"mean": float(dA.mean()), "std": float(dA.std()), "per_model": dA.tolist(), "n_per_domain": int(n)}
        print(f"proxy A-distance ({name}): {dA.mean():.3f} +- {dA.std():.3f} over {len(dA)} models")
    return res


def rho(opts):
    from scipy.optimize import curve_fit
    S = REPO / "sbi" / "summaries_wb1_final_n5"
    P = np.load(S / "params_bahamas.npz")
    soft = np.load(S / "softmax_bahamas.npy")[:, :, P["cross_section"] == 0]       # (n_models, n_noise, N)
    M = soft.shape[0]
    Ns = np.arange(1, M + 1)
    law = lambda N, s, r: s * np.sqrt((1 - r) / N + r)                               # noqa: E731
    rng = np.random.default_rng(opts.seed)

    def fit(s, nrandom):                         # as figureA2_summaries in notebooks/plots_for_paper.py
        y = np.broadcast_to(s.mean((0, 1)), s.shape[1:]).ravel()
        pred = s.reshape(M, -1)
        scale = np.std(pred - y, axis=0)
        rmse = np.array([np.mean([np.sqrt(np.mean(((pred[rng.choice(M, N, replace=False)].mean(0) - y) / scale) ** 2))
                                  for _ in range(nrandom)]) for N in Ns])
        (_, r), _ = curve_fit(law, Ns, rmse, p0=[1, 0.1])
        return r, np.polyfit(np.log(Ns), np.log(rmse), 1)[0]

    def pairwise(s):                             # mean correlation between two models' errors about the cluster mean
        c = np.corrcoef((s - s.mean((0, 1), keepdims=True)).reshape(M, -1))
        return c[~np.eye(M, dtype=bool)].mean()

    r0, b0 = fit(soft, 1000)
    boot = []
    for _ in tqdm(range(opts.n_boot), desc="bootstrap over clusters"):
        idx = rng.integers(0, soft.shape[2], soft.shape[2])
        boot.append((*fit(soft[:, :, idx], 100), pairwise(soft[:, :, idx])))
    boot = np.array(boot)
    return {"n_models": M, "n_noise": soft.shape[1], "n_clusters": soft.shape[2],
            "rho_fit": float(r0), "rho_fit_boot_std": float(boot[:, 0].std()),
            "rho_pairwise": float(pairwise(soft)), "rho_pairwise_boot_std": float(boot[:, 2].std()),
            "loglog_slope": float(b0), "loglog_slope_boot_std": float(boot[:, 1].std())}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["noise-floor", "a-distance", "rho"])
    p.add_argument("--n", type=int, default=1000, help="noise-floor: rotations")
    p.add_argument("--per-domain", type=int, default=1500, help="a-distance: clusters drawn per domain")
    p.add_argument("--C", type=float, default=1.0, help="a-distance: inverse regularisation of the classifier")
    p.add_argument("--n-boot", type=int, default=200, help="rho: bootstrap resamples of the clusters")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    opts = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"noise-floor": noise_floor, "a-distance": a_distance, "rho": rho}[opts.what](opts)
    res["settings"] = {**vars(opts), "model": MODEL, "include_str": INCLUDE}
    json.dump(res, open(OUT / f"{opts.what}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "settings"}, indent=1)[:2000])


if __name__ == "__main__":
    main()
