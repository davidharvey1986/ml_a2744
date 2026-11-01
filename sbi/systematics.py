#!/usr/bin/env python
"""
Systematics tests for the A2744 result (appendix figures A4 and A5), redone on the
new maps with clusters matched to A2744.

Every test perturbs either the forward model of the simulations ("model"
systematics) or both the A2744 data and the simulations in the same way ("data"
systematics), re-runs the CNNs, and reports for each simulation

    log10 sigma/m   median and 16-84% of the posterior from a GP (the paper's kernel)
                    through the ensemble-mean P(SIDM) against log sigma/m, with
                    A2744's ensemble P(SIDM) and its error in the mean over the
                    models (posterior()); flat prior on -3 ... 1, so a flat
                    response gives a posterior that fills the prior (open markers)
    z_CDM           where A2744 sits among CDM clusters (one-sided, in sigma)

with the shift relative to the fiducial. All curves use clusters in A2744's merger
class (n_comp >= 3) weighted to the A2744 mass (mass_weighting.py), so the
comparison is like for like.

Data systematics (A4)
    galaxy selection   lower magnitude, upper magnitude and size cuts on the A2744
                       catalogue; the simulations are re-sampled at the selected
                       galaxies' positions and get shape noise at those positions;
                       plus the cluster-member photo-z cut (galaxies matched to an
                       UNCOVER z_ml < ZCUTS are removed; fiducial ZCUT_FID = 0.455),
                       rebuilding the catalogue with get_obs_data at each cut
    resolution         the maps are averaged to 40, 80, 200 kpc pixels and
                       interpolated back (the Downsample transform), data and
                       simulations alike
    strong-lensing cut galaxies where the convergence is >= KAPPA_CUTS (1.0 ... 0.5;
                       fiducial KAPPA_THRESH) are dropped: from the A2744 map on the
                       Lenstool convergence, from each simulated map on its own
    halo centring      the A2744 stamp centre is moved CENTRING_KPC from the adopted
                       centre in 8 directions (data only: the simulated clusters stay
                       centred, so their outputs are the fiducial ones)
Model systematics (A5)
    cluster-member contamination (shear x (1 - c kappa), c = -0.5 ... 0.5), shape-measurement bias (m, c on e1/e2),
    source-redshift bias (n(z) scaled by 0.75-1.25), mass dependence (A2744 mass
    Gaussian shifted by -2...+2 sigma), merging scenario (n_comp class) and
    projection depth (the same FLAMINGO CDM haloes projected through 25-500 cMpc;
    the change in their mean P(SIDM) relative to 25 cMpc scales every simulated
    cluster's P(SIDM); needs the sbi/dz_cx.sh summaries)

Consistency choices: one catalogue (S/N cuts as prepare_uncovers_data) is used for
the A2744 map, the signal sampling and the shape-noise positions, and the per-pixel
source-redshift map is oriented like the shear maps ([dec, ra]).

Results are cached per test in sbi/results/systematics_<model>/ so interrupted runs
resume; figures: systematics_data.pdf/png (A4) and systematics_model.pdf/png (A5).

Usage
-----
    cd ml_check
    astro-env/bin/python sbi/systematics.py                       # everything
    astro-env/bin/python sbi/systematics.py --tests contamination shape
"""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.stats import binned_statistic_2d, norm
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
LAUNCH_DIR = Path.cwd()
sys.path[:0] = [str(REPO), str(REPO / "notebooks"), str(REPO / "sbi")]

import make_summaries as ms  # noqa: E402  (chdirs to notebooks/, loads the default args)
from add_shear_to_data import (  # noqa: E402
    A2744_STAMP_CENTRE, KAPPA_THRESH, SHEAR_CUTS, data_to_shear, strong_lensing_kappa, get_background_sigma, get_boxsize, get_obs_data,
    get_surface_density_norms,
    ra_dec_to_simulation_image_pos, sample_maps_on_galaxies,
)
from dataset import (  # noqa: E402
    MAP_ZL, MAP_ZS, Downsample, apply_intrinsic_ell, apply_shape_measurement_bias,
    cluster_member_contamination, get_cross_section_from_filename, rescale_lens_source_configuration,
)
from get_model_probabilities import args as default_args  # noqa: E402
from lenspack.utils import bin2d  # noqa: E402
from mass_weighting import a2744_mass_gaussian, effective_sample_size, gaussian_mass_weights  # noqa: E402
from sidm_inference_on_data_and_models import gp as GP_TEMPLATE  # noqa: E402
from add_shear_to_data import return_error_in_mean  # noqa: E402
from sklearn.base import clone  # noqa: E402

DOMAINS = ["bahamas", "darkskies"]
ALL = DOMAINS + ["combined"]           # combined: both simulations pooled, equal weight at each sigma/m
DATA_DIR = "../data/100/a2744/"
FILTER_CUTS = SHEAR_CUTS
NPIX = 100
NCOMP_MAX = 6
A2744_NCOMP = (3, 4, 5, 6)            # n_comp >= 3 (core, NW, N have mass ratios < 10); --min-ncomp
SUBSET_DEX = 0.45                     # clusters within this of the A2744 mass are simulated
EPS = 1e-4
LOG_SIGMA_PRIOR = (-3, 1)              # flat prior on log10 sigma/m (CDM placed at -3, as infer_sidm)
UNCONSTRAINED = 0.5                   # 16-84% wider than this fraction of the prior: flagged unconstrained

TESTS = {
    "galaxy_selection": "data", "resolution": "data", "contamination": "model", "shape": "model",
    "redshift": "model", "mass": "model", "merging": "model", "projection": "model",
    "kappa_cut": "data", "centring": "data",
}
CUT_RANGES = {"low_mag": (21, 24), "high_mag": (28, 26), "size": (2, 3)}   # as the paper (11 cuts each)
ZCUTS = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]   # cluster-member cut: matched galaxies with z_ml < this go
ZCUT_FID = 0.305 + 0.05 * 3                                   # get_obs_data default
CONTAMINATION = [round(c, 2) for c in np.arange(-0.5, 0.51, 0.05) if abs(c) > 1e-6]   # f = c * kappa; c < 0 boosts
SHAPE_BIAS = [(comp, c, m) for c in (0, 0.02) for m in (0, 0.02) for comp in ("e1", "e2") if c or m]
ZS_BIAS = [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25]
DOWNSAMPLE = [2, 4, 5, 10]             # 40, 80, 100, 200 kpc pixels (must divide the 100-pixel map)
MASS_SHIFTS = [-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2]   # in units of the A2744 mass error
MERGING = {"3": (3,), "4+": (4, 5, 6)}         # fiducial is 3+ (with --min-ncomp 2: "2", "3", "4+")
KAPPA_CUTS = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]    # strong-lensing cut; KAPPA_THRESH is the fiducial
CENTRING_KPC = [12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100]   # A2744 stamp centre offsets, each in 8 directions
CENTRING_DIRS = np.arange(0, 360, 45)          # degrees from +RA (stamp x)
DZ_DEPTHS = [25, 50, 100, 200, 250, 500]       # FLAMINGO projection depths [cMpc]; the shallowest is the reference
DZ_MIN_HALOES = 50


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="tweakedalign_ba_best")
    p.add_argument("--include-str", default="nob1", help="only checkpoints whose name contains this (e.g. spin2)")
    p.add_argument("--min-ncomp", type=int, default=3, help="A2744 merger class: n_comp >= this")
    p.add_argument("--tests", nargs="+", default=list(TESTS), choices=list(TESTS))
    p.add_argument("--n-noise", type=int, default=1,
                   help="shape-noise draws per cluster; the same draws are used in every test, so shifts are precise")
    p.add_argument("--cuts-per-sweep", type=int, default=11, help="galaxy-selection cuts per sweep (paper: 11)")
    p.add_argument("--noise-seed", type=int, default=4321)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--summaries", default=str(REPO / "sbi" / "summaries_ta"),
                   help="any summaries dir: only its cluster masses and n_comp are used")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--redo", action="store_true", help="ignore cached test results")
    p.add_argument("--from-cache", action="store_true",
                   help="only summarise and plot cached results (no catalogue or models; missing cases skipped)")
    p.add_argument("--dz-summaries", default=None,
                   help="FLAMINGO projection-depth summaries (sbi/dz_cx.sh; default <summaries>_dz)")
    p.add_argument("--max-clusters", type=int, default=None, help="random subset per simulation (quick checks)")
    return p.parse_args()


# ── A2744 catalogue, maps and geometry ─────────────────────────────────────────
class Catalogue:
    '''The A2744 shear catalogue with simulation-pixel positions about the full-catalogue centre.'''

    def __init__(self, data):
        self.full = data
        self.ra_0, self.dec_0 = A2744_STAMP_CENTRE

    def select(self, kind=None, value=None):
        d = self.full
        if kind == "low_mag":
            d = d[d["MAG"] > value]
        elif kind == "high_mag":
            d = d[d["MAG"] < value]
        elif kind == "size":
            d = d[d["SIZE"] > value]
        return d

    def positions(self, d):
        return ra_dec_to_simulation_image_pos(d, zl=MAP_ZL, image_size=NPIX, ra_0=self.ra_0, dec_0=self.dec_0)


def obs_raw_map(d, ra, dec):
    '''Binned A2744 distortion map (as bin_obs_data, before normalisation), indexed [dec, ra].'''
    g = np.sqrt(d["gamma1"]**2 + d["gamma2"]**2)
    v = (2 * d["gamma1"] / (1 + g**2), 2 * d["gamma2"] / (1 + g**2))
    e1, e2 = bin2d(ra, dec, v=v, npix=NPIX, extent=[-NPIX // 2, NPIX // 2, -NPIX // 2, NPIX // 2])
    return np.array([e1, e2], dtype=np.float32)


def noise_geometry(ra, dec):
    '''Galaxy-to-pixel assignment in the form apply_intrinsic_ell expects.'''
    edges = np.linspace(-NPIX // 2, NPIX // 2, NPIX + 1)
    ix, iy = np.digitize(ra, edges) - 1, np.digitize(dec, edges) - 1
    valid = (ix >= 0) & (ix < NPIX) & (iy >= 0) & (iy < NPIX)
    return {"delta_ra": ra, "valid": valid, "flat_bin": iy[valid] * NPIX + ix[valid]}


def redshift_map(d, ra, dec, default_zs):
    '''Median photo-z per pixel, indexed [dec, ra] like the shear maps; empty pixels get default_zs.'''
    z = binned_statistic_2d(ra, dec, d["redshift"], statistic=np.median, bins=NPIX,
                            range=[[-NPIX // 2, NPIX // 2], [-NPIX // 2, NPIX // 2]]).statistic.T
    z[~np.isfinite(z)] = default_zs
    return z


def normalise(t):
    '''As BalancedClusterDataset.__getitem__: subtract the minimum, divide by the original maximum.'''
    mn, mx = torch.min(t), torch.max(t)
    return (t - mn) / mx


# ── simulations ────────────────────────────────────────────────────────────────
def select_clusters(summaries, domain, mu):
    '''Clusters that can carry weight: A2744's merger class and within SUBSET_DEX of the A2744 mass.'''
    P = dict(np.load(Path(summaries) / f"params_{domain}.npz"))
    keep = (P["ncomponents"] >= min(A2744_NCOMP)) & (np.abs(P["log10_mass_750"] - mu) < SUBSET_DEX)
    return {k: P[k][keep] for k in ("filename", "image_idx", "cross_section", "log10_mass_750", "ncomponents")}


class SimMaps:
    '''Ideal (non-reduced) shear and convergence of the selected clusters, re-sampled on demand.'''

    def __init__(self, sel):
        self.sel = sel
        e1, e2, k = [], [], []
        for f in tqdm(np.unique(sel["filename"]), desc="shear fields", leave=False):
            rows = np.where(sel["filename"] == f)[0]
            conv = f"../data/100/convergence/{f}"
            meta, data = ms.pkl.load(open(conv, "rb"))
            if "redshift" not in meta:
                meta["redshift"] = np.full(data.shape[0], 0.305)
            idx = sel["image_idx"][rows]
            norms = get_surface_density_norms(meta, conv)[idx, 0]
            bg = get_background_sigma(meta, conv, meta["redshift"])[idx]
            a, b, c = data_to_shear(data[idx, 0], norms, meta["redshift"][idx], get_boxsize(conv),
                                    reduce_shear=False, background=bg, zs=MAP_ZS, zl=MAP_ZL)
            e1.append((rows, a)), e2.append((rows, b)), k.append((rows, c))
        n = len(sel["filename"])
        self.e1, self.e2, self.k = (self._assemble(x, n) for x in (e1, e2, k))

    @staticmethod
    def _assemble(parts, n):
        out = np.zeros((n, NPIX, NPIX), dtype=np.float32)
        for rows, a in parts:
            out[rows] = a
        return out

    def sample(self, ra, dec, kappa_thresh=KAPPA_THRESH, with_kappa=False):
        maps, _ = sample_maps_on_galaxies(self.e1, self.e2, self.k, ra, dec, thresh_k=kappa_thresh, reduce_shear=True,
                                          npix=NPIX)
        if with_kappa:                    # binned convergence as a third channel (cluster_member_contamination)
            k, _ = sample_maps_on_galaxies(self.k, self.k, self.k, ra, dec, thresh_k=kappa_thresh, reduce_shear=False,
                                           npix=NPIX)
            maps = np.concatenate([maps, k[:, :1]], axis=1)
        return maps.astype(np.float32)


def make_args(**overrides):
    a = SimpleNamespace(**{k: v for k, v in vars(default_args).items() if not k.startswith("_")})
    a.apply_intrinsic_ell, a.verbose, a.image_size = 1.0, False, NPIX
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


def sim_transforms(a, geometry):
    noise = apply_intrinsic_ell(a)
    noise.binned_data = geometry                         # shape noise at the catalogue's galaxies
    return [rescale_lens_source_configuration(a), cluster_member_contamination(a),
            apply_shape_measurement_bias(a.shape_measurement_bias), noise, Downsample(a)]


@torch.no_grad()
def run_models(models, images, device, batch_size):
    '''P(SIDM), (n_models, N).'''
    out = []
    for imodel in models:
        p = [ms.embed(imodel, images[i:i + batch_size].to(device))[1].cpu() for i in range(0, len(images), batch_size)]
        out.append(torch.cat(p))
    return torch.stack(out).numpy()


def simulate(models, maps, transforms, opts, device):
    '''Forward-model maps (shared noise seeds) and return P(SIDM), (n_models, n_noise, N).'''
    out = []
    for r in range(opts.n_noise):
        np.random.seed(opts.noise_seed + r)
        torch.manual_seed(opts.noise_seed + r)
        imgs = []
        for m in maps:
            t = torch.tensor(m)
            for tf in transforms:
                t = tf(t)
            imgs.append(normalise(t))
        out.append(run_models(models, torch.stack(imgs), device, opts.batch_size))
    return np.stack(out, axis=1)


def observe(models, raw, downsample, device):
    '''P(SIDM) of each model for the A2744 map, (n_models,).'''
    t = torch.tensor(raw)
    if downsample > 1:
        t = Downsample(make_args(downsample=downsample))(t)
    return run_models(models, normalise(t)[None], device, 1)[:, 0]


# ── metrics ────────────────────────────────────────────────────────────────────
def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def weights(sel, mu, sig, ncomp_set=None):
    '''A2744 merger class, weighted to the A2744 mass separately for each cross-section.'''
    ncomp_set = A2744_NCOMP if ncomp_set is None else ncomp_set
    w = np.zeros(len(sel["cross_section"]))
    for c in np.unique(sel["cross_section"]):
        rows = (sel["cross_section"] == c) & np.isin(np.minimum(sel["ncomponents"], NCOMP_MAX), ncomp_set)
        if rows.sum() > 1:
            w[rows] = gaussian_mass_weights(sel["log10_mass_750"][rows], mu, sig)
    return w


def posterior(sigmas, curve, p_obs):
    '''
    Posterior on log10 sigma/m: a GP (the paper's kernel) through the ensemble response curve,
    likelihood N(A2744's ensemble P(SIDM); GP mean, sqrt(GP var + error in the ensemble mean^2)),
    the error in the mean over the models as return_error_in_mean, flat prior LOG_SIGMA_PRIOR.
    A flat response therefore gives a posterior that fills the prior. Returns the median and the
    16-84% range, and flags it unconstrained when that range is wider than UNCONSTRAINED of the prior.
    '''
    gp = clone(GP_TEMPLATE).fit(np.log10(sigmas)[:, None], curve)
    x = np.linspace(*LOG_SIGMA_PRIOR, 2001)
    mu, sd = gp.predict(x[:, None], return_std=True)
    err = float(return_error_in_mean(np.asarray(p_obs, dtype=float)))
    like = norm.pdf(float(np.mean(p_obs)), mu, np.sqrt(sd ** 2 + err ** 2))
    cdf = np.cumsum(like)
    cdf /= cdf[-1]
    lo, med, hi = np.interp([0.16, 0.5, 0.84], cdf, x)
    return {"log_sigma": float(med), "log_sigma_err": [float(med - lo), float(hi - med)],
            "log_sigma_peak": float(x[np.argmax(like)]), "a2744_p_err": err,
            "unconstrained": bool(hi - lo > UNCONSTRAINED * np.ptp(LOG_SIGMA_PRIOR))}


def metrics(p_sim, p_obs, sel, w):
    '''Inferred log10 sigma/m (GP inversion) and A2744's significance against CDM.'''
    cs = sel["cross_section"]
    sigmas = np.unique(cs)
    pm = p_sim.mean(1)                                                   # (n_models, N), averaged over noise draws
    curve = np.array([(pm[:, cs == c] * w[cs == c]).sum(1) / w[cs == c].sum() for c in sigmas])   # (n_sigma, n_models)
    post = posterior(np.where(sigmas == 0, 10. ** LOG_SIGMA_PRIOR[0], sigmas), curve.mean(1), p_obs)
    x = logit(p_sim.mean(0))                                             # (n_noise, N) ensemble logit
    cdm = cs == 0
    wc = np.broadcast_to(w[cdm], x[:, cdm].shape)
    p = float(np.sum(wc * (x[:, cdm] >= logit(p_obs.mean()))) / wc.sum())
    n_eff = effective_sample_size(w[cdm])
    return {**post,
            "z_cdm": float(norm.isf(max(p, 1 / n_eff))), "p_cdm": p, "z_is_lower_limit": bool(p < 1 / n_eff),
            "curve": {f"{c:g}": float(curve[i].mean()) for i, c in enumerate(sigmas)},
            "a2744_p_sidm": float(p_obs.mean())}


# ── tests ──────────────────────────────────────────────────────────────────────
class Runner:
    def __init__(self, opts, out):
        self.opts, self.out = opts, out
        self.mu, self.sig = a2744_mass_gaussian()
        self.sel = {d: select_clusters(opts.summaries, d, self.mu) for d in DOMAINS}
        self.models, self.zcats = [], {}
        if opts.from_cache:
            return
        self.device = ms.get_device()
        cat = get_obs_data("concat", data_dir=DATA_DIR, photoz=True, cuts=FILTER_CUTS)
        self.cat = Catalogue(cat)
        a = make_args()
        self.default_zs = a.default_zs
        self.models = ms.load_models(SimpleNamespace(filter="concat", model=opts.model, include_str=opts.include_str,
                                                     nmodels=None), make_args(), self.device)
        print(f"{len(self.models)} models ({opts.model}), {len(cat)} galaxies, device {self.device}")
        if opts.max_clusters:
            rng = np.random.default_rng(0)
            for d in DOMAINS:
                keep = np.sort(rng.choice(len(self.sel[d]["filename"]), min(opts.max_clusters, len(self.sel[d]["filename"])),
                                          replace=False))
                self.sel[d] = {k: v[keep] for k, v in self.sel[d].items()}
        self.maps = {}
        self.fid_ra, self.fid_dec = self.cat.positions(self.cat.full)

    def sim_maps(self, d):
        if d not in self.maps:
            print(f"  building shear fields for {len(self.sel[d]['filename'])} {d} clusters")
            self.maps[d] = SimMaps(self.sel[d])
        return self.maps[d]

    def zcut_catalogue(self, z):
        '''The catalogue with the cluster-member cut at z_ml < z (rewrites the filtered fits; restore_catalogue
        puts the fiducial back).'''
        if z not in self.zcats:
            self.zcats[z] = get_obs_data("concat", data_dir=DATA_DIR, photoz=True, cuts=FILTER_CUTS, redshift_cut=z)
        return self.zcats[z]

    def restore_catalogue(self):
        if self.zcats:
            get_obs_data("concat", data_dir=DATA_DIR, photoz=True, cuts=FILTER_CUTS)
            self.zcats = {}

    def evaluate(self, d, args_over=None, cut=None, downsample=1, kappa_thresh=None, zcut=None):
        '''Run one configuration for one simulation; returns (p_sim, p_obs, n_gal). n_gal counts the
        catalogue galaxies, or with kappa_thresh given those kept by the strong-lensing cut.'''
        sweep_kappa = kappa_thresh is not None
        kappa_thresh = KAPPA_THRESH if kappa_thresh is None else kappa_thresh
        data = self.zcut_catalogue(zcut) if zcut is not None else self.cat.select(*cut) if cut else self.cat.full
        ra, dec = self.cat.positions(data)
        zs_map = redshift_map(data, ra, dec, self.default_zs)
        over = {"zs": zs_map, "downsample": downsample}
        over.update(args_over or {})
        if "zs_scale" in over:
            over["zs"] = zs_map * over.pop("zs_scale")
        a = make_args(**over)
        maps = self.sim_maps(d).sample(ra, dec, kappa_thresh, with_kappa=a.cluster_member_contamination != 0)
        p_sim = simulate(self.models, maps, sim_transforms(a, noise_geometry(ra, dec)), self.opts, self.device)
        keep = strong_lensing_kappa(data) < kappa_thresh          # data map only; the simulations cut per cluster
        p_obs = observe(self.models, obs_raw_map(data[keep], ra[keep], dec[keep]), downsample, self.device)
        return p_sim, p_obs, int(keep.sum()) if sweep_kappa else len(data)

    def observe_offset(self, dx, dy):
        '''P(SIDM) of each model for the A2744 map with the stamp centre moved by (dx, dy) stamp pixels.'''
        data = self.cat.full
        ra, dec = self.cat.positions(data)
        keep = strong_lensing_kappa(data) < KAPPA_THRESH
        return observe(self.models, obs_raw_map(data[keep], ra[keep] - dx, dec[keep] - dy), 1, self.device)

    def cached(self, name, fn):
        path = self.out / f"{name}.npz"
        if path.exists() and not self.opts.redo:
            z = np.load(path)
            return {d: (z[f"{d}_sim"], z[f"{d}_obs"]) for d in DOMAINS}, int(z["n_gal"])
        if self.opts.from_cache:
            return None, None
        res, n_gal = {}, None
        for d in DOMAINS:
            p_sim, p_obs, n_gal = fn(d)
            res[d] = (p_sim, p_obs)
        np.savez(path, n_gal=n_gal, **{f"{d}_sim": v[0] for d, v in res.items()},
                 **{f"{d}_obs": v[1] for d, v in res.items()})
        return res, n_gal

    def summarise(self, res, mu=None, ncomp_set=None):
        mu = self.mu if mu is None else mu
        w = {d: weights(self.sel[d], mu, self.sig, ncomp_set) for d in DOMAINS}
        out = {d: metrics(res[d][0], res[d][1], self.sel[d], w[d]) for d in DOMAINS}
        # combined: each simulation's weights normalised to 1 at each of its cross-sections, so a shared
        # sigma/m (CDM, 0.1) is the mean of the two curves and the CDM tail counts both equally
        wn = []
        for d in DOMAINS:
            cs, wd = self.sel[d]["cross_section"], w[d].copy()
            for c in np.unique(cs):
                if wd[cs == c].sum() > 0:
                    wd[cs == c] /= wd[cs == c].sum()
            wn.append(wd)
        sel = {k: np.concatenate([self.sel[d][k] for d in DOMAINS]) for k in self.sel[DOMAINS[0]]}
        p_sim = np.concatenate([res[d][0] for d in DOMAINS], axis=-1)
        out["combined"] = metrics(p_sim, res[DOMAINS[0]][1], sel, np.concatenate(wn))
        return out


def run_tests(R, tests):
    out = {}
    fid, n_fid = R.cached("fiducial", lambda d: R.evaluate(d))
    out["fiducial"] = {"metrics": R.summarise(fid), "n_gal": n_fid}
    print_row("fiducial", out["fiducial"]["metrics"])

    def sweep(name, cases):
        out[name] = []
        for label, value, kwargs in cases:
            res, n_gal = R.cached(f"{name}_{label}", lambda d: R.evaluate(d, **kwargs))
            if res is None:
                continue
            out[name].append({"label": label, "value": value, "n_gal": n_gal, "metrics": R.summarise(res)})
            print_row(f"{name} {label}", out[name][-1]["metrics"], out["fiducial"]["metrics"])

    if "galaxy_selection" in tests:
        for kind, (lo, hi) in CUT_RANGES.items():
            cuts = np.linspace(lo, hi, R.opts.cuts_per_sweep)
            sweep(f"galaxy_{kind}", [(f"{v:.2f}", float(v), {"cut": (kind, v)}) for v in cuts])
        sweep("galaxy_redshift", [(f"{z:g}", z, {"zcut": z}) for z in ZCUTS])
        R.restore_catalogue()
    if "kappa_cut" in tests:
        sweep("kappa_cut", [(f"{k:g}", k, {"kappa_thresh": k}) for k in KAPPA_CUTS])
    if "centring" in tests:
        out["centring"] = []
        for r_kpc in CENTRING_KPC:
            for ang in CENTRING_DIRS:
                dx, dy = r_kpc / 20 * np.cos(np.deg2rad(ang)), r_kpc / 20 * np.sin(np.deg2rad(ang))
                label = f"{r_kpc:g}kpc_{ang}deg"
                res, _ = R.cached(f"centring_{label}", lambda d: (fid[d][0], R.observe_offset(dx, dy), n_fid))
                if res is None:
                    continue
                out["centring"].append({"label": label, "value": r_kpc, "angle": float(ang), "n_gal": n_fid,
                                        "metrics": R.summarise(res)})
                print_row(f"centring {label}", out["centring"][-1]["metrics"], out["fiducial"]["metrics"])
    if "resolution" in tests:
        sweep("resolution", [(f"{20 * f}kpc", 20 * f, {"downsample": f}) for f in DOWNSAMPLE])
    if "contamination" in tests:
        sweep("contamination", [(f"{c:g}", c, {"args_over": {"cluster_member_contamination": c}}) for c in CONTAMINATION])
    if "shape" in tests:
        cases = []
        for comp, c, m in SHAPE_BIAS:
            bias = {"e1": {"c": 0, "m": 0}, "e2": {"c": 0, "m": 0}}
            bias[comp] = {"c": c, "m": m}
            cases.append((f"{comp}_c{c:g}_m{m:g}", f"{comp}: c={c:g}, m={m:g}", {"args_over": {"shape_measurement_bias": bias}}))
        sweep("shape", cases)
    if "redshift" in tests:
        sweep("redshift", [(f"{f:g}", f, {"args_over": {"zs_scale": f}}) for f in ZS_BIAS])
    if "mass" in tests:
        out["mass"] = [{"label": f"{k:+g}sigma", "value": k,
                        "metrics": R.summarise(fid, mu=R.mu + k * R.sig)} for k in MASS_SHIFTS]
        for r in out["mass"]:
            print_row(f"mass {r['label']}", r["metrics"], out["fiducial"]["metrics"])
    if "merging" in tests:
        out["merging"] = [{"label": k, "value": k, "metrics": R.summarise(fid, ncomp_set=v)} for k, v in MERGING.items()]
        for r in out["merging"]:
            print_row(f"merging n_comp {r['label']}", r["metrics"], out["fiducial"]["metrics"])
    if "projection" in tests:
        out["projection"] = projection_depth(R, fid)
    return out


def projection_depth(R, fid):
    '''
    As the paper: the same FLAMINGO CDM haloes projected through DZ_DEPTHS cMpc (flamingo_dz<N>), matched
    by their centre of mass; the ratio of their mean ensemble P(SIDM) at each depth to the shallowest is
    applied to every simulated cluster's P(SIDM) (each simulation's curve and its CDM tail), and the
    metrics recomputed. Haloes in A2744's merger class at the shallowest depth (all, if fewer than
    DZ_MIN_HALOES). Needs the summaries from sbi/dz_cx.sh (--dz-summaries).
    '''
    d = Path(R.opts.dz_summaries) if R.opts.dz_summaries else Path(R.opts.summaries + "_dz")
    depths = [z for z in DZ_DEPTHS if (d / f"params_flamingo_dz{z}.npz").exists()]
    if len(depths) < 2:
        print(f"projection depth: no flamingo_dz summaries in {d}, skipped")
        return []
    per = {}
    for z in depths:
        P = np.load(d / f"params_flamingo_dz{z}.npz")
        p = np.load(d / f"softmax_flamingo_dz{z}.npy").mean((0, 1))           # ensemble, mean over noise draws
        meta, _ = ms.pkl.load(open(f"../data/100/convergence/flamingo_dz{z}.pkl", "rb"))
        key = [tuple(np.round(np.asarray(meta["500_crit"][i]["CentreOfMass"], float), 2).ravel()) for i in P["image_idx"]]
        halo = {}
        for k, pi, nc in zip(key, p, P["ncomponents"]):
            halo.setdefault(k, []).append((pi, nc))
        per[z] = {k: (np.mean([a for a, _ in v]), v[0][1]) for k, v in halo.items()}   # projections averaged
    ref = depths[0]
    common = set.intersection(*[set(per[z]) for z in depths])
    cls = [k for k in common if per[ref][k][1] >= min(A2744_NCOMP)]
    use = cls if len(cls) >= DZ_MIN_HALOES else sorted(common)
    print(f"projection depth: {len(common)} haloes at every depth {depths} cMpc; using "
          f"{len(use)} ({'n_comp >= %d' % min(A2744_NCOMP) if use is cls else 'all n_comp'})")
    p_ref = np.array([per[ref][k][0] for k in use])
    rows = []
    for z in depths[1:]:
        pz = np.array([per[z][k][0] for k in use])
        ratio = float(pz.mean() / p_ref.mean())
        res = {dom: (np.clip(fid[dom][0] * ratio, EPS, 1 - EPS), fid[dom][1]) for dom in DOMAINS}
        rows.append({"label": f"{z}cMpc", "value": z, "ratio": ratio, "n_haloes": len(use),
                     "mean_p": float(pz.mean()), "mean_p_ref": float(p_ref.mean()),
                     "per_halo_delta_16_50_84": np.percentile(pz - p_ref, [16, 50, 84]).tolist(),
                     "metrics": R.summarise(res)})
        print_row(f"projection {z} cMpc (x{ratio:.3f})", rows[-1]["metrics"], fid_metrics(R, fid))
    return rows


def fid_metrics(R, fid):
    return R.summarise(fid)


def print_row(name, m, fid=None):
    s = " | ".join(f"{d[:4]} log sigma {m[d]['log_sigma']:+.2f}"
                   + (f" ({m[d]['log_sigma'] - fid[d]['log_sigma']:+.2f})" if fid else "")
                   + f", z_CDM {m[d]['z_cdm']:.2f}" + (f" ({m[d]['z_cdm'] - fid[d]['z_cdm']:+.2f})" if fid else "")
                   for d in ALL)
    print(f"  {name:28s} {s}")


# ── plots ──────────────────────────────────────────────────────────────────────
COLOUR = {"bahamas": "#56B4E9", "darkskies": "#E69F00", "combined": "#222222"}
MARKER = {"bahamas": "o", "darkskies": "s", "combined": "D"}
INK, MUTED = "#222222", "#6b6b6b"


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
                         "axes.grid": True, "grid.color": "#e6e6e6", "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False, "font.size": 10})
    return plt


def plot_data(out, path):
    plt = style()
    sweeps = [("galaxy_low_mag", "Lower magnitude cut (MAG >)"), ("galaxy_high_mag", "Upper magnitude cut (MAG <)"),
              ("galaxy_size", "Size cut (SIZE >)"), ("galaxy_redshift", "Member cut (remove photo-z <)"), ("resolution", "Map resolution [kpc]"),
              ("kappa_cut", "Strong-lensing cut (κ <)"), ("centring", "A2744 centre offset [kpc]")]
    sweeps = [s for s in sweeps if s[0] in out]
    if not sweeps:
        return
    fid = out["fiducial"]
    fig, axes = plt.subplots(2, len(sweeps), figsize=(4.2 * len(sweeps), 6.5), sharey="row",
                             constrained_layout=True, squeeze=False)
    def fid_lines(j):
        for d in ALL:                         # fiducial estimate per simulation
            axes[0, j].axhline(fid["metrics"][d]["log_sigma"], color=COLOUR[d], lw=1, ls="--", alpha=0.7)
        axes[1, j].axhline(0, color=MUTED, lw=1)

    for j, (key, xlabel) in enumerate(sweeps):
        rows = out[key]
        x = np.array([r["value"] for r in rows])
        frac = np.array([r["n_gal"] for r in rows]) / fid["n_gal"]
        if key == "centring":                 # mean over the 8 directions, min-max shaded
            xs = np.unique(x)
            for d in ALL:
                dl = np.array([[r["metrics"][d]["log_sigma"] for r in rows if r["value"] == v] for v in xs])
                dz = np.array([[r["metrics"][d]["z_cdm"] - fid["metrics"][d]["z_cdm"] for r in rows if r["value"] == v]
                               for v in xs])
                for ax, y in ((axes[0, j], dl), (axes[1, j], dz)):
                    ax.fill_between(xs, y.min(1), y.max(1), color=COLOUR[d], alpha=0.2, lw=0)
                    ax.plot(xs, y.mean(1), MARKER[d] + "-", color=COLOUR[d], ms=6, lw=1.8, mec="white",
                            label=d.upper() if ax is axes[0, j] else None)
            fid_lines(j)
            axes[1, j].set_xlabel(xlabel + " (8 directions: mean, range shaded)")
            continue
        for d in ALL:
            dl = np.array([r["metrics"][d]["log_sigma"] for r in rows])
            err = np.array([r["metrics"][d]["log_sigma_err"] for r in rows]).T
            dz = np.array([r["metrics"][d]["z_cdm"] - fid["metrics"][d]["z_cdm"] for r in rows])
            axes[0, j].errorbar(x, dl, err, fmt=MARKER[d] + "-", color=COLOUR[d], ms=6, lw=1.8, capsize=2,
                                mec="white", label=d.upper())
            unc = np.array([r["metrics"][d].get("unconstrained", False) for r in rows])
            axes[0, j].plot(x[unc], dl[unc], MARKER[d], color=COLOUR[d], mfc="white", ms=6, zorder=5)
            axes[1, j].plot(x, dz, MARKER[d] + "-", color=COLOUR[d], ms=6, lw=1.8, mec="white")
        fid_lines(j)
        if key == "kappa_cut":
            axes[1, j].axvline(KAPPA_THRESH, color=MUTED, ls=":", lw=1)
        if key == "galaxy_redshift":
            axes[1, j].axvline(ZCUT_FID, color=MUTED, ls=":", lw=1)
        if key != "resolution":
            for xi, fi in zip(x[::2], frac[::2]):
                axes[1, j].annotate(f"{100 * fi:.0f}%", (xi, 0), xycoords=("data", "axes fraction"),
                                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7, color=MUTED)
        axes[1, j].set_xlabel(xlabel)
    axes[0, 0].set_ylabel(r"$\log_{10}\,\sigma/m$ (A2744 estimate)")
    axes[1, 0].set_ylabel(r"$\Delta z_{\rm CDM}$ [$\sigma$]")
    axes[0, 0].legend(loc="best")
    fig.suptitle("Data systematics: A2744 estimate with posterior 16-84% (dashed: fiducial; open: unconstrained) and "
                 "z_CDM shift from the fiducial ("
                 "grey numbers: fraction of galaxies kept)",
                 color=INK, fontsize=12)
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=180)


def plot_model(out, path):
    plt = style()
    groups = [("contamination", "Cluster-member contamination"), ("shape", "Shape-measurement bias"),
              ("redshift", "Source-redshift bias"), ("mass", "Mass dependence"), ("merging", "Merging scenario"),
              ("projection", "Projection depth")]
    groups = [g for g in groups if out.get(g[0])]
    if not groups:
        return
    fid = out["fiducial"]["metrics"]
    fig, axes = plt.subplots(1, len(ALL), figsize=(6 * len(ALL), 0.9 * len(groups) + 1.8), sharey=True, constrained_layout=True)
    for ax, d in zip(axes, ALL):
        f, (lo, hi) = fid[d]["log_sigma"], fid[d]["log_sigma_err"]
        ax.axvspan(f - lo, f + hi, color="k", alpha=0.08, lw=0)
        ax.axvline(f, color="k", lw=1.5)
        for gi, (key, _) in enumerate(groups):
            rows = out[key]
            shades = plt.cm.Reds(np.linspace(0.35, 0.95, len(rows)))
            offs = np.linspace(-0.3, 0.3, len(rows))
            for r, c, o in zip(rows, shades, offs):
                ax.plot(r["metrics"][d]["log_sigma"], len(groups) - 1 - gi + o, MARKER[d], ms=7, mec=INK, mew=0.5,
                        color=c, mfc="white" if r["metrics"][d].get("unconstrained") else c)
            ax.annotate(f"{rows[0]['label']} … {rows[-1]['label']}", (1, len(groups) - 1 - gi - 0.42),
                        xycoords=("axes fraction", "data"), ha="right", fontsize=7, color=MUTED)
        ax.set_title(f"{d.upper()} (fiducial {f:+.2f}; light→dark = increasing bias)", color=INK)
        ax.set_xlabel(r"$\log_{10}\,\sigma/m$ inferred for A2744")
    axes[0].set_yticks(range(len(groups)))
    axes[0].set_yticklabels([g[1] for g in groups][::-1])
    fig.suptitle("Model systematics", color=INK, fontsize=12)
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=180)


def main():
    opts = parse()
    global A2744_NCOMP, MERGING
    A2744_NCOMP = tuple(range(opts.min_ncomp, 7))
    MERGING = {**({"2": (2,)} if opts.min_ncomp <= 2 else {}), "3": (3,), "4+": (4, 5, 6)}
    out_dir = LAUNCH_DIR / (opts.out_dir or REPO / "sbi" / "results" / f"systematics_{opts.model}")
    out_dir.mkdir(parents=True, exist_ok=True)
    opts.summaries = str((LAUNCH_DIR / opts.summaries).resolve())
    R = Runner(opts, out_dir)
    print(f"  clusters simulated: " + ", ".join(f"{d} {len(R.sel[d]['filename'])}" for d in DOMAINS))
    out = run_tests(R, opts.tests)
    out["settings"] = {"model": opts.model, "n_models": len(R.models) or None, "n_noise": opts.n_noise,
                       "a2744_ncomp": list(A2744_NCOMP), "a2744_log_mass": [R.mu, R.sig], "cuts": FILTER_CUTS}
    json.dump(out, open(out_dir / "systematics.json", "w"), indent=2)
    plot_data(out, out_dir / "systematics_data")
    plot_model(out, out_dir / "systematics_model")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
