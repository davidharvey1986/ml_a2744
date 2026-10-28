#!/usr/bin/env python
'''
Choose the shear-catalogue cuts (per filter: lower S/N, lower and upper size, upper
magnitude) that maximise the S/N of A2744's tangential shear about its main halo.

For every set of cuts the pipeline of add_shear_to_data.get_obs_data is repeated in
memory: pyRRG calc_shear on the raw f115w / f150w catalogues (so the G1(S/N) calibration
is refitted on the selected galaxies, stat_type 'snr'), cluster members removed (UNCOVER
z_ml < 0.455 within 0.5"), and the filters combined (galaxies in both get the mean shear;
one-to-one within 2", as stilts tmatch2 find=best). Cluster members are found once on the raw catalogues.

Per set of cuts, using the galaxies in the 100x100 stamp centred on A2744's main halo,
from 100 kpc to 1.4 Mpc (the inner 100 kpc is the strong-lensing region):
  snr_t      measured tangential S/N: sqrt(sum_bins (<chi_t> / sigma_bin)^2), with
             sigma_bin = rms(chi) / sqrt(n) from the selected galaxies themselves
  snr_x      the same for the cross shear (a null test: ~sqrt(n_bins) for noise)
  snr_model  expected S/N for a fixed chi_t(r) (power law fitted to the baseline cuts):
             independent of the measured shears, so not prone to selecting on noise

    astro-env/bin/python sbi/optimise_cuts.py --cats /Users/davidharvey/Work/webbPSF/cats_28_10_26
'''
import argparse
import contextlib
import io
import itertools
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u
from pyRRG.calc_shear import calc_shear

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO)]
from add_shear_to_data import SHEAR_CUTS, ra_dec_to_simulation_image_pos  # noqa: E402

FILTERS = ["f115w", "f150w"]
NAME = "abell2744clu-grizli-v5.4-{}-clear_drc_sci_clean.shears"
FID_CUTS = {"size_cut": [2, 200], "signal_noise_cut": 0, "stat_type": "median", "mag_cut": [0, 30]}   # get_obs_data defaults
MEMBER_Z = 0.305 + 0.05 * 3
R_EDGES = np.array([5, 10, 15, 20, 30, 40, 50, 71])     # pixels (20 kpc)

GRID = {
    "signal_noise_cut": [0, 3, 5, 7, 10, 15],
    "size_lo": [0, 1.5, 2, 2.5, 3, 4],
    "size_hi": [10, 20, 200],
    "mag_hi": [27, 28, 29, 30],
}


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cats", required=True, help="directory with <filter>/" + NAME.format("<filter>") + " or the files themselves")
    p.add_argument("--uncover", default=str(REPO / "data" / "100" / "a2744" / "UNCOVER_DR4_SPS_catalog.fits"))
    p.add_argument("--out-dir", default=str(REPO / "sbi" / "results" / "optimise_cuts"))
    return p.parse_args()


def raw_catalogue(cats, f):
    for path in (Path(cats) / f / NAME.format(f), Path(cats) / NAME.format(f)):
        if path.exists():
            return fits.open(path)[1].data
    raise FileNotFoundError(NAME.format(f) + " in " + cats)


def coords(d, ra="RA", dec="DEC"):
    return SkyCoord(np.asarray(d[ra]) * u.deg, np.asarray(d[dec]) * u.deg)


def mutual_match(a, b, radius):
    '''One-to-one pairs (ia, ib) closer than radius, assigned greedily by separation (as stilts find=best).'''
    ib_all, ia_all, d, _ = a.search_around_sky(b, radius)      # pairs (b index, a index)
    order = np.argsort(d)
    used_a, used_b, ia, ib = set(), set(), [], []
    for k in order:
        i, j = ia_all[k], ib_all[k]
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            ia.append(i)
            ib.append(j)
    return np.array(ia, dtype=int), np.array(ib, dtype=int)


def run_calc_shear(raw, cuts, tmp):
    kw = {**FID_CUTS, **cuts, "verbose": False}
    with contextlib.redirect_stdout(io.StringIO()):
        calc_shear(raw, tmp, **kw)
    return fits.open(tmp)[1].data


def combine(cat):
    '''As combine_catalogues: galaxies in both filters (matched after the cuts) get the mean shear.'''
    a, b = cat["f115w"], cat["f150w"]
    ia, ib = mutual_match(coords(a), coords(b), 2 * u.arcsec)
    extra_a = np.setdiff1d(np.arange(len(a)), ia)
    extra_b = np.setdiff1d(np.arange(len(b)), ib)
    out = {}
    for k in ("x", "y"):
        out[k] = np.r_[a[k][ia], a[k][extra_a], b[k][extra_b]]
    for k in ("gamma1", "gamma2"):
        out[k] = np.r_[(a[k][ia] + b[k][ib]) / 2, a[k][extra_a], b[k][extra_b]]
    return out


def statistics(d, model=None):
    ra, dec = ra_dec_to_simulation_image_pos(d)
    g = np.hypot(d["gamma1"], d["gamma2"])
    c1, c2 = 2 * d["gamma1"] / (1 + g ** 2), 2 * d["gamma2"] / (1 + g ** 2)
    r = np.hypot(ra, dec)
    phi = np.arctan2(dec, ra)
    ct = -(c1 * np.cos(2 * phi) + c2 * np.sin(2 * phi))
    cx = c1 * np.sin(2 * phi) - c2 * np.cos(2 * phi)
    ins = (np.abs(ra) < 50) & (np.abs(dec) < 50) & (r >= R_EDGES[0]) & (r < R_EDGES[-1])
    sig = np.sqrt(0.5 * np.mean(c1[ins] ** 2 + c2[ins] ** 2))
    b = np.digitize(r[ins], R_EDGES) - 1
    n = np.bincount(b, minlength=len(R_EDGES) - 1)
    mt = np.bincount(b, weights=ct[ins], minlength=len(n)) / np.maximum(n, 1)
    mx = np.bincount(b, weights=cx[ins], minlength=len(n)) / np.maximum(n, 1)
    err = sig / np.sqrt(np.maximum(n, 1))
    res = {"n_stamp": int(ins.sum()), "sigma_chi": float(sig), "chi_t": mt, "err": err,
           "snr_t": float(np.sqrt(np.sum((mt / err) ** 2))), "snr_x": float(np.sqrt(np.sum((mx / err) ** 2)))}
    if model is not None:
        res["n_total"] = int(len(ra))
        res["snr_model"] = float(np.sqrt(np.sum(model(r[ins] * 20.) ** 2)) / sig)
    return res


def main():
    opts = parse()
    out = Path(opts.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = {f: raw_catalogue(opts.cats, f) for f in FILTERS}

    uncover = fits.open(opts.uncover)[1].data
    members = {}
    for f in FILTERS:
        i, d, _ = coords(raw[f]).match_to_catalog_sky(coords(uncover, "ra", "dec"))
        members[f] = set(raw[f]["NUMBER"][(d < 0.5 * u.arcsec) & (uncover["z_ml"][i] < MEMBER_Z)])
        print(f"{f}: {len(raw[f])} objects, {len(members[f])} cluster members")

    tmp = tempfile.NamedTemporaryFile(suffix=".fits", delete=False).name
    combos = list(itertools.product(*GRID.values()))
    shears = {f: {} for f in FILTERS}
    for f in FILTERS:
        for c in combos:
            snr, lo, hi, mag = c
            if lo >= hi:
                continue
            try:
                s = run_calc_shear(raw[f], {"signal_noise_cut": snr, "size_cut": [lo, hi], "mag_cut": [0, mag],
                                            "stat_type": "snr"}, tmp)
            except Exception:
                continue
            shears[f][c] = s[~np.isin(s["NUMBER"], list(members[f]))]
        print(f"{f}: {len(shears[f])} cut sets calibrated")
    os.remove(tmp)

    def evaluate(c115, c150, model=None):
        return statistics(combine({"f115w": shears["f115w"][c115], "f150w": shears["f150w"][c150]}), model)

    # baseline: the current SHEAR_CUTS (signal_noise_cut 0 / 5, default size and magnitude limits)
    base = {f: (SHEAR_CUTS[f]["signal_noise_cut"], FID_CUTS["size_cut"][0], FID_CUTS["size_cut"][1], FID_CUTS["mag_cut"][1])
            for f in FILTERS}
    b = evaluate(base["f115w"], base["f150w"])
    rk = 10. * (R_EDGES[1:] + R_EDGES[:-1])
    ok = b["chi_t"] > 0
    slope, amp = np.polyfit(np.log(rk[ok]), np.log(b["chi_t"][ok]), 1)

    def model(r_kpc):
        return np.exp(amp) * r_kpc ** slope

    b = evaluate(base["f115w"], base["f150w"], model)
    print(f"baseline {base}: n {b['n_total']} after cuts (get_obs_data: 16552 for the 28/10 catalogues), {b['n_stamp']} in stamp, S/N_t {b['snr_t']:.2f}, S/N_x {b['snr_x']:.2f}, S/N_model {b['snr_model']:.2f}"
          f"  (model chi_t = {np.exp(amp):.3g} r^{slope:.2f})")

    # coordinate ascent on the measured S/N: each filter's cuts in turn, the other held fixed
    seen = {}

    def row(c115, c150):
        if (c115, c150) not in seen:
            s = evaluate(c115, c150, model)
            seen[(c115, c150)] = {"f115w": c115, "f150w": c150,
                                  **{k: s[k] for k in ("n_stamp", "sigma_chi", "snr_t", "snr_x", "snr_model")}}
        return seen[(c115, c150)]

    cur = dict(base)
    for it in range(4):
        prev = dict(cur)
        cur["f115w"] = max(shears["f115w"], key=lambda c: row(c, cur["f150w"])["snr_t"])
        cur["f150w"] = max(shears["f150w"], key=lambda c: row(cur["f115w"], c)["snr_t"])
        print(f"round {it + 1}: S/N_t {row(cur['f115w'], cur['f150w'])['snr_t']:.2f}  {cur}")
        if cur == prev:
            break
    rows = list(seen.values())
    rows.sort(key=lambda r: -r["snr_t"])
    keys = ("signal_noise_cut", "size_lo", "size_hi", "mag_hi")
    fmt = lambda c: " ".join(f"{k.split('_')[0]}={v:g}" for k, v in zip(keys, c))  # noqa: E731
    print(f"\n{len(rows)} combinations evaluated; top by measured S/N_t")
    for r in rows[:15]:
        print(f"  S/N_t {r['snr_t']:6.2f}  S/N_model {r['snr_model']:6.2f}  S/N_x {r['snr_x']:5.2f}  n {r['n_stamp']:5d}  "
              f"sigma {r['sigma_chi']:.3f} | f115w {fmt(r['f115w'])} | f150w {fmt(r['f150w'])}")
    best_model = max(rows, key=lambda r: r["snr_model"])
    print(f"\nbest by S/N_model: {best_model['snr_model']:.2f} (S/N_t {best_model['snr_t']:.2f}, n {best_model['n_stamp']}) | "
          f"f115w {fmt(best_model['f115w'])} | f150w {fmt(best_model['f150w'])}")
    json.dump({"grid": GRID, "baseline": {"cuts": base, **{k: b[k] for k in ("n_stamp", "sigma_chi", "snr_t", "snr_x", "snr_model")}},
               "model": {"amplitude": float(np.exp(amp)), "slope": float(slope)},
               "rows": [{**r, "f115w": list(r["f115w"]), "f150w": list(r["f150w"])} for r in rows]},
              open(out / "optimise_cuts.json", "w"), indent=1)


if __name__ == "__main__":
    main()
