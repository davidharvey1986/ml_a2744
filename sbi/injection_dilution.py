#!/usr/bin/env python
'''
Cluster-member dilution of the A2744 JWST shear catalogue from GalSim image injections, a la
Schrabback et al. 2025 (Euclid ERO A2390, arXiv:2507.07629, Sect. 5.4).

Background galaxies (de Vaucouleurs bulge + exponential disk, convolved with a webbpsf PSF) are
injected into copies of cut-outs of the F115W and F150W mosaics, pyRRG is re-run from scratch
exactly as for the live catalogue (cats_28_10_26/run_all.sh: mult 2, psf_lib_jwst_2.0), and the
live selection (SHEAR_CUTS, photo-z member removal, filter combination, kappa < KAPPA_THRESH) is
applied with add_shear_to_data.get_obs_data in a scratch data_dir. The recovered fraction p(kappa)
is the obscuration term; the observed source density of the real catalogue divided by p and by
the magnification depletion mu^(2.5 s - 1) gives the boost B(kappa) = residual member excess.

Steps (each a sub-command; WORK holds every image and catalogue, nothing in data/100/a2744 is touched):
    cutout      sci/wht cut-outs of the live mosaics around the stamp          -> WORK/base/<filter>/
    population  injection population from the live catalogue (mag, size, colour) -> WORK/population.fits
    inject R    inject realisation R into copies of the cut-outs              -> WORK/real_R/<filter>/
    (run_pyrrg_dir.sh on WORK/real_R/<filter>)
    recover R   live selection on realisation R, match to the injections      -> WORK/real_R/recovered.fits
    analyse     p(kappa), p(r), n_obs, magnification, B(kappa) fits, plots     -> sbi/results/injection_dilution/
'''
import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
from astropy import units
from astropy.cosmology import Planck18
from astropy.io import fits
from astropy.wcs import WCS

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sbi"))

WORK = Path("/Users/davidharvey/Work/injection_dilution")
LIVE = Path("/Users/davidharvey/Work/webbPSF/cats_28_10_26")
OUT = REPO / "sbi" / "results" / "injection_dilution"
ROOT = "abell2744clu-grizli-v5.4-{}-clear_drc_sci"
FILTERS = ("f115w", "f150w")
PIX = 0.02                                   # arcsec per mosaic pixel
ZL = 0.305
KPC_PER_PIX = PIX / (units.radian.to(units.arcsec) / Planck18.angular_diameter_distance(ZL).to(units.kpc).value)
STAMP_HALF_KPC = 1000.                       # the 2 Mpc stamp
MARGIN_PIX = 600                             # cut-out margin beyond the stamp (12")
MIN_SEP_PIX = 100                            # minimum separation between injected sources (2")
STAMP_SIZE = 64                              # half-size of an injected postage stamp (pixels)


def asd():
    import add_shear_to_data
    return add_shear_to_data


def cutout_bounds():
    '''(x0, x1, y0, y1) of the cut-out in full-mosaic 0-based pixels (stamp +- 1 Mpc + margin).'''
    cx, cy = asd().A2744_STAMP_CENTRE
    half = STAMP_HALF_KPC / KPC_PER_PIX + MARGIN_PIX
    hdr = fits.getheader(LIVE / "f150w" / (ROOT.format("f150w") + ".fits"))
    nx, ny = hdr["NAXIS1"], hdr["NAXIS2"]
    x0, x1 = max(int(cx - half), 0), min(int(cx + half) + 1, nx)
    y0, y1 = max(int(cy - half), 0), min(int(cy + half) + 1, ny)
    return x0, x1, y0, y1


def cutout():
    x0, x1, y0, y1 = cutout_bounds()
    print(f"cut-out x {x0}:{x1}, y {y0}:{y1} ({x1 - x0} x {y1 - y0}); {KPC_PER_PIX:.4f} kpc/pix")
    for f in FILTERS:
        d = WORK / "base" / f
        d.mkdir(parents=True, exist_ok=True)
        for name in (ROOT.format(f) + ".fits", f"abell2744clu-grizli-v5.4-{f}-clear_drc_wht.fits"):
            out = d / name
            if out.exists():
                continue
            with fits.open(LIVE / f / name, memmap=True) as h:
                hdr = h[0].header.copy()
                data = np.array(h[0].data[y0:y1, x0:x1])
            hdr["CRPIX1"] -= x0
            hdr["CRPIX2"] -= y0
            hdr["CUTX0"], hdr["CUTY0"] = x0, y0
            fits.PrimaryHDU(data, hdr).writeto(out)
            print("wrote", out)
    json.dump({"x0": x0, "x1": x1, "y0": y0, "y1": y1}, open(WORK / "cutout.json", "w"))


def uncor_size(cat):
    '''pyRRG's size for the size cut (calc_shear): sqrt of the mean Gaussian-weighted second moment of the
    PSF-convolved image, before PSF correction (weight radius sqrt(A_IMAGE B_IMAGE) x mult), in 0.02" pixels.'''
    return np.sqrt(0.5 * (cat["xx_uncorrected"] + cat["yy_uncorrected"]))


def zero_point(f):
    '''SExtractor MAG_ZEROPOINT pyRRG uses (RRGtools.source_extract, zero_point='jwst').'''
    h = fits.getheader(WORK / "base" / f / (ROOT.format(f) + ".fits"))
    return -2.5 * np.log10(h["PHOTMJSR"] * h["PIXAR_SR"] * 1e6) + 8.9


def full_pixels(ra, dec):
    '''Full-mosaic pixel coordinates (as the live catalogue's x, y) of RA, Dec.'''
    w = WCS(fits.getheader(LIVE / "f150w" / (ROOT.format("f150w") + ".fits")))
    x, y = w.world_to_pixel_values(np.asarray(ra), np.asarray(dec))
    return x + 1., y + 1.                      # SExtractor / pyRRG 1-based


def radius_kpc(ra, dec):
    x, y = full_pixels(ra, dec)
    cx, cy = asd().A2744_STAMP_CENTRE
    return np.hypot(x - cx, y - cy) * KPC_PER_PIX, (x - cx) * KPC_PER_PIX, (y - cy) * KPC_PER_PIX


def live_selection(data_dir, shears):
    '''get_obs_data('concat') with SHEAR_CUTS and photo-z member removal on {filter: .shears path} in a scratch
    data_dir (symlinks; run_match writes to the cwd). Strong-lensing galaxies kept (cut applied by the caller).'''
    d = Path(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    for f, src in shears.items():
        link = d / (ROOT.format(f) + "_clean.shears")
        if not link.exists():
            link.symlink_to(src)
    unc = d / "UNCOVER_DR4_SPS_catalog.fits"
    if not unc.exists():
        unc.symlink_to(REPO / "data" / "100" / "a2744" / "UNCOVER_DR4_SPS_catalog.fits")
    assert d.resolve() != (REPO / "data" / "100" / "a2744").resolve()
    cwd = os.getcwd()
    os.chdir(d)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return asd().get_obs_data("concat", data_dir=str(d), cuts=asd().SHEAR_CUTS, photoz=True,
                                      drop_strong_lensing=False)
    finally:
        os.chdir(cwd)


def sky_match(ra1, dec1, ra2, dec2, rad=0.1):
    '''Index into catalogue 2 of the nearest object within rad arcsec of each catalogue-1 object (-1 if none).'''
    from astropy.coordinates import SkyCoord
    if len(ra2) == 0:
        return np.full(len(ra1), -1)
    c1 = SkyCoord(np.asarray(ra1) * units.deg, np.asarray(dec1) * units.deg)
    c2 = SkyCoord(np.asarray(ra2) * units.deg, np.asarray(dec2) * units.deg)
    idx, sep, _ = c1.match_to_catalog_sky(c2)
    return np.where(sep.arcsec < rad, idx, -1)


def population():
    '''Observed (mag, uncorrected size) in both filters of the live catalogue's galaxies beyond 400 kpc
    (few members, low magnification), for the injection population.'''
    live = live_selection(WORK / "live_cat", {f: REPO / "data" / "100" / "a2744" / (ROOT.format(f) + "_clean.shears")
                                              for f in FILTERS})
    r, _, _ = radius_kpc(live["RA"], live["DEC"])
    stamp = (np.abs(full_pixels(live["RA"], live["DEC"])[0] - asd().A2744_STAMP_CENTRE[0]) * KPC_PER_PIX < 1000) & \
            (np.abs(full_pixels(live["RA"], live["DEC"])[1] - asd().A2744_STAMP_CENTRE[1]) * KPC_PER_PIX < 1000)
    kap = asd().strong_lensing_kappa(live)
    print(f"live catalogue: {len(live)}, in stamp {stamp.sum()}, in stamp with kappa<{asd().KAPPA_THRESH}: "
          f"{(stamp & (kap < asd().KAPPA_THRESH)).sum()}")
    fits.writeto(WORK / "live_concat.fits", live, overwrite=True)
    sel = live[stamp & (r > 400) & (kap < asd().KAPPA_THRESH)]
    cols = {}
    for f in FILTERS:
        raw = fits.getdata(REPO / "data" / "100" / "a2744" / (ROOT.format(f) + "_clean.shears"))
        i = sky_match(sel["RA"], sel["DEC"], raw["RA"], raw["DEC"], 0.1)
        ok = i >= 0
        cols[f"mag_{f}"] = np.where(ok, raw["MAG_AUTO"][i], np.nan)
        cols[f"size_{f}"] = np.where(ok, uncor_size(raw)[i], np.nan)
    both = np.isfinite(cols["mag_f115w"]) & np.isfinite(cols["mag_f150w"])
    colour = np.median(cols["mag_f115w"][both] - cols["mag_f150w"][both])
    ratio = np.median(cols["size_f115w"][both] / cols["size_f150w"][both])
    for f, g, sgn in (("f115w", "f150w", 1), ("f150w", "f115w", -1)):
        miss = ~np.isfinite(cols[f"mag_{f}"])
        cols[f"mag_{f}"][miss] = cols[f"mag_{g}"][miss] + sgn * colour
        cols[f"size_{f}"][miss] = cols[f"size_{g}"][miss] * ratio ** sgn
    print(f"population: {len(sel)} galaxies (r > 400 kpc), {both.sum()} in both filters; "
          f"median F115W-F150W {colour:.3f}, size ratio {ratio:.3f}")
    t = fits.BinTableHDU.from_columns([fits.Column(name=k, format="D", array=v) for k, v in cols.items()])
    t.writeto(WORK / "population.fits", overwrite=True)


# --------------------------------------------------------------------------------------------- rendering
_PSF = {}


def psf(f):
    import galsim
    if f not in _PSF:
        _PSF[f] = galsim.InterpolatedImage(galsim.Image(fits.getdata(WORK / f"psf_{f}.fits"), scale=PIX))
    return _PSF[f]


def draw_params(n, rng):
    '''Intrinsic morphology: disk half-light radius (arcsec), bulge fraction, shapes, orientation. Bulge-to-total
    from a two-population mix (75% disk-dominated, B/T ~ U(0, 0.3); 25% bulge-dominated, B/T ~ U(0.5, 1)); bulge
    hlr = 0.5 disk hlr; disk |g| ~ Rayleigh(0.25) (< 0.8), bulge |g| ~ Rayleigh(0.1) (< 0.5); common position angle.'''
    bulgy = rng.random(n) < 0.25
    bt = np.where(bulgy, rng.uniform(0.5, 1., n), rng.uniform(0., 0.3, n))
    gd = np.minimum(rng.rayleigh(0.25, n), 0.8)
    gb = np.minimum(rng.rayleigh(0.1, n), 0.5)
    pa = rng.uniform(0, np.pi, n)
    return bt, gd, gb, pa


def render(flux, hlr, bt, gd, gb, pa, f, dx, dy):
    import galsim
    beta = pa * galsim.radians
    disk = galsim.Exponential(half_light_radius=hlr).shear(g=gd, beta=beta)
    bulge = galsim.DeVaucouleurs(half_light_radius=0.5 * hlr).shear(g=gb, beta=beta)
    gal = (1. - bt) * disk + bt * bulge
    obj = galsim.Convolve(gal.withFlux(flux), psf(f))
    n = int(np.clip(2 * np.ceil(8 * hlr / PIX) + 1, 2 * STAMP_SIZE + 1, 401))
    return obj.drawImage(nx=n, ny=n, scale=PIX, offset=(dx, dy)).array


def inject_image(sci, wht, x, y, mags, zp, hlr, bt, gd, gb, pa, f):
    '''Add galaxies at 0-based pixel positions (x, y) of the arrays (in place; nothing where wht == 0).'''
    flux = 10 ** (-0.4 * (mags - zp))
    for i in range(len(x)):
        ix, iy = int(np.round(x[i])), int(np.round(y[i]))
        stamp = render(flux[i], hlr[i], bt[i], gd[i], gb[i], pa[i], f, x[i] - ix, y[i] - iy)
        h = stamp.shape[0] // 2
        ya, yb, xa, xb = iy - h, iy + h + 1, ix - h, ix + h + 1
        sa, sb, ta, tb = max(0, -ya), stamp.shape[0] - max(0, yb - sci.shape[0]), max(0, -xa), stamp.shape[1] - max(0, xb - sci.shape[1])
        ya, yb, xa, xb = max(ya, 0), min(yb, sci.shape[0]), max(xa, 0), min(xb, sci.shape[1])
        sci[ya:yb, xa:xb] += stamp[sa:sb, ta:tb] * (wht[ya:yb, xa:xb] > 0)


def positions(n, wht, xlim, ylim, rng, min_sep=MIN_SEP_PIX, keep=None):
    '''n uniform positions (0-based, array pixels) within xlim, ylim on covered pixels (wht > 0 at the
    centre and +-2 pixels), at least min_sep apart (random sequential addition; fails rather than jams).'''
    xs, ys = np.empty(0), np.empty(0)
    for _ in range(200):
        m = 4 * (n - len(xs)) + 100
        x, y = rng.uniform(*xlim, m), rng.uniform(*ylim, m)
        ix, iy = x.astype(int), y.astype(int)
        ok = (ix > 2) & (iy > 2) & (ix < wht.shape[1] - 3) & (iy < wht.shape[0] - 3)
        x, y, ix, iy = x[ok], y[ok], ix[ok], iy[ok]
        ok = np.ones(len(x), bool)
        for ddx in (-2, 0, 2):
            for ddy in (-2, 0, 2):
                ok &= wht[iy + ddy, ix + ddx] > 0
        if keep is not None:
            ok &= keep(x, y)
        for xi, yi in zip(x[ok], y[ok]):
            if len(xs) >= n:
                return xs, ys
            if len(xs) and np.min((xs - xi) ** 2 + (ys - yi) ** 2) < min_sep ** 2:
                continue
            xs, ys = np.append(xs, xi), np.append(ys, yi)
        if len(xs) >= n:
            return xs, ys
    raise RuntimeError(f"only {len(xs)} of {n} positions fit at {min_sep} px separation")


def write_injected(d, f, sci, hdr):
    d.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(sci, hdr).writeto(d / (ROOT.format(f) + ".fits"), overwrite=True)


# ------------------------------------------------------------------------------------- size calibration
PILOT_BOX = (16000, 22000, 12500, 18500)     # x0, x1, y0, y1 in cut-out pixels (~700 kpc from the centre)


def pilot(n="1200", seed="1", name="pilot", mag_lo="22", mag_hi="29.5", hlr_lo="0.01", hlr_hi="0.8"):
    '''Broad hlr/magnitude grid in an outer 6000^2 sub-cut-out, to map intrinsic (hlr, mag) to pyRRG's
    (uncorrected size, MAG_AUTO).'''
    rng = np.random.default_rng(int(seed))
    n = int(n)
    x0, x1, y0, y1 = PILOT_BOX
    out = WORK / name
    wht150 = fits.getdata(WORK / "base" / "f150w" / f"abell2744clu-grizli-v5.4-f150w-clear_drc_wht.fits", memmap=True)[y0:y1, x0:x1]
    x, y = positions(n, np.array(wht150), (60, x1 - x0 - 60), (60, y1 - y0 - 60), rng, min_sep=130)
    hlr = 10 ** rng.uniform(np.log10(float(hlr_lo)), np.log10(float(hlr_hi)), n)
    mag150 = rng.uniform(float(mag_lo), float(mag_hi), n)
    colour = rng.normal(0.1, 0.3, n)
    bt, gd, gb, pa = draw_params(n, rng)
    for f in FILTERS:
        base = WORK / "base" / f
        with fits.open(base / (ROOT.format(f) + ".fits"), memmap=True) as h:
            hdr = h[0].header.copy()
            sci = np.array(h[0].data[y0:y1, x0:x1], dtype=np.float32)
        wht = np.array(fits.getdata(base / f"abell2744clu-grizli-v5.4-{f}-clear_drc_wht.fits", memmap=True)[y0:y1, x0:x1])
        hdr["CRPIX1"] -= x0
        hdr["CRPIX2"] -= y0
        mags = mag150 + (colour if f == "f115w" else 0.)
        inject_image(sci, wht, x, y, mags, zero_point(f), hlr, bt, gd, gb, pa, f)
        write_injected(out / f, f, sci, hdr)
        fits.PrimaryHDU(wht, hdr).writeto(out / f / f"abell2744clu-grizli-v5.4-{f}-clear_drc_wht.fits", overwrite=True)
    w = WCS(hdr)
    ra, dec = w.pixel_to_world_values(x, y)
    cols = dict(x=x, y=y, ra=ra, dec=dec, hlr=hlr, mag_f150w=mag150, mag_f115w=mag150 + colour, bt=bt, gd=gd, gb=gb, pa=pa)
    fits.BinTableHDU.from_columns([fits.Column(name=k, format="D", array=v) for k, v in cols.items()]).writeto(
        out / "injected.fits", overwrite=True)
    print(f"pilot: {n} sources in {out}")


# ------------------------------------------------------------------------------------ injection runs
PILOTS = ("pilot", "pilot2", "pilot3")
N_PER_REAL = 4000
K_NN = 8


def calibration():
    '''Pilot galaxies recovered as F150W galaxies: measured (MAG_AUTO, log uncorrected size) and their intrinsic
    (hlr, input F150W mag, B/T, disk and bulge |g|), for the nearest-neighbour mapping of the observed
    population to intrinsic galaxies.'''
    rows = []
    for name in PILOTS:
        inj = fits.getdata(WORK / name / "injected.fits")
        c = fits.getdata(WORK / name / "f150w" / (ROOT.format("f150w") + "_clean.shears"))
        i = sky_match(inj["ra"], inj["dec"], c["RA"], c["DEC"], 0.12)
        ok = i >= 0
        rows.append(np.c_[c["MAG_AUTO"][i[ok]], np.log10(uncor_size(c)[i[ok]]), inj["hlr"][ok], inj["mag_f150w"][ok],
                          inj["bt"][ok], inj["gd"][ok], inj["gb"][ok]])
    cal = np.vstack(rows)
    return cal[np.all(np.isfinite(cal), axis=1)]


def intrinsic(pop, rng):
    '''For each population galaxy, the morphology of one of its K_NN nearest pilot galaxies in measured
    (F150W MAG_AUTO, log uncorrected size); its input magnitude is offset by that pilot galaxy's
    (input - measured), F115W follows from the population galaxy's measured colour.'''
    from scipy.spatial import cKDTree
    cal = calibration()
    scale = np.array([0.25, 0.05])                     # mag, dex: comparable neighbour distances
    _, nn = cKDTree(cal[:, :2] / scale).query(np.c_[pop["mag_f150w"], np.log10(pop["size_f150w"])] / scale, k=K_NN)
    pick = nn[np.arange(len(pop)), rng.integers(0, K_NN, len(pop))]
    mag150 = pop["mag_f150w"] + (cal[pick, 3] - cal[pick, 0])
    return dict(hlr=cal[pick, 2], mag_f150w=mag150, mag_f115w=mag150 + (pop["mag_f115w"] - pop["mag_f150w"]),
                bt=cal[pick, 4], gd=cal[pick, 5], gb=cal[pick, 6], pa=rng.uniform(0, np.pi, len(pop)),
                tgt_mag_f150w=np.asarray(pop["mag_f150w"], float), tgt_size_f150w=np.asarray(pop["size_f150w"], float))


def inject(real, n=None, pop_file="population.fits", mu="1", rmin="0", name=None, seed=None):
    '''Realisation `real`: n (N_PER_REAL) galaxies drawn from the population, uniform over the covered (either
    filter) part of the 2 Mpc stamp beyond rmin kpc, at least MIN_SEP_PIX apart, injected into copies of both
    cut-outs, magnified by mu (flux x mu, half-light radius x sqrt(mu)). Output in WORK/<name> (real_<real>).
    The random draws depend only on seed (1000 + real), so runs differing only in mu are paired.'''
    real, mu, rmin = int(real), float(mu), float(rmin)
    n = N_PER_REAL if n is None else int(n)
    rng = np.random.default_rng(1000 + real if seed is None else int(seed))
    cut = json.load(open(WORK / "cutout.json"))
    pop = fits.getdata(WORK / pop_file)
    pop = pop[np.isfinite(pop["mag_f150w"]) & np.isfinite(pop["size_f150w"]) & np.isfinite(pop["mag_f115w"])]
    pop = pop[rng.integers(0, len(pop), n)]
    par = intrinsic(pop, rng)
    whts = {f: np.array(fits.getdata(WORK / "base" / f / f"abell2744clu-grizli-v5.4-{f}-clear_drc_wht.fits", memmap=True))
            for f in FILTERS}
    cover = ((whts["f115w"] > 0) | (whts["f150w"] > 0)).astype(np.uint8)
    cx, cy = np.array(asd().A2744_STAMP_CENTRE) - 1 - np.array([cut["x0"], cut["y0"]])
    half = STAMP_HALF_KPC / KPC_PER_PIX
    keep = (lambda x, y: np.hypot(x - cx, y - cy) * KPC_PER_PIX > rmin) if rmin > 0 else None
    x, y = positions(n, cover, (cx - half, cx + half), (cy - half, cy + half), rng, keep=keep)
    del cover
    par["mu"] = np.full(n, mu)
    par["hlr"] = par["hlr"] * np.sqrt(mu)
    for f in FILTERS:
        par[f"mag_{f}"] = par[f"mag_{f}"] - 2.5 * np.log10(mu)
    out = WORK / (f"real_{real}" if name is None else name)
    for f in FILTERS:
        with fits.open(WORK / "base" / f / (ROOT.format(f) + ".fits"), memmap=True) as h:
            hdr = h[0].header.copy()
            sci = np.array(h[0].data, dtype=np.float32)
        inject_image(sci, whts[f], x, y, par[f"mag_{f}"], zero_point(f), par["hlr"], par["bt"], par["gd"],
                     par["gb"], par["pa"], f)
        write_injected(out / f, f, sci, hdr)
        del sci
    ra, dec = WCS(hdr).pixel_to_world_values(x, y)
    cols = dict(x=x, y=y, ra=ra, dec=dec, **par)
    fits.BinTableHDU.from_columns([fits.Column(name=k, format="D", array=v) for k, v in cols.items()]).writeto(
        out / "injected.fits", overwrite=True)
    print(f"{out.name}: {n} sources (mu = {mu}, r > {rmin} kpc)")


MAG_MUS = (1.0, 1.4, 2.0, 2.8, 4.0)


def inject_mag(seed, mu):
    '''Magnification-response run: the broad field population beyond 500 kpc, magnified by mu; the same
    galaxies and positions for every mu of a seed.'''
    inject(0, pop_file="population_broad.fits", mu=mu, rmin="500", name=f"mag_s{seed}_mu{float(mu):.1f}",
           seed=5000 + int(seed))


def population_broad():
    '''Field galaxies (stamp, r > 400 kpc, kappa_sl < KAPPA_THRESH) detected and measured by pyRRG in F150W with
    MAG_AUTO < 28.5 and uncorrected size > 1.5 pixel, i.e. including those below the shear cuts that
    magnification can lift into them; photo-z members (UNCOVER z < 0.455 within 0.5") removed. F115W from the
    match (or the median colour / size ratio).'''
    raw = {f: fits.getdata(REPO / "data" / "100" / "a2744" / (ROOT.format(f) + "_clean.shears")) for f in FILTERS}
    c = raw["f150w"]
    r, dx, dy = radius_kpc(c["RA"], c["DEC"])
    ok = (r > 400) & (np.abs(dx) < STAMP_HALF_KPC) & (np.abs(dy) < STAMP_HALF_KPC) & \
        (asd().strong_lensing_kappa(c) < asd().KAPPA_THRESH) & (c["galStarFlag"] == 1) & (c["nExposures"] >= 2) & \
        (c["prob"] == 0) & (c["MAG_AUTO"] < 28.5) & (uncor_size(c) > 1.5)
    unc = fits.getdata(REPO / "data" / "100" / "a2744" / "UNCOVER_DR4_SPS_catalog.fits")
    good = np.isfinite(unc["ra"]) & np.isfinite(unc["dec"])
    k = sky_match(c["RA"], c["DEC"], unc["ra"][good], unc["dec"][good], 0.5)
    member = (k >= 0) & (unc["z_ml"][good][np.maximum(k, 0)] < 0.305 + 0.05 * 3)
    c = c[ok & ~member]
    i = sky_match(c["RA"], c["DEC"], raw["f115w"]["RA"], raw["f115w"]["DEC"], 0.1)
    m = i >= 0
    colour = np.median(raw["f115w"]["MAG_AUTO"][i[m]] - c["MAG_AUTO"][m])
    cols = dict(mag_f150w=c["MAG_AUTO"], size_f150w=uncor_size(c),
                mag_f115w=np.where(m, raw["f115w"]["MAG_AUTO"][np.maximum(i, 0)], c["MAG_AUTO"] + colour))
    fits.BinTableHDU.from_columns([fits.Column(name=k, format="D", array=np.asarray(v, float)) for k, v in cols.items()]
                                  ).writeto(WORK / "population_broad.fits", overwrite=True)
    print(f"broad population: {len(c)} ({(ok & member).sum()} members removed); "
          f"{(cols['size_f150w'] > 2.75).mean():.2f} above the size cut")


def baseline():
    '''Live selection on the un-injected cut-out run (real_0).'''
    f0 = WORK / "real_0" / "concat.fits"
    if not f0.exists():
        fits.writeto(f0, live_selection(WORK / "real_0" / "cat", {
            f: WORK / "real_0" / f / (ROOT.format(f) + "_clean.shears") for f in FILTERS}), overwrite=True)
    return fits.getdata(f0)


def recover(real):
    '''Live selection on realisation `real` and the match of the injections to it. An injected galaxy is
    recovered if a selected object lies within 0.12" of it and that object is not better explained (closer
    in magnitude) by a selected baseline (un-injected) object at the same place.'''
    d = WORK / (f"real_{int(real)}" if str(real).isdigit() else str(real))
    cat = live_selection(d / "cat", {f: d / f / (ROOT.format(f) + "_clean.shears") for f in FILTERS})
    fits.writeto(d / "concat.fits", cat, overwrite=True)
    base = baseline()
    inj = fits.getdata(d / "injected.fits")
    i = sky_match(inj["ra"], inj["dec"], cat["RA"], cat["DEC"], 0.12)
    ok = i >= 0
    j = sky_match(cat["RA"][i[ok]], cat["DEC"][i[ok]], base["RA"], base["DEC"], 0.12)
    m_rec = cat["MAG"][i[ok]]                          # F115W MAG_AUTO if detected there, else F150W
    m_inj = inj["mag_f115w"][ok]
    real_better = (j >= 0) & (np.abs(base["MAG"][np.maximum(j, 0)] - m_rec) < np.abs(m_inj - m_rec))
    det = np.zeros(len(inj), bool)
    det[np.where(ok)[0][~real_better]] = True
    per_filter = {}
    for f in FILTERS:
        c = fits.getdata(d / f / (ROOT.format(f) + "_clean.shears"))
        per_filter[f"pipe_{f}"] = sky_match(inj["ra"], inj["dec"], c["RA"], c["DEC"], 0.12) >= 0
    kap = asd().strong_lensing_kappa({"RA": inj["ra"], "DEC": inj["dec"]})
    r, dx, dy = radius_kpc(inj["ra"], inj["dec"])
    cols = dict(ra=inj["ra"], dec=inj["dec"], r_kpc=r, dx_kpc=dx, dy_kpc=dy, kappa_sl=kap, det=det,
                blend_real=ok & ~det, **per_filter)
    fits.BinTableHDU.from_columns([fits.Column(name=k, format="L" if v.dtype == bool else "D", array=v)
                                   for k, v in cols.items()]).writeto(d / "recovered.fits", overwrite=True)
    sel = kap < asd().KAPPA_THRESH
    print(f"{d.name}: {det.sum()} of {len(inj)} recovered ({det[sel].mean():.3f} at kappa < "
          f"{asd().KAPPA_THRESH}); {(ok & ~det).sum()} matched to real objects instead")


# ------------------------------------------------------------------------------------------ analysis
R_S = 350.                                     # kpc (Schrabback et al. 2025)
R_EDGES = np.array([0, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000, 1150, 1414.])
K_EDGES = np.array([0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7])  # strong_lensing_kappa (0 = off map)
KL_EDGES = np.array([-1, 0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 2.])  # Lenstool kappa
PROFILE_EDGES = np.geomspace(10, 2500, 41)     # as sbi/boost_kappa.py
DPL = WORK / "lenstool"                        # Bergamini+23 deflection maps at zs = 2.28 (940" field, 2400 pix)
ZS_MAP = 2.28


class Lenstool:
    '''kappa, gamma of the Lenstool model at zs = 2.28 from the deflection maps (as sbi/pyrrg_sweep.LenstoolShear).'''

    def __init__(self):
        hx, hy = fits.open(DPL / "dplx_zs2.28.fits")[0], fits.open(DPL / "dply_zs2.28.fits")[0]
        ax, ay = hx.data.astype(float), hy.data.astype(float)
        step = 940. / (ax.shape[0] - 1)
        dxax, dyax = np.gradient(ax, step, axis=1), np.gradient(ax, step, axis=0)
        dxay, dyay = np.gradient(ay, step, axis=1), np.gradient(ay, step, axis=0)
        self.k, self.g1, self.g2 = 0.5 * (dxax + dyay), 0.5 * (dxax - dyay), 0.5 * (dyax + dxay)
        self.wcs = WCS(hx.header)

    def at(self, ra, dec):
        from scipy.ndimage import map_coordinates
        pi, pj = self.wcs.world_to_pixel_values(np.asarray(ra), np.asarray(dec))
        s = lambda m: map_coordinates(m, [pj, pi], order=1, mode="constant", cval=0.)
        return s(self.k), np.hypot(s(self.g1), s(self.g2))

    def kappa_profile(self):
        '''Azimuthal mean kappa vs radius about the stamp centre, made non-increasing (boost_kappa's
        a2744_kappa_profile, with the radius from the catalogue WCS instead of the stamp fit).'''
        jj, ii = np.indices(self.k.shape)
        ra, dec = self.wcs.pixel_to_world_values(ii.ravel(), jj.ravel())
        r, _, _ = radius_kpc(ra, dec)
        k = self.k.ravel()
        rc, kb = [], []
        for lo, hi in zip(PROFILE_EDGES[:-1], PROFILE_EDGES[1:]):
            s = (r >= lo) & (r < hi) & np.isfinite(k)
            if s.sum() > 3:
                rc.append(np.sqrt(lo * hi))
                kb.append(k[s].mean())
        return np.array(rc), np.minimum.accumulate(np.array(kb))


def beta(z):
    z = np.asarray(z, float)
    out = np.zeros_like(z)
    ok = z > ZL + 0.01
    out[ok] = (Planck18.angular_diameter_distance_z1z2(ZL, z[ok]) / Planck18.angular_diameter_distance(z[ok])).value
    return out


def count_response(dm_grid):
    '''N_sel(fluxes x 10^(0.4 dm)) / N_sel for the live selection of field galaxies (stamp, r > 400 kpc): MAG_AUTO
    brighter by dm and S/N up by 10^(0.4 dm) (sky-limited); sizes held fixed ("flux") or also magnified, the
    uncorrected size scaling as 10^(0.2 dm) ("flux+size", an upper limit since the PSF part does not grow).
    A galaxy is selected if it passes SHEAR_CUTS in either filter (the concat catalogue is their union).'''
    raws = {f: fits.getdata(REPO / "data" / "100" / "a2744" / (ROOT.format(f) + "_clean.shears")) for f in FILTERS}
    # union of the two filters, in the f150w-detected list plus f115w-only objects
    i115 = sky_match(raws["f150w"]["RA"], raws["f150w"]["DEC"], raws["f115w"]["RA"], raws["f115w"]["DEC"], 0.1)
    only115 = ~np.isin(np.arange(len(raws["f115w"])), i115[i115 >= 0])
    ra = np.r_[raws["f115w"]["RA"][only115], raws["f150w"]["RA"]]
    dec = np.r_[raws["f115w"]["DEC"][only115], raws["f150w"]["DEC"]]
    idx = {"f115w": np.r_[np.where(only115)[0], i115], "f150w": np.r_[np.full(only115.sum(), -1), np.arange(len(raws["f150w"]))]}
    r, dx, dy = radius_kpc(ra, dec)
    field = (r > 400) & (np.abs(dx) < 1000) & (np.abs(dy) < 1000)
    out = {}
    for mode in ("flux", "flux+size"):
        res = []
        for dm in dm_grid:
            sel = np.zeros(len(ra), bool)
            for f in FILTERS:
                c, cut = raws[f], asd().SHEAR_CUTS[f]
                j = idx[f]
                have = j >= 0
                jj = np.maximum(j, 0)
                snr = c["FLUX_AUTO"][jj] / c["FLUXERR_AUTO"][jj] * 10 ** (0.4 * dm)
                size = uncor_size(c)[jj] * (10 ** (0.2 * dm) if mode == "flux+size" else 1.)
                mag = c["MAG_AUTO"][jj] - dm
                sel |= have & (snr >= cut["signal_noise_cut"]) & (size >= cut["size_cut"][0]) & \
                    (size <= cut["size_cut"][1]) & (mag >= cut["mag_cut"][0]) & (mag <= cut["mag_cut"][1]) & \
                    (c["nExposures"][jj] >= 2) & (c["prob"][jj] == 0)
            res.append((sel & field).sum())
        out[mode] = np.array(res) / res[0]
    return out


def depletion(kap, gam, zs, w_z, dm_grid, resp):
    '''Mean over the source-redshift distribution of mu^-1 N_sel(mu)/N_sel at each (kappa, gamma) at zs = 2.28.'''
    b = beta(zs) / beta(np.array([ZS_MAP]))[0]
    k = kap[:, None] * b[None, :]
    g = gam[:, None] * b[None, :]
    mu = 1. / np.abs((1 - k) ** 2 - g ** 2)
    mu = np.clip(mu, 1., 10 ** (0.4 * dm_grid[-1]))
    ratio = np.interp(2.5 * np.log10(mu), dm_grid, resp)
    return np.sum(w_z[None, :] * ratio / mu, axis=1) / w_z.sum()


def magnification_response():
    '''R(mu) = N_sel(mu) / N_sel(1) from the paired magnified-injection runs (mag_s<seed>_mu<mu>): per seed, the
    selected-and-recovered count at each mu over that at mu = 1; mean and spread over seeds.'''
    out = {}
    for d in sorted(WORK.glob("mag_s*_mu*/recovered.fits")):
        seed, mu = d.parent.name.split("_")[1], float(d.parent.name.split("_mu")[1])
        out.setdefault(seed, {})[mu] = int(np.sum(fits.getdata(d)["det"]))
    rows = {}
    for seed, v in out.items():
        if 1.0 in v:
            for mu, n in v.items():
                rows.setdefault(mu, []).append(n / v[1.0])
    if len(rows) < 2:
        return None
    mus = np.array(sorted(rows))
    return dict(mu=mus, R=np.array([np.mean(rows[m]) for m in mus]),
                R_err=np.array([np.std(rows[m]) / np.sqrt(max(len(rows[m]) - 1, 1)) for m in mus]),
                n_seeds=np.array([len(rows[m]) for m in mus]), counts=out)


def load_recovered():
    rows = [fits.getdata(p) for p in sorted(WORK.glob("real_*/recovered.fits")) if "real_0/" not in str(p)]
    keys = rows[0].columns.names
    return {k: np.concatenate([np.asarray(r[k]) for r in rows]) for k in keys}, len(rows)


def binned_fraction(x, det, edges):
    n = np.histogram(x, edges)[0]
    k = np.histogram(x[det], edges)[0]
    p = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    # Wilson-like error, floored for p near 0/1
    e = np.sqrt(np.maximum(p * (1 - p), 1. / np.maximum(n, 1)) / np.maximum(n, 1))
    return n, k, p, e


def fit_boost(xc, y, ye, model, p0, bounds):
    from scipy.optimize import curve_fit
    ok = np.isfinite(y) & np.isfinite(ye) & (ye > 0)
    popt, pcov = curve_fit(model, xc[ok] if xc.ndim == 1 else xc[:, ok], y[ok], sigma=ye[ok], absolute_sigma=True,
                           p0=p0, bounds=bounds, maxfev=20000)
    chi2 = np.sum(((y[ok] - model(xc[ok] if xc.ndim == 1 else xc[:, ok], *popt)) / ye[ok]) ** 2)
    return popt, np.sqrt(np.diag(pcov)), chi2, ok.sum() - len(popt)


def analyse():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    rec, nreal = load_recovered()
    kthr = asd().KAPPA_THRESH
    L = Lenstool()
    res = {"n_realisations": nreal, "n_injected": int(len(rec["det"]))}

    # -- injections: obscuration / selection probability
    inside = (np.abs(rec["dx_kpc"]) < STAMP_HALF_KPC) & (np.abs(rec["dy_kpc"]) < STAMP_HALF_KPC)
    lowk = inside & (rec["kappa_sl"] < kthr)
    det = rec["det"].astype(bool)
    kL_inj, gL_inj = L.at(rec["ra"], rec["dec"])

    # -- the real catalogue (live selection, members removed), same footprint
    live = fits.getdata(WORK / "live_concat.fits")
    r_obs, dx_o, dy_o = radius_kpc(live["RA"], live["DEC"])
    kap_obs = asd().strong_lensing_kappa(live)
    obs = (np.abs(dx_o) < STAMP_HALF_KPC) & (np.abs(dy_o) < STAMP_HALF_KPC) & (kap_obs < kthr)
    kL_obs, _ = L.at(live["RA"], live["DEC"])
    base = baseline()
    r_b, dx_b, dy_b = radius_kpc(base["RA"], base["DEC"])
    obs_b = (np.abs(dx_b) < STAMP_HALF_KPC) & (np.abs(dy_b) < STAMP_HALF_KPC) & (asd().strong_lensing_kappa(base) < kthr)

    # -- magnification depletion
    dm_grid = np.linspace(0, 2.5, 51)
    resp = count_response(dm_grid)
    mr = magnification_response()
    if mr is not None:              # measured: interpolated in log mu, held at the largest mu run beyond it
        resp["injection"] = np.interp(dm_grid, 2.5 * np.log10(mr["mu"]), mr["R"])
        res["magnification_response"] = {k: (np.asarray(v).tolist() if k != "counts" else v) for k, v in mr.items()}
    s_eff = {m: float(np.log10(np.interp(0.2, dm_grid, v)) / 0.2) for m, v in resp.items()}
    modes = list(resp)
    field_z = np.asarray(live["redshift"])[obs & (r_obs > 400)]
    zs_grid = np.linspace(0.3, 7, 135)
    w_z = np.histogram(field_z, np.r_[zs_grid - 0.025, zs_grid[-1] + 0.025])[0].astype(float)
    dep = {m: depletion(kL_inj, gL_inj, zs_grid, w_z, dm_grid, v) for m, v in resp.items()}
    res["count_slope_s"] = s_eff
    res["source_z_median"] = float(np.median(field_z))

    def profile(xi, xo, xb, edges, sel_i, sel_o, sel_b, mode="flux"):
        n_inj, n_det, p, pe = binned_fraction(xi[sel_i], det[sel_i], edges)
        n_obs = np.histogram(xo[sel_o], edges)[0].astype(float)
        n_base = np.histogram(xb[sel_b], edges)[0].astype(float)
        D = np.array([dep[mode][sel_i & (xi >= lo) & (xi < hi)].mean() if ((xi >= lo) & (xi < hi) & sel_i).any()
                      else np.nan for lo, hi in zip(edges[:-1], edges[1:])])
        xc = np.array([xi[sel_i & (xi >= lo) & (xi < hi)].mean() if ((xi >= lo) & (xi < hi) & sel_i).any()
                       else np.nan for lo, hi in zip(edges[:-1], edges[1:])])
        # density per unit covered area (injections are uniform over the covered area): n_obs / n_inj
        dens = n_obs / np.maximum(n_inj, 1)
        corr = dens / p / D
        corr_e = corr * np.sqrt(1. / np.maximum(n_obs, 1) + (pe / p) ** 2)
        return dict(xc=xc, n_inj=n_inj, n_det=n_det, p=p, pe=pe, n_obs=n_obs, n_base=n_base, dens=dens, D=D,
                    corr=corr, corr_e=corr_e)

    # radial profile (stamp, kappa_sl < KAPPA_THRESH)
    rp = {m: profile(rec["r_kpc"], r_obs, r_b, R_EDGES, lowk, obs, obs_b, m) for m in modes}
    p_r_all = binned_fraction(rec["r_kpc"][inside], det[inside], R_EDGES)        # incl. the kappa cut
    # strong_lensing_kappa bins
    kp = profile(rec["kappa_sl"], kap_obs, asd().strong_lensing_kappa(base), K_EDGES, lowk, obs, obs_b)
    # Lenstool-kappa bins (the kappa of boost_kappa.py)
    kLp = profile(kL_inj, kL_obs, L.at(base["RA"], base["DEC"])[0], KL_EDGES, lowk, obs, obs_b)

    # -- fits
    def B_r(r, n0, A, rs=R_S):
        return n0 * (1 + A * np.exp(1 - r / rs))

    rc_prof, kb_prof = L.kappa_profile()
    order = np.argsort(kb_prof)

    def r_of_kappa(k):
        return np.interp(k, kb_prof[order], rc_prof[order])     # clamps outside the tabulated range

    # kappa model: mean over the injection positions of each bin of 1 + A exp(1 - r_A2744(kappa)/r_s)
    kbin = np.digitize(kL_inj, KL_EDGES) - 1

    def B_k_factory(rs_fixed=True):
        def B_k(idx, n0, A, rs=R_S):
            out = []
            for b in idx.astype(int):
                kk = kL_inj[lowk & (kbin == b)]
                Bk = np.where(kk > 0, 1 + A * np.exp(1 - r_of_kappa(kk) / rs), 1.)
                out.append(Bk.mean())
            return n0 * np.array(out)
        return B_k

    fits_out = {}
    for mode in modes:
        d = rp[mode]
        n0g = np.nanmedian(d["corr"][-4:])
        f1 = fit_boost(d["xc"], d["corr"], d["corr_e"], lambda r, n0, A: B_r(r, n0, A), [n0g, 0.1], ([0, -1], [10, 10]))
        f2 = fit_boost(d["xc"], d["corr"], d["corr_e"], B_r, [n0g, 0.1, 350], ([0, -1, 30], [10, 50, 3000]))
        Bk = B_k_factory()
        idx = np.arange(len(KL_EDGES) - 1).astype(float)
        dk = profile(kL_inj, kL_obs, L.at(base["RA"], base["DEC"])[0], KL_EDGES, lowk, obs, obs_b, mode)
        f3 = fit_boost(idx, dk["corr"], dk["corr_e"], lambda i, n0, A: Bk(i, n0, A), [n0g, 0.1], ([0, -1], [10, 10]))
        f4 = fit_boost(idx, dk["corr"], dk["corr_e"], Bk, [n0g, 0.1, 350], ([0, -1, 30], [10, 50, 3000]))
        fo = {}
        for name, (popt, perr, chi2, dof) in (("r_rs350", f1), ("r_rsfree", f2), ("kappa_rs350", f3), ("kappa_rsfree", f4)):
            A, Ae = popt[1], perr[1]
            rs = popt[2] if len(popt) > 2 else R_S
            fo[name] = dict(n0=popt[0], A=A, A_err=Ae, A_err_scaled=Ae * np.sqrt(max(chi2 / max(dof, 1), 1.)), r_s=rs, r_s_err=perr[2] if len(popt) > 2 else 0., chi2=chi2,
                            dof=int(dof), B100=1 + A * np.exp(1 - 100 / rs), B300=1 + A * np.exp(1 - 300 / rs))
        # no magnification correction (as the obscuration-only boost)
        f0 = fit_boost(d["xc"], d["corr"] * d["D"], d["corr_e"] * d["D"], lambda r, n0, A: B_r(r, n0, A), [n0g, 0.1],
                       ([0, -1], [10, 10]))
        fo["r_rs350_no_magnification"] = dict(n0=f0[0][0], A=f0[0][1], A_err=f0[1][1], chi2=f0[2], dof=int(f0[3]))
        # no obscuration correction (raw counts, magnification-corrected)
        fr = fit_boost(d["xc"], d["dens"] / d["D"], d["corr_e"] * d["p"], lambda r, n0, A: B_r(r, n0, A), [n0g, 0.1],
                       ([0, -1], [10, 10]))
        fo["r_rs350_no_obscuration"] = dict(n0=fr[0][0], A=fr[0][1], A_err=fr[1][1], chi2=fr[2], dof=int(fr[3]))
        fits_out[mode] = fo
        rp[mode]["kappa_bins"] = dk
    res["fits"] = fits_out
    res["profiles_r"] = {m: {k: np.asarray(v).tolist() for k, v in rp[m].items() if k != "kappa_bins"} for m in rp}
    res["profiles_kappaL"] = {m: {k: np.asarray(v).tolist() for k, v in rp[m]["kappa_bins"].items()} for m in rp}
    res["profile_kappa_sl"] = {k: np.asarray(v).tolist() for k, v in kp.items()}
    res["p_r_including_kappa_cut"] = {"n": p_r_all[0].tolist(), "k": p_r_all[1].tolist(), "p": p_r_all[2].tolist()}
    res["edges"] = {"r_kpc": R_EDGES.tolist(), "kappa_sl": K_EDGES.tolist(), "kappa_lenstool": KL_EDGES.tolist()}
    res["a2744_kappa_profile"] = {"r_kpc": rc_prof.tolist(), "kappa": kb_prof.tolist()}
    res["fiducial_magnification"] = "injection" if "injection" in modes else "flux+size"
    res["field_p"] = float(det[lowk & (rec["r_kpc"] > 700)].mean())
    res["blend_with_real_fraction"] = float(rec["blend_real"][lowk].mean())
    # 2-D B map (200 kpc pixels): n_obs / n_det(injections) / D, normalised by the fitted field level
    fid = res["fiducial_magnification"]
    e2 = np.linspace(-STAMP_HALF_KPC, STAMP_HALF_KPC, 11)
    h = lambda x, y, w=None: np.histogram2d(x, y, [e2, e2], weights=w)[0]
    n_o = h(dx_o[obs], dy_o[obs])
    n_d = h(rec["dx_kpc"][lowk & det], rec["dy_kpc"][lowk & det])
    n_i = h(rec["dx_kpc"][lowk], rec["dy_kpc"][lowk])
    Dm = np.where(n_i > 30, h(rec["dx_kpc"][lowk], rec["dy_kpc"][lowk], dep[fid][lowk]) / np.maximum(n_i, 1), np.nan)
    Bmap = np.where(n_d > 30, n_o / np.maximum(n_d, 1) / Dm, np.nan) / fits_out[fid]["r_rs350"]["n0"]
    raw = np.where(n_i > 30, n_o / np.maximum(n_i, 1), np.nan)
    res["B_map_200kpc"] = {"edges_kpc": e2.tolist(), "B": Bmap.T.tolist(), "n_obs": n_o.T.tolist()}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    kL_img = L.at(*WCS(fits.getheader(LIVE / "f150w" / (ROOT.format("f150w") + ".fits"))).pixel_to_world_values(
        *np.meshgrid(asd().A2744_STAMP_CENTRE[0] - 1 + np.linspace(-1000, 1000, 101) / KPC_PER_PIX,
                     asd().A2744_STAMP_CENTRE[1] - 1 + np.linspace(-1000, 1000, 101) / KPC_PER_PIX)))[0]
    for a, img, lab, vr in ((ax[0], raw / np.nanmedian(raw), "n_obs / covered area (rel.)", (0.4, 1.6)),
                            (ax[1], Dm, f"magnification depletion D [{fid}]", (0.2, 0.8)),
                            (ax[2], Bmap, f"B = n_obs/(p D)/n0 [{fid}]", (0.5, 3.))):
        im = a.imshow(img.T, origin="lower", extent=[-1000, 1000, -1000, 1000], vmin=vr[0], vmax=vr[1], cmap="viridis")
        a.contour(np.linspace(-1000, 1000, 101), np.linspace(-1000, 1000, 101), kL_img, levels=[0.2, 0.3, 0.5],
                  colors="w", linewidths=0.7)
        plt.colorbar(im, ax=a, label=lab)
        a.set_xlabel("dx [kpc]")
    ax[0].set_ylabel("dy [kpc]")
    plt.tight_layout(); plt.savefig(OUT / "B_map.png", dpi=110); plt.close()
    json.dump(res, open(OUT / "injection_dilution.json", "w"), indent=1, default=float)
    plots(res, rec, lowk, det, kL_inj, plt)
    report(res)


def plots(res, rec, lowk, det, kL_inj, plt):
    E = res["edges"]
    rp, kp = res["profiles_r"]["flux"], res["profile_kappa_sl"]
    # p(kappa), p(r)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    xc = np.array(kp["xc"])
    ax[0].errorbar(xc, kp["p"], kp["pe"], fmt="o", color="C0", label="kappa_sl < 0.7 positions")
    ax[0].set_xlabel(r"Lenstool $\kappa$ (sl_kappa_zs2.28, nearest pixel; 0 = off map)")
    ax[0].set_ylabel("p (detected + selected)")
    ax[0].axhline(res["field_p"], color="gray", ls=":", label=f"field (r > 700 kpc) {res['field_p']:.3f}")
    ax[0].legend(fontsize=8)
    r = np.array(rp["xc"])
    ax[1].errorbar(r, rp["p"], rp["pe"], fmt="o", label=r"$\kappa < 0.7$ positions")
    pa = res["p_r_including_kappa_cut"]
    ax[1].plot(r, pa["p"], "s", mfc="none", color="C3", label=r"all positions (incl. $\kappa \geq 0.7$ cut)")
    ax[1].axhline(res["field_p"], color="gray", ls=":")
    ax[1].set_xlabel("r from stamp centre [kpc]"); ax[1].set_ylabel("p")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.set_ylim(0, None)
    plt.tight_layout(); plt.savefig(OUT / "p_kappa_r.png", dpi=110); plt.close()

    # p map
    fig, ax = plt.subplots(figsize=(5.5, 5))
    e = np.linspace(-1000, 1000, 21)
    n = np.histogram2d(rec["dx_kpc"], rec["dy_kpc"], [e, e])[0]
    k = np.histogram2d(rec["dx_kpc"][det], rec["dy_kpc"][det], [e, e])[0]
    im = ax.imshow((k / np.where(n > 5, n, np.nan)).T, origin="lower", extent=[-1000, 1000, -1000, 1000], vmin=0, vmax=0.8)
    plt.colorbar(im, label="p (100 kpc pixels)")
    ax.set_xlabel("dx [kpc]"); ax.set_ylabel("dy [kpc]")
    plt.tight_layout(); plt.savefig(OUT / "p_map.png", dpi=110); plt.close()

    # n_obs / n_field and B with fits
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    rr = np.linspace(20, 1414, 300)
    for mode, col in zip(res["fits"], ("C0", "C2", "C4")):
        d = res["profiles_r"][mode]
        fo = res["fits"][mode]
        n0 = fo["r_rs350"]["n0"]
        r = np.array(d["xc"])
        dens = np.array(d["dens"])
        if mode == res["fiducial_magnification"]:
            n0raw = res["fits"][mode]["r_rs350_no_obscuration"]["n0"]
            ax[0].errorbar(r, dens / np.nanmean(dens[-4:]), dens / np.nanmean(dens[-4:]) / np.sqrt(np.maximum(d["n_obs"], 1)),
                           fmt="s", mfc="none", color="gray", label=r"$n_{\rm obs}$ (raw)")
            ax[0].plot(r, np.array(d["dens"]) / np.array(d["p"]) / np.nanmean((dens / np.array(d["p"]))[-4:]), "^",
                       color="C1", label=r"$n_{\rm obs}/p$")
            ax[0].plot(r, np.array(d["D"]), "-", color="C3", label=r"magnification $\langle\mu^{-1}N(\mu)/N\rangle$ [" + mode + "]")
        ax[1].errorbar(r * (1 + 0.01 * list(res["fits"]).index(mode)), np.array(d["corr"]) / n0, np.array(d["corr_e"]) / n0,
                       fmt="o", color=col, label=f"B: n/(p D) / n0 [{mode} magnification]")
        f1, f2 = fo["r_rs350"], fo["r_rsfree"]
        ax[1].plot(rr, 1 + f1["A"] * np.exp(1 - rr / R_S), color=col,
                   label=f"A = {f1['A']:.3f} ± {f1['A_err']:.3f} (r_s = 350)")
        ax[1].plot(rr, (f2["n0"] / n0) * (1 + f2["A"] * np.exp(1 - rr / f2["r_s"])), color=col, ls="--",
                   label=f"A = {f2['A']:.3f} ± {f2['A_err']:.3f}, r_s = {f2['r_s']:.0f} ± {f2['r_s_err']:.0f}")
    ax[0].axhline(1, color="k", lw=0.5)
    ax[0].set_xlabel("r [kpc]"); ax[0].set_ylabel("relative to r > 900 kpc"); ax[0].legend(fontsize=8)
    ax[1].axhline(1, color="k", lw=0.5)
    ax[1].set_xlabel("r [kpc]"); ax[1].set_ylabel(r"$B(r)$"); ax[1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(OUT / "n_obs_and_B_r.png", dpi=110); plt.close()

    # B(kappa) with the kappa-recast fit
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    prof = res["a2744_kappa_profile"]
    kb, rc = np.array(prof["kappa"]), np.array(prof["r_kpc"])
    o = np.argsort(kb)
    kk = np.linspace(0.02, 0.75, 200)
    for mode, col in zip(res["fits"], ("C0", "C2", "C4")):
        d = res["profiles_kappaL"][mode]
        fo = res["fits"][mode]["kappa_rs350"]
        ax.errorbar(np.array(d["xc"]), np.array(d["corr"]) / fo["n0"], np.array(d["corr_e"]) / fo["n0"], fmt="o",
                    color=col, label=f"B [{mode} magnification]")
        ax.plot(kk, 1 + fo["A"] * np.exp(1 - np.interp(kk, kb[o], rc[o]) / R_S), color=col,
                label=f"A = {fo['A']:.3f} ± {fo['A_err']:.3f} (r_s = 350)")
    ax.axhline(1, color="k", lw=0.5)
    ax.set_xlabel(r"Lenstool $\kappa$ ($z_s$ = 2.28, deflection maps)"); ax.set_ylabel(r"$B(\kappa)$")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / "B_kappa.png", dpi=110); plt.close()


def fit_variants():
    '''Robustness of A (r_s = 350 kpc) for the fiducial magnification model, from injection_dilution.json:
    fit range r < 900 kpc (drops the outer bins, which carry the northern substructure), field level n0 fixed to
    the 400-900 kpc mean, and A from the inner bins alone (B(r < 250 kpc) matched with n0 fixed).'''
    res = json.load(open(OUT / "injection_dilution.json"))
    out = {}
    for mode in res["profiles_r"]:
        d = {k: np.array(v, float) for k, v in res["profiles_r"][mode].items()}
        r, y, e = d["xc"], d["corr"], d["corr_e"]
        ok = np.isfinite(y) & (e > 0)
        n0_out = np.average(y[ok & (r > 400) & (r < 900)], weights=1 / e[ok & (r > 400) & (r < 900)] ** 2)
        model = lambda r, n0, A: n0 * (1 + A * np.exp(1 - r / R_S))
        v = {}
        sel = ok & (r < 900)
        po, pe, chi2, dof = fit_boost(r[sel], y[sel], e[sel], model, [n0_out, 0.1], ([0, -1], [10, 10]))
        v["r<900_free_n0"] = dict(A=po[1], A_err=pe[1], A_err_scaled=pe[1] * np.sqrt(max(chi2 / dof, 1)), chi2=chi2, dof=int(dof))
        for name, rmax in (("n0_fixed_400-900", 1500), ("n0_fixed_inner_r<250", 250)):
            sel = ok & (r < rmax)
            po, pe, chi2, dof = fit_boost(r[sel], y[sel], e[sel], lambda r, A: model(r, n0_out, A), [0.1], ([-1], [10]))
            v[name] = dict(A=po[0], A_err=pe[0], A_err_scaled=pe[0] * np.sqrt(max(chi2 / max(dof, 1), 1)), chi2=chi2, dof=int(dof))
        B = y / n0_out
        Be = e / n0_out
        v["B_bins_n0_400-900"] = dict(r=r.tolist(), B=B.tolist(), B_err=Be.tolist())
        for name in list(v):
            if "A" in v[name]:
                v[name]["B100"] = 1 + v[name]["A"] * np.exp(1 - 100 / R_S)
                v[name]["B300"] = 1 + v[name]["A"] * np.exp(1 - 300 / R_S)
        out[mode] = v
    json.dump(out, open(OUT / "fit_variants.json", "w"), indent=1, default=float)
    for mode, v in out.items():
        for name, f in v.items():
            if "A" in f:
                print(f"[{mode}] {name}: A = {f['A']:.3f} ± {f['A_err']:.3f} (± {f['A_err_scaled']:.3f}), "
                      f"B100 {f['B100']:.3f}, B300 {f['B300']:.3f}, chi2 {f['chi2']:.1f}/{f['dof']}")


def report(res):
    print(f"{res['n_realisations']} realisations, {res['n_injected']} injected; field p = {res['field_p']:.3f}; "
          f"count slope s = {res['count_slope_s']}")
    for mode, fo in res["fits"].items():
        for name, f in fo.items():
            extra = f", r_s = {f['r_s']:.0f} ± {f['r_s_err']:.0f}, B(100) = {f['B100']:.3f}, B(300) = {f['B300']:.3f}" \
                if "B100" in f else ""
            sc = f" (± {f['A_err_scaled']:.3f} scaled)" if "A_err_scaled" in f else ""
            print(f"  [{mode}] {name}: A = {f['A']:.3f} ± {f['A_err']:.3f}{sc}{extra}, chi2/dof = {f['chi2']:.1f}/{f['dof']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("step")
    p.add_argument("args", nargs="*")
    a = p.parse_args()
    {"cutout": cutout, "population": population, "pilot": pilot, "inject": inject, "recover": recover,
     "baseline": baseline, "analyse": analyse,
     "population_broad": population_broad, "inject_mag": inject_mag,
     "fit_variants": fit_variants}[a.step](*a.args)
