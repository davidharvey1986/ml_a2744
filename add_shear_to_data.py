#!/usr/bin/env python
'''
Turn simulated convergence maps into shear maps for training.

main(sample_data=False) writes ideal (gridded) shear maps for the base models;
main(sample_data=True) samples the shear at the observed A2744 galaxy positions
and re-bins it, for the final masked models.

Also holds the observed-catalogue helpers (get_obs_data, combine_catalogues, ...)
and model-loading helpers that the notebooks import from here.
'''

import os
import pickle as pkl
import re
from glob import glob

import numpy as np
import numpy.lib.recfunctions as rfn
from astropy import units
from astropy.cosmology import Planck18
from astropy.io import fits
from lenspack.image.inversion import ks93inv
from lenspack.utils import sigma_critical, bin2d
from scipy.interpolate import make_interp_spline, BSpline
from scipy.sparse import csr_matrix
from tqdm import tqdm

from pyRRG.calc_shear import calc_shear
from RRGtools import run_match

from model import create_model

# Directory holding the extracted halo components (*.comps), relative to the repo root
COMPONENTS_DIR = "notebooks/pickles"

# Galaxies where the convergence (at zs=1.72) is at least this are dropped when the
# simulated shear is sampled at the A2744 galaxy positions
KAPPA_THRESH = 0.9

# Shear-catalogue cuts used for the A2744 data map (prepare_uncovers_data)
SHEAR_CUTS = {'f115w': {'signal_noise_cut': 0, 'stat_type': 'snr'},
              'f150w': {'signal_noise_cut': 5, 'stat_type': 'snr'}}


def load_pickle(path):
    with open(path, "rb") as f:
        return pkl.load(f)


def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pkl.dump(obj, f)


def get_boxsize(set_name, return_units=units.pc, h=0.7):
    if ('bahamas' in set_name) or ('tng' in set_name) or ('flamingo' in set_name):
        boxsize = 10./h*units.Mpc
    else:
        boxsize = 400/h*units.Mpc
    return boxsize.to(return_units)


def crop_center(img, cropx, cropy):
    y, x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]


def main(
    search_path="data/100/convergence/*.pkl",
    filter_list=['concat'],
    thresh_k=KAPPA_THRESH,
    sample_data=True,
    data_dir="data/100/a2744",
    reduce_shear=True,
    zl=0.305, zs=1.72,
    add_ncomps=True
    ):
    '''
    Note - zs=1.72 is rescaled during training to the true redshift distribution
    so this is a place holder.
    '''
    image_size = 100
    shear_kwargs = dict(zs=zs, zl=zl, ngal_per_sq_arcmin=200.)

    all_data_sets = glob(search_path)

    ### This is for the base_models
    if not sample_data:
        for idata_set in tqdm(all_data_sets):

            meta, data = load_pickle(idata_set)

            if add_ncomps:
                meta['ncomponents'] = get_num_merging_components(idata_set)

            if 'redshift' not in meta.keys():
                meta['redshift'] = np.full(data.shape[0], 0.250, dtype=np.float64)

            meta['norms'] = get_surface_density_norms(meta, idata_set)

            # All base-model clusters are placed at the A2744 redshift
            redshifts = np.full(data.shape[0], 0.305)
            e1, e2, kappa = data_to_shear(
                data[:, 0], meta['norms'][:, 0],
                redshifts,
                get_boxsize(idata_set),
                reduce_shear=reduce_shear,
                background=get_background_sigma(meta, idata_set, redshifts),
                **shear_kwargs
            )

            data[:, 0, :, :] = kappa
            add_e1e2 = np.concatenate([e1[:, None], e2[:, None], data], axis=1)

            save_pickle([meta, add_e1e2], idata_set.replace("convergence", "shear"))

        return

    ### This is for the final masked models
    for ifilter in filter_list:

        obs_data = get_obs_data(ifilter, data_dir=data_dir, photoz=True)

        delta_ra, delta_dec = ra_dec_to_simulation_image_pos(
            obs_data, zl=zl, image_size=image_size
        )

        for idata_set in tqdm(all_data_sets):

            meta, data = load_pickle(idata_set)

            if 'redshift' not in meta.keys():
                meta['redshift'] = np.full(data.shape[0], 0.305, dtype=np.float64)

            meta['norms'] = get_surface_density_norms(meta, idata_set)

            #Interpolate the non reduced shear then reduce it later.
            e1_ideal, e2_ideal, kappa = data_to_shear(
                data[:, 0], meta['norms'][:, 0],
                meta['redshift'],
                get_boxsize(idata_set),
                reduce_shear=False,
                background=get_background_sigma(meta, idata_set, meta['redshift']),
                **shear_kwargs
            )

            new_data, _ = sample_maps_on_galaxies(
                e1_ideal, e2_ideal, kappa,
                delta_ra, delta_dec,
                thresh_k=thresh_k, reduce_shear=reduce_shear, npix=image_size
            )

            save_pickle([meta, new_data], idata_set.replace('convergence', f'obs/{ifilter}'))


def cubic_spline_sampler(grid, ra, dec):
    '''
    Precompute the not-a-knot bicubic spline (as RegularGridInterpolator
    method='cubic') for fixed sample points, so many maps can be interpolated
    with matrix products. Returns (coeffs, sample_matrix) such that for a map
    M indexed [dec, ra], the interpolated values are
        sample_matrix @ (coeffs @ M @ coeffs.T).ravel()
    sample_matrix is sparse: each point touches a 4x4 patch of coefficients.
    '''
    n = len(grid)
    spline = make_interp_spline(grid, np.eye(n), k=3)
    coeffs = spline.c   # maps grid values -> B-spline coefficients

    def basis(points):
        design = BSpline.design_matrix(np.clip(points, grid[0], grid[-1]), spline.t, 3)
        design = design.tocsr()
        return design.indices.reshape(-1, 4), design.data.reshape(-1, 4)

    idx_dec, w_dec = basis(dec)
    idx_ra, w_ra = basis(ra)
    cols = (idx_dec[:, :, None]*n + idx_ra[:, None, :]).reshape(len(ra), 16)
    weights = (w_dec[:, :, None]*w_ra[:, None, :]).reshape(len(ra), 16)
    rows = np.repeat(np.arange(len(ra)), 16)
    sample_matrix = csr_matrix((weights.ravel(), (rows, cols.ravel())), shape=(len(ra), n*n))
    return coeffs, sample_matrix


def sample_maps_on_galaxies(
        e1_maps, e2_maps, kappa_maps, ra, dec,
        thresh_k=0.9, reduce_shear=True, npix=100, batch_size=32
    ):
    '''
    Interpolate (non-reduced) shear and convergence maps onto the galaxy
    positions, drop galaxies with kappa >= thresh_k, reduce the shear and
    re-bin onto an npix x npix grid over [-npix/2, npix/2].

    Equivalent to building a cubic RegularGridInterpolator per cluster and
    calling bin2d, but done for a batch of clusters at once since the galaxy
    positions are the same for every cluster. Galaxies outside the pixel-centre
    grid take the median non-zero value of each map, as fill_value would.

    Returns the binned (e1, e2) maps, shape (nclusters, 2, npix, npix), and the
    percentage of galaxies removed by the kappa cut for each cluster.
    '''
    grid = np.arange(npix) - npix/2. + 0.5
    coeffs, sample_matrix = cubic_spline_sampler(grid, ra, dec)

    # Same out-of-bounds rule as RegularGridInterpolator
    out_of_grid = (ra < grid[0]) | (ra > grid[-1]) | (dec < grid[0]) | (dec > grid[-1])

    # Same pixel assignment as np.histogram2d in bin2d (right edge in last bin)
    edges = np.linspace(-npix/2., npix/2., npix+1)
    ira = np.clip(np.searchsorted(edges, ra, side='right') - 1, 0, npix-1)
    idec = np.clip(np.searchsorted(edges, dec, side='right') - 1, 0, npix-1)
    in_image = (ra >= edges[0]) & (ra <= edges[-1]) & (dec >= edges[0]) & (dec <= edges[-1])
    flat_pix = idec*npix + ira   # bin2d returns maps indexed [dec, ra]

    def interp(maps):
        # coeffs @ maps[b] @ coeffs.T for every b, written as two 2D matmuls
        # (numpy's broadcast matmul on a stack is ~30x slower here)
        nbatch = maps.shape[0]
        right = (maps.reshape(-1, npix) @ coeffs.T).reshape(nbatch, npix, npix)
        both = coeffs @ right.transpose(1, 0, 2).reshape(npix, -1)
        spline_coeffs = both.reshape(npix, nbatch, npix).transpose(1, 0, 2).reshape(nbatch, -1)
        vals = (sample_matrix @ spline_coeffs.T).T
        fill_values = np.array([np.median(m[m != 0]) for m in maps])
        vals[:, out_of_grid] = fill_values[:, None]
        return vals

    nclusters = e1_maps.shape[0]
    new_data = np.zeros((nclusters, 2, npix, npix))
    remove = np.zeros(nclusters)
    for start in tqdm(range(0, nclusters, batch_size), leave=False):
        sl = slice(start, start+batch_size)
        nbatch = e1_maps[sl].shape[0]

        e1 = interp(e1_maps[sl])
        e2 = interp(e2_maps[sl])
        k = interp(kappa_maps[sl])

        cut = k < thresh_k
        remove[sl] = (~cut).sum(axis=1) / ra.shape[0] * 100.
        if reduce_shear:
            with np.errstate(divide='ignore', invalid='ignore'):
                e1 = e1 / (1. - k)
                e2 = e2 / (1. - k)

        use = cut & in_image[None, :]
        bins = (np.arange(nbatch)[:, None]*npix*npix + flat_pix[None, :])[use]
        nmap = npix*npix*nbatch
        counts = np.bincount(bins, minlength=nmap)
        with np.errstate(divide='ignore', invalid='ignore'):
            for ic, comp in enumerate((e1, e2)):
                sums = np.bincount(bins, weights=comp[use], minlength=nmap)
                new_data[sl, ic] = np.where(counts > 0, sums / counts, 0.).reshape(nbatch, npix, npix)

    return new_data, remove


def get_components_file(dataset, components_dir=COMPONENTS_DIR):
    return f"{components_dir}/{os.path.basename(dataset)}.comps"


def get_convergence_file(dataset):
    '''The convergence map matching a convergence, shear or obs file (same name under data/<npix>/convergence).'''
    if "/convergence/" in dataset:
        return dataset
    root = re.match(r"(.*data/\d+)/", dataset)
    if root is None:
        raise ValueError(f"Cannot find the convergence map for {dataset}")
    return f"{root.group(1)}/convergence/{os.path.basename(dataset)}"


def get_num_merging_components(
        dataset,
        mass_ratio_limit=10,
        components_dir=COMPONENTS_DIR,
    ):
    '''
    Number of components with mass (flux) ratio to the main one below mass_ratio_limit,
    per map. If the component file does not exist it is made first with
    extract_num_merging_components, from the matching convergence map.
    '''
    comp_name = get_components_file(dataset, components_dir)
    if not os.path.isfile(comp_name):
        print(f"{comp_name} not found: extracting components from {get_convergence_file(dataset)}")
        extract_num_merging_components([get_convergence_file(dataset)], components_dir=components_dir)
    extracted_components = load_pickle(comp_name)
    ncomponents = []
    for icomp in extracted_components:
        mass_ratio = icomp['FLUX_AUTO'].max() / icomp['FLUX_AUTO']
        ncomponents.append(np.sum(mass_ratio < mass_ratio_limit))
    return np.array(ncomponents)


def extract_num_merging_components(dataset=None, zoomed_amount=2, components_dir=COMPONENTS_DIR):
    '''
    Run the component extractor over every convergence map and save the
    components where get_num_merging_components expects them.
    '''
    # component_extractor is a separate package, only needed here
    from component_extractor import component_extractor
    from scipy.ndimage import zoom

    if dataset is None:
        dataset = glob("data/100/convergence/darkskies_*")

    for ishear in dataset:

        _, all_halos = load_pickle(ishear)

        all_components = []
        for ihalo in tqdm(all_halos):

            extracted_components = component_extractor(
                zoom(ihalo[0, :, :], zoomed_amount),
                use_sex=True,
            )
            extracted_components['X_IMAGE'] /= zoomed_amount
            extracted_components['Y_IMAGE'] /= zoomed_amount

            all_components.append(extracted_components)
        save_pickle(all_components, get_components_file(ishear, components_dir))


def is_physical_surface_density(meta):
    '''
    Maps made with the fixed darkskies_images.py store Msun / pMpc^2 on a
    physical stamp and record it in meta['map_units']. Older maps do not.
    '''
    return 'map_units' in meta.keys()


def get_surface_density_norms(meta, set_name):
    '''
    Peak surface density of each map in Msun / Mpc^2 (physical).

    Old darkskies/flamingo maps were mass per (2 cMpc)^2, hence the /4. New maps
    (meta['map_units'] present) are already Msun / pMpc^2 and must not be divided.
    '''
    norms = np.array(meta['norms'], dtype=np.float64)
    if (('darkskies' in set_name) or ('flamingo' in set_name)) and not is_physical_surface_density(meta):
        norms /= 4.
    return norms


def get_background_sigma(meta, set_name, redshifts):
    '''
    Mean matter surface density through each projection in Msun / pc^2 (physical),
    subtracted from the map in data_to_shear.

    New maps record it per image in meta['background_sigma'] (Msun / pMpc^2), which
    accounts for the actual comoving projection depth and the snapshot redshift.
    Older maps fall back to the previous approximation.
    '''
    redshifts = np.atleast_1d(redshifts)
    boxsize = get_boxsize(set_name)
    old = np.array([
        (Planck18.critical_density(z).to(units.Msun/units.pc**3)*boxsize*Planck18.Om0).value
        for z in redshifts
    ])
    if 'background_sigma' not in meta.keys():
        return old*units.Msun/units.pc**2

    new = np.array([np.nan if b is None else b for b in meta['background_sigma']], dtype=np.float64)
    new = new*1e-12          # Msun / pMpc^2 -> Msun / pc^2
    return np.where(np.isfinite(new), new, old)*units.Msun/units.pc**2


def data_to_shear(images, norms, redshifts, boxsize,
                  reduce_shear=True,
                  kappa_thresh=None,
                  background=None,
                  zl=0.3,
                  zs=2.,
                  norm_units=units.Msun/(units.Mpc*units.Mpc),
                  **kwargs
                 ):
    '''
    Convert surface density maps to convergence and (noise free) shear via
    Kaiser-Squires.

    images : maps normalised to a peak of 1, shape (N, npix, npix)
    norms : peak surface density of each map, in norm_units
    redshifts, boxsize : only used for the old background approximation
    background : optional per-image mean surface density (astropy quantity,
        e.g. from get_background_sigma). If None, use the old
        critical_density * boxsize * Om0 approximation.
    kappa_thresh : if set, zero the shear where kappa exceeds it
    kwargs : accepted for backwards compatibility (e.g. ngal_per_sq_arcmin)
        but unused, since no noise is added here.
    '''
    sigma_crit = sigma_critical(zl, zs, Planck18)

    peak = norms*norm_units
    peak_kappa = peak.to(units.Msun/units.pc/units.pc)/sigma_crit

    all_e1, all_e2, all_converge = [], [], []

    for idx, image in enumerate(images):

        #I sum over all z so need to remove the mean density along the line of sight
        if background is None:
            mean_density = Planck18.critical_density(redshifts[idx]).to(units.Msun/units.pc**3)*boxsize*Planck18.Om0
        else:
            mean_density = background[idx]

        convergence = image*peak_kappa[idx] - mean_density/sigma_crit

        e1, e2 = ks93inv(convergence, convergence*0.)

        if reduce_shear:
            e1 /= 1.-convergence
            e2 /= 1.-convergence

        if kappa_thresh is not None:
            e1[convergence > kappa_thresh] = 0.
            e2[convergence > kappa_thresh] = 0.

        all_e1.append(e1)
        all_e2.append(e2)
        all_converge.append(convergence)

    return np.array(all_e1), np.array(all_e2), np.array(all_converge)


def get_obs_data(
    ifilter,
    cuts=None,
    data_dir='data/100/a2744',
    photoz=False,
    remove_members=True,
    redshift_cut=0.305+0.05*3):
    '''
    Apply the shear cuts to the f115w and f150w catalogues (rewriting
    a2744_<filter>_filtered.fits), optionally remove cluster members and,
    for ifilter='concat', combine the two. With photoz=True a 'redshift' field
    is added from the UNCOVER photo-z's, defaulting to the lensing-weighted
    mean for unmatched galaxies.
    '''
    cuts = {} if cuts is None else dict(cuts)

    # Cuts already taken during the WL process - verified.
    fid_cuts = {
            'size_cut': [2, 200],
            'signal_noise_cut': 0,
            'stat_type': 'median',
            'mag_cut': [0, 30],
            'verbose': False
        }

    for this_filter in ['f115w', 'f150w']:
        if this_filter not in cuts:
            print(f"Missing {this_filter} in cuts dict referring to default")
        filter_cuts = {**fid_cuts, **cuts.get(this_filter, {})}

        filtered_name = f"{data_dir}/a2744_{this_filter}_filtered.fits"
        shear_cat = f"{data_dir}/abell2744clu-grizli-v5.4-{this_filter}-clear_drc_sci_clean.shears"

        calc_shear(fits.open(shear_cat)[1].data, filtered_name, **filter_cuts)

        if remove_members:
            obs_data = remove_cluster_members(
                filtered_name,
                f"{data_dir}/UNCOVER_DR4_SPS_catalog.fits",
                redshift_cut=redshift_cut
            )
            fits.writeto(filtered_name, obs_data, overwrite=True)

    filtered_name = f"{data_dir}/a2744_{ifilter}_filtered.fits"
    if ifilter == 'concat':
        obs_data = combine_catalogues(
            f"{data_dir}/a2744_f115w_filtered.fits",
            f"{data_dir}/a2744_f150w_filtered.fits",
        )
        fits.writeto(filtered_name, obs_data, overwrite=True)
    else:
        obs_data = fits.open(filtered_name)[1].data

    if photoz:
        photo_z_matched = run_match(
            f"{data_dir}/UNCOVER_DR4_SPS_catalog.fits",
            filtered_name,
            search_rad=0.5
        )[1].data

        default_zs = sig_mean(photo_z_matched['z_ml'])

        print(f"DEFAULT ZS {default_zs}")

        z_lookup = dict(zip(photo_z_matched['NUMBER'], photo_z_matched['z_ml']))
        redshift = np.array([z_lookup.get(number, np.nan) for number in obs_data['NUMBER']],
                            dtype=np.float64)
        redshift[~np.isfinite(redshift)] = default_zs

        obs_data = rfn.append_fields(
            obs_data,
            names='redshift',
            data=redshift,
            dtypes=np.float64,
            usemask=False,
            asrecarray=True
        )

    return obs_data


def remove_cluster_members(cat_raw, photoz, redshift_cut=0.):

    combined = run_match(cat_raw, photoz, search_rad=0.5)

    all_cat = fits.open(cat_raw)[1].data

    cluster_numbers = combined[1].data['NUMBER'][combined[1].data['z_ml'] < redshift_cut]

    return all_cat[~np.isin(all_cat['NUMBER'], cluster_numbers)]


def combine_catalogues(cat_a_name, cat_b_name, identifier='NUMBER'):
    '''
    Merge two shear catalogues: galaxies matched in both get the mean
    shear/ellipticity (and the first catalogue's other properties), unmatched
    galaxies from either catalogue are appended. Galaxies are renumbered from 1.
    '''
    cat_a = fits.open(cat_a_name)[1].data
    cat_b = fits.open(cat_b_name)[1].data

    matched_cat = run_match(cat_a_name, cat_b_name)[1].data

    extra_a = ~np.isin(cat_a[identifier], matched_cat[f'{identifier}_1'])
    extra_b = ~np.isin(cat_b[identifier], matched_cat[f'{identifier}_2'])

    def stack(matched, name):
        return np.concatenate([matched, cat_a[name][extra_a], cat_b[name][extra_b]])

    def average(name):
        return stack((matched_cat[f'{name}_1'] + matched_cat[f'{name}_2'])/2., name)

    def first(name):
        return stack(matched_cat[f'{name}_1'], name)

    obs_data = {
        'x': first('x'),
        'y': first('y'),
        'gamma1': average('gamma1'),
        'gamma2': average('gamma2'),
        'e1': average('e1'),
        'e2': average('e2'),
        'RA': first('ra'),
        'DEC': first('dec'),
    }
    if 'z_1' in matched_cat.dtype.names:
        obs_data['z'] = first('z')
    obs_data['NUMBER'] = np.arange(obs_data['DEC'].shape[0])+1
    obs_data['MAG'] = first('MAG_AUTO')
    obs_data['SIZE'] = first('gal_size')

    new_cols = [
        fits.Column(name=ikey, format=obs_data[ikey].dtype, array=obs_data[ikey])
        for ikey in obs_data
    ]
    return fits.BinTableHDU.from_columns(new_cols).data


def bin_obs_data(in_obs_data, image_size=100, npix=100):

    obs_data = in_obs_data.copy()

    g = np.sqrt(obs_data['gamma1']**2 + obs_data['gamma2']**2)

    obs_data['gamma1'] = 2.*obs_data['gamma1'] / (1.+g**2)
    obs_data['gamma2'] = 2.*obs_data['gamma2'] / (1.+g**2)

    delta_ra, delta_dec = ra_dec_to_simulation_image_pos(obs_data)

    extent = [-image_size//2, image_size//2, -image_size//2, image_size//2]

    e1_radec, e2_radec = bin2d(
        delta_ra, delta_dec,
        v=(obs_data['gamma1'], obs_data['gamma2']),
        npix=npix,
        extent=extent
    )

    ### TO SPEED UP THE AUGMENTATION ####
    edges = np.linspace(-image_size//2, image_size//2, npix+1)

    ix = np.digitize(delta_ra, edges) - 1
    iy = np.digitize(delta_dec, edges) - 1

    valid = (
        (ix >= 0) & (ix < npix) &
        (iy >= 0) & (iy < npix)
    )

    flat_bin = iy[valid] * npix + ix[valid]

    ###################

    ngal = bin2d(delta_ra, delta_dec, v=None, npix=npix, extent=extent)

    max_val = np.max([e1_radec, e2_radec])
    min_val = np.min([e1_radec, e2_radec])

    e1_radec -= min_val
    e1_radec /= max_val

    e2_radec -= min_val
    e2_radec /= max_val

    return {'e1': e1_radec, 'e2': e2_radec, 'ngal': ngal,
            'delta_ra': delta_ra, 'delta_dec': delta_dec,
            'valid': valid, 'flat_bin': flat_bin}


def ra_dec_to_simulation_image_pos(
    obs_data,
    fov_sim=2e3*units.kpc,
    jwst_pixel_size=0.02*units.arcsecond,
    zl=0.305,
    image_size=100,
    ra_0=None, dec_0=None,
    ):
    '''
    Convert catalogue pixel positions (x, y) to simulation pixel coordinates,
    centred on (ra_0, dec_0) (default: the median position) and scaled so that
    fov_sim at zl spans [-image_size//2, image_size//2].
    '''
    arcsec_per_kpc = units.radian.to(units.arcsecond)/Planck18.angular_diameter_distance(zl).to(units.kpc)

    fov_sim_arcsec = fov_sim * arcsec_per_kpc / 2.

    if ra_0 is None:
        ra_0 = np.median(obs_data['x'])
    if dec_0 is None:
        dec_0 = np.median(obs_data['y'])
    delta_ra = (obs_data['x'] - ra_0)*jwst_pixel_size
    delta_dec = (obs_data['y'] - dec_0)*jwst_pixel_size

    delta_ra /= fov_sim_arcsec / (image_size//2)
    delta_dec /= fov_sim_arcsec / (image_size//2)

    return delta_ra.value, delta_dec.value


def rebin(arr, Q):
    """
    Rebin an array of shape (N, M, P, P) to (N, M, Q, Q)
    using block-wise averaging, then repeat back up to (N, M, P, P).

    Parameters
    ----------
    arr : ndarray
        Input array of shape (N, M, P, P).
    Q : int
        Resolution of the block average.

    Returns
    -------
    ndarray
        Array of shape (N, M, P, P), piecewise constant on a Q x Q grid.
    """
    N, M, P1, P2 = arr.shape
    if P1 != P2:
        raise ValueError("Last two dimensions must be square.")
    if P1 % Q != 0:
        raise ValueError("P must be divisible by Q.")

    factor = P1 // Q

    coarse = arr.reshape(
        N, M,
        Q, factor,
        Q, factor
    ).mean(axis=(3, 5))

    return np.repeat(
        np.repeat(coarse, factor, axis=2),
        factor, axis=3
    )


def resize_data(
    new_resolution,
    fiducual_res=100,
    dir_path="/Users/davidharvey/Work/Dark_Matter_DA/code_nature/data/"
    ):
    '''
    A referee asked to see depenence on the resolution of the image
    so i need to rebin the images. but i want to do this at the last stage
    so i dont FFT on low res images
    '''
    # Only create data that i have to
    for root, _, files in os.walk(dir_path):
        if 'convergence' in root:
            continue
        for ifile in files:
            if 'pkl' not in ifile:
                continue

            newroot = root.replace("code_nature/data", f"code/data/{int(new_resolution)}")
            old_path = f"{root}/{ifile}"
            new_path = f"{newroot}/{ifile}"
            if new_path == old_path:
                raise ValueError("Overwriting original data")

            meta, data = load_pickle(old_path)
            save_pickle([meta, rebin(data, new_resolution)], new_path)


def get_model_names(
        ifilter='concat',
        model_dir="../models",
        model='final',
        exclude_str=None,
        include_str=None,
        load_model_args=None,
        nmodels=None
    ):

    list_of_models = glob(f"{model_dir}/{ifilter}/*{model}.pth")

    if exclude_str is not None:
        list_of_models = [i for i in list_of_models if exclude_str not in i]

    if include_str is not None:
        list_of_models = [i for i in list_of_models if include_str in i]

    if nmodels is not None:
        list_of_models = list_of_models[:nmodels]

    assert len(list_of_models) > 0, f"No models found check path ({model_dir})"

    #get the seed number index
    names_in_model = np.array(os.path.basename(list_of_models[0]).split('_'))
    seed_index = np.where(names_in_model == 'seed')[0][0] + 1

    if load_model_args is not None:
        all_models = []
        for imodel in list_of_models:
            load_model_args.checkpoint = imodel
            all_models.append(create_model(load_model_args))

        return np.array(all_models), seed_index
    return np.array(list_of_models), seed_index


def sig_mean(redshift, zl=0.305):
    '''Mean source redshift weighted by 1 / Sigma_crit.'''
    sig = sigma_critical(zl, redshift, Planck18).value
    return np.sum(redshift/sig)/np.sum(1/sig)


def make_noise_model(ifilter='concat', data_dir='data/100/a2744', cuts=None):
    '''
    Write <data_dir>/ngal_<ifilter>.pkl, read by dataset.apply_intrinsic_ell: the binned
    A2744 catalogue (galaxy-to-pixel assignment, bin_obs_data) and a chi fit to the
    galaxy ellipticities, so the simulated shape noise sits at the same galaxies as the
    data map.
    '''
    from scipy.stats import chi
    obs_data = get_obs_data(ifilter, data_dir=data_dir, cuts=SHEAR_CUTS if cuts is None else cuts)
    binned_data = bin_obs_data(obs_data)
    egal = np.sqrt(obs_data['e1']**2 + obs_data['e2']**2)
    noise_model = {'ngal': binned_data['ngal'], 'chi_fit_function': chi.fit(egal),
                   'binned_data': binned_data}
    out = f"{data_dir}/ngal_{ifilter}.pkl"
    save_pickle(noise_model, out)
    return out


def get_source_redshift(ifilter, data_dir='data/100/a2744', cuts=None):

    if cuts is None:
        cuts = SHEAR_CUTS
    data = get_obs_data(ifilter, photoz=True, data_dir=data_dir, cuts=cuts)

    #remove artifical redshifts
    nz = data['redshift'][np.abs(data['redshift'] - np.median(data['redshift'])) > 1e-3]

    return sig_mean(nz)


def return_error_in_mean(all_thresholds, model_correlation=0.283):
    '''
    Equation 7.51 in https://www.deeplearningbook.org/contents/regularization.html
    '''
    nmodels = all_thresholds.shape[0]

    sigma = np.std(all_thresholds, axis=0)

    return sigma * np.sqrt((1 - model_correlation) / nmodels + model_correlation)


if __name__ == "__main__":

    main(
        search_path="data/100/convergence/*.pkl",
        sample_data=False,
        add_ncomps=True
    )

    main(
        search_path="data/100/convergence/*.pkl",
        sample_data=True,
        data_dir='data/100/a2744'
    )
