#!/usr/bin/env python

from lenspack.image.inversion import ks93inv, ks93
from lenspack.utils import sigma_critical, bin2d
from glob import glob
import pickle as pkl
import numpy as np
from astropy import units
from astropy.cosmology import Planck18
from astropy.io import fits
from tqdm import tqdm
from scipy.interpolate import interp2d
from pyRRG.calc_shear import calc_shear
from RRGtools import run_match
import os
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator
import numpy.lib.recfunctions as rfn

from model import create_model

def get_boxsize(set_name, return_units=units.pc, h=0.7):
    if ('bahamas' in set_name) or ('tng' in set_name) or ('flamingo' in set_name):
        boxsize= 10./h*units.Mpc
    else:
        boxsize =  400/h*units.Mpc
    return boxsize.to(return_units)
    
def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def main( 
    search_path="data/100/convergence/*.pkl", 
    filter_list = ['concat'],
    thresh_k = 0.9, 
    h=0.7, 
    sample_data=True,
    data_dir="data/100/a2744",
    reduce_shear=True,
    zl = 0.305,
    add_ncomps=True
    ):
    
    
    
     
    ##### Some definitions ####
    
    pixel_size_kpc = 20.*units.kpc
    conversion = (1.*units.radian.to(units.arcsecond)/Planck18.angular_diameter_distance(zl).to(units.kpc))
    pixel_size_arc = conversion*pixel_size_kpc
    jwst_pixel_size = 0.02*units.arcsecond
    fov_sim = 2e3*units.kpc * conversion / 2.
    image_size = 100
    
    ######

    
    all_data_sets = glob(search_path)
 
    ### This is for the base_models
    if not sample_data:
        for idx in tqdm(range(len(all_data_sets))):
            
            idata_set = all_data_sets[idx]
            meta, data = pkl.load( open( idata_set, "rb"))
            new_data_set = idata_set.replace("convergence","shear")
            new_data_path = os.path.dirname(new_data_set)      
            
            if add_ncomps:
                ncomponents = get_num_merging_components(
                    idata_set
                )
                meta['ncomponents'] = ncomponents
            
            if ('darkskies' in idata_set) | ('flamingo' in idata_set):
                meta['norms'] /= 4.*h
                
                
            if 'redshift' not in meta.keys():
                meta['redshift'] = np.full(data.shape[0], 0.250, dtype=np.float64)

                
            e1, e2, kappa = data_to_shear( 
                    data[:,0], meta['norms'][:,0], 
                    meta['redshift']*0.+0.305, 
                    get_boxsize(idata_set),
                    zs=get_source_redshift('concat',data_dir=data_dir),
                    zl=zl, **{'ngal_per_sq_arcmin':200.},
                    reduce_shear=reduce_shear,
            )
       
            e1e2 = np.append(e1[:,None,:,:],e2[:,None,:,:], axis=1)

            data[:,0,:,:] = kappa
            add_e1e2 = np.append(e1e2, data, axis=1)

            if not os.path.isdir( new_data_path ):
                os.system(f"mkdir -p {new_data_path}")
                
            pkl.dump( [meta, add_e1e2], open(new_data_set, "wb"))
    
        return
    
    ### This is for the final masked models
    for ifx, ifilter in enumerate(filter_list):
        
        obs_data = get_obs_data( 
            ifilter, data_dir=data_dir,
            photoz=True
        )
   
        ra_0 = np.median(obs_data['x']) 
        dec_0 = np.median(obs_data['y']) 
        delta_ra = (obs_data['x'] - ra_0)*jwst_pixel_size 
        delta_dec = (obs_data['y'] - dec_0)*jwst_pixel_size

        #rescale them to be between 0 and 2 Mpc
        max_val = fov_sim 

        delta_ra /= fov_sim / 50.
        delta_dec /= fov_sim / 50.

       
    
        obs_data['x'] = delta_ra
        obs_data['y'] = delta_dec

        npix = image_size
        ra_range = delta_ra.max() - delta_ra.min()
        dec_range = delta_dec.max() - delta_dec.min()

        xc = 0.
        yc = 0.

        for idata_set in tqdm(all_data_sets):
            
            new_file_name = idata_set.replace('convergence',f'obs/{ifilter}')
            new_data_path = os.path.dirname(new_file_name)   
            if not os.path.isdir( new_data_path ):
                os.system(f"mkdir -p {new_data_path}")
                

            meta, data = pkl.load( open( idata_set, "rb"))
            if ('darkskies' in idata_set) | ('flamingo' in idata_set):
                meta['norms'] /= 4.*h
                
            if 'redshift' not in meta.keys():
                meta['redshift'] = np.full(data.shape[0], 0.305, dtype=np.float64)

            #Interpolate the non reduced shear then reduce it later.
            e1_ideal, e2_ideal, kappa = data_to_shear( 
                    data[:,0], meta['norms'][:,0], 
                    meta['redshift'], 
                    get_boxsize(idata_set),
                    zs=get_source_redshift('concat'), 
                    zl=zl, **{'ngal_per_sq_arcmin':200.},
                    reduce_shear=False,
            )


            e1_obs = []
            e2_obs = []
            remove = []
            for icluster in range(data.shape[0]):

                fill_value = np.median(e1_ideal[icluster][ e1_ideal[icluster] !=0])

                interp_e1 = RegularGridInterpolator(
                    (np.arange(100)-50.+0.5,np.arange(0,100)-50.+0.5), 
                    e1_ideal[icluster],fill_value=fill_value,
                    method='cubic', bounds_error=False
                )

                e1 = interp_e1( 
                    (delta_dec, delta_ra), 
                )

                fill_value = np.median( e2_ideal[icluster][ e2_ideal[icluster] !=0])

                interp_e2 = RegularGridInterpolator(
                    (np.arange(100)-50.+0.5,np.arange(0,100)-50.+0.5), 
                    e2_ideal[icluster], fill_value=fill_value,
                    method='cubic', bounds_error=False
                )

                e2 = interp_e2( 
                    (delta_dec, delta_ra ), 
                )

                interp_k = RegularGridInterpolator(
                    (np.arange(100)-50.+0.5,np.arange(0,100)-50.+0.5), 
                    kappa[icluster], fill_value=fill_value,
                    method='cubic', bounds_error=False
                )

                k = interp_k( 
                    (delta_dec, delta_ra ), 
                )       

                #remove very large shear values

                g = np.sqrt( e1**2 + e2**2 )

                cut = (k<thresh_k)
                if reduce_shear:
                    e1_red = e1 / ( 1. - k)
                    e2_red = e2 / ( 1. - k)
                
                e1_red = e1_red[ cut ]
                e2_red = e2_red[ cut ]
                ra = delta_ra.value[ cut ] 
                dec = delta_dec.value[ cut ] 
                remove.append((delta_ra.shape[0] - ra.shape[0])/delta_ra.shape[0]*100.)

                e1_radec, e2_radec = bin2d( 
                   ra, dec, 
                    v=(e1_red, e2_red),
                    npix=npix,
                    extent=[
                        -50,50,-50,50
                    ]
                )


                e1_obs.append(e1_radec)
                e2_obs.append(e2_radec)

            new_data = np.moveaxis(np.array([e1_obs, e2_obs]), 0, 1)


            pkl.dump([ meta, new_data], open(new_file_name,"wb"))

def get_num_merging_components(
                dataset, 
                mass_ratio_limit=10,
    ):
    comp_name =  f"notebooks/pickles/{dataset.split('/')[-1]}.comps"
    try:
        extracted_components = pkl.load(open(comp_name,"rb"))
    except:
        raise IOError(f"Unable to open {comp_name}")
    ncomponents = []
    for icomp in extracted_components:
        mass_ratio = icomp['FLUX_AUTO'].max() / icomp['FLUX_AUTO']
        
        icomp = icomp[ mass_ratio < mass_ratio_limit ]
        ncomponents.append( icomp.shape[0])
    return np.array(ncomponents)

def data_to_shear( images, norms, redshifts, boxsize, 
                  reduce_shear=True, 
                  kappa_thresh=None, 
                  **kwargs 
                 ):
    '''
    add weak lensing noise 
    '''
    
    if 'zl' in kwargs.keys():
        zl = kwargs['zl']
    else:
        zl = 0.3
        
    if 'zs' in kwargs.keys():
        zs = kwargs['zs']
    else:
        zs=2.
    if 'ngal_per_sq_arcmin' in kwargs.keys():
        ngal_per_sq_arcmin = kwargs['ngal_per_sq_arcmin']
    else:
        ngal_per_sq_arcmin = 100.
    if 'kpc_per_pixel' in kwargs.keys():
        kpc_per_pixel = kwargs['kpc_per_pixel']
    else:
        kpc_per_pixel = 20.
    if 'ell_disp' in kwargs.keys():
        ell_disp = kwargs['ell_disp']
    else:
        ell_disp = 0.26
        
    if 'e1_bias' in kwargs.keys():
        e1_bias = kwargs['e1_bias']     
    if 'e2_bias' in kwargs.keys():
        e2_bias = kwargs['e2_bias']  
        
    if 'interpolate' in kwargs.keys():
        interpolate = kwargs['interpolate']
        
    if 'gals_per_bin' in kwargs.keys():
        gals_per_bin = kwargs['gals_per_bin']
    else:
        gals_per_bin = 2
        
    if 'norm_units' not in  kwargs.keys():
        kwargs['norm_units'] = units.Msun/(units.Mpc*units.Mpc)
        
    ngal_per_sq_arcmin  /= units.arcminute**2
    
    kpc_per_pixel *= units.kpc
    
    
    kpc_per_arcmin = \
        1.*units.arcminute.to(units.radian)*Planck18.angular_diameter_distance(zl).to(units.kpc)/units.arcminute

    arcmin_per_pixel = kpc_per_pixel / kpc_per_arcmin 
    
    sq_arcmin_per_pixel = arcmin_per_pixel * arcmin_per_pixel  
    
    ngalaxies_per_pixel =  ngal_per_sq_arcmin * sq_arcmin_per_pixel
    
    sigma_crit = sigma_critical(zl, zs, Planck18)
    
    peak = norms*kwargs['norm_units']
   
    peak_pc_sq = peak.to(units.Msun/units.pc/units.pc)/sigma_crit
    
    
    all_e1, all_e2, all_converge = [], [], []
    ampl = []

    
    for idx, image in enumerate(images):
        
        #I sum over all z so need to remove the critical density
        critical_density = Planck18.critical_density( redshifts[idx] ).to(units.Msun/units.pc**3)*boxsize*Planck18.Om0
                
        convergence = image*peak_pc_sq[idx] - critical_density/sigma_crit
        
        ampl.append(np.max(convergence))
        
        e1 , e2 = ks93inv(convergence,convergence*0.)
        
        if reduce_shear:
            e1 /= 1.-convergence
            e2 /= 1.-convergence
        
        if kappa_thresh is not None:
            e1[ convergence > kappa_thresh] = 0.
            e2[ convergence > kappa_thresh] = 0.
            
        all_e1.append( e1 )
        all_e2.append( e2 )
        
        all_converge.append(convergence)
        


     
    return np.array(all_e1), np.array(all_e2), np.array(all_converge)




def get_obs_data( 
    ifilter, 
    cuts={}, 
    data_dir='data/100/a2744', 
    photoz=False, 
    remove_members=True,
    redshift_cut=0.305+0.05*3):
        
    # Cuts already taken during the WL process - verified.
    fid_cuts = {
            'size_cut':[2,200],
            'signal_noise_cut':0,
            'stat_type':'median',
            'mag_cut':[0,30],
            'verbose':False
        } 
        
    for this_filter in ['f115w','f150w']:
        if this_filter not in list(cuts.keys()):
            print(f"Missing {this_filter} in cuts dict referring to default")
            cuts[this_filter] = {}
        for ikey in fid_cuts.keys():
            if ikey not in list(cuts[this_filter].keys()):
                cuts[this_filter][ikey] = fid_cuts[ikey]

        shear_cat =  f"{data_dir}/abell2744clu-grizli-v5.4-{this_filter}-clear_drc_sci_clean.shears"
        obs_data = fits.open(shear_cat)[1].data  

        calc_shear(
            obs_data, 
            f"{data_dir}/a2744_{this_filter}_filtered.fits", 
            **cuts[this_filter]
            )      

        if remove_members:
            obs_data = remove_cluster_members( 
                f"{data_dir}/a2744_{this_filter}_filtered.fits", 
                f"{data_dir}/UNCOVER_DR4_SPS_catalog.fits",
                redshift_cut=redshift_cut
            )

        fits.writeto(
             f"{data_dir}/a2744_{this_filter}_filtered.fits", 
            obs_data, overwrite=True
        )
            

 
    
    if ifilter == 'concat':
        cat_a_name =     f"{data_dir}/a2744_f115w_filtered.fits"
        cat_b_name =     f"{data_dir}/a2744_f150w_filtered.fits"
        obs_data = combine_catalogues( cat_a_name, cat_b_name, identifier='NUMBER' )
        concat_name =  f"{data_dir}/a2744_concat_filtered.fits"

        fits.writeto(  concat_name, obs_data, overwrite=True) 

    else:

        obs_data = fits.open(
            f"{data_dir}/a2744_{ifilter}_filtered.fits"
        )[1].data
        
    if photoz:
        photo_z_matched = run_match(
            f"{data_dir}/UNCOVER_DR4_SPS_catalog.fits",
            f"{data_dir}/a2744_{ifilter}_filtered.fits",
            search_rad=0.5
        )[1].data
        
        default_zs = sig_mean( photo_z_matched['z_ml'] )

        print(f"DEFAULT ZS {default_zs}")
        
        redshift = np.full(obs_data.shape[0], default_zs, dtype=np.float64)

        # build lookup table
        z_lookup = dict(zip(photo_z_matched['NUMBER'],
                            photo_z_matched['z_ml']))

        # fill values
        n_missing = 0
        for j, number in enumerate(obs_data['NUMBER']):
            if number in z_lookup:
                if np.isfinite(z_lookup[number] ):
                    
                    redshift[j] = z_lookup[number] 
                else:
                    redshift[j] = default_zs
            
        # append field
        obs_data = rfn.append_fields(
            obs_data,
            names='redshift',
            data=redshift,
            dtypes=np.float64,
            usemask=False,
            asrecarray=True
        )

    return obs_data

def remove_cluster_members( cat_raw, photoz, redshift_cut=0.): #0.305+2*0.06):
    
    combined = run_match( cat_raw, photoz,search_rad=0.5)

    all_cat = fits.open(cat_raw)[1].data
    
    cluster_numbers = combined[1].data['NUMBER'][ combined[1].data['z_ml'] < redshift_cut]
        
    mask = ~np.isin(all_cat['NUMBER'], cluster_numbers)
    
    removed = all_cat[mask]
    
    #print(f"REMOVING {cluster_numbers.shape[0]} GALAXIES")
    return removed
       
def combine_catalogues( cat_a_name, cat_b_name, identifier='NUMBER' ):
    
    cat_a = fits.open(cat_a_name)[1].data
    cat_b = fits.open(cat_b_name)[1].data
   
    matched_cat = run_match(cat_a_name, cat_b_name)[1].data

    extra_cat_a =  np.array([
        i for i in range(cat_a.shape[0]) if cat_a['NUMBER'][i] not in matched_cat['NUMBER_1'] 
    ])
    extra_cat_b =  np.array([
        i for i in range(cat_b.shape[0]) if cat_b['NUMBER'][i] not in matched_cat['NUMBER_2'] 
    ]) 
    

    
    concat_gamma_1 = ( matched_cat['gamma1_1'] +matched_cat['gamma1_2'])/2. 
    concat_gamma_2 = ( matched_cat['gamma2_1'] +matched_cat['gamma2_2'])/2. 
    
    concat_e_1 = ( matched_cat['e1_1'] +matched_cat['e1_2'])/2. 
    concat_e_2 = ( matched_cat['e2_1'] +matched_cat['e2_2'])/2. 
    
    
    
    if 'z_1' in list(matched_cat.dtype.names):
        z = matched_cat['z_1']
    
    
    ra = matched_cat['ra_1']
    dec = matched_cat['dec_1']
    
    final_x = np.concatenate([
        matched_cat['x_1'],
        cat_a['x'][ extra_cat_a ],
        cat_b['x'][ extra_cat_b ]
    ])
    
      
    final_y = np.concatenate([
        matched_cat['y_1'],
        cat_a['y'][ extra_cat_a ],
        cat_b['y'][ extra_cat_b ]
    ])
        
    final_g1 = np.concatenate([
        concat_gamma_1,
        cat_a['gamma1'][ extra_cat_a ],
        cat_b['gamma1'][ extra_cat_b ]
    ])
    
    final_g2 = np.concatenate([
        concat_gamma_2,
        cat_a['gamma2'][ extra_cat_a ],
        cat_b['gamma2'][ extra_cat_b ]
    ])
    
    final_e1 = np.concatenate([
        concat_e_1,
        cat_a['e1'][ extra_cat_a ],
        cat_b['e1'][ extra_cat_b ]
    ])
    
    final_e2 = np.concatenate([
        concat_e_2,
        cat_a['e2'][ extra_cat_a ],
        cat_b['e2'][ extra_cat_b ]
    ])  
    
    final_ra = np.concatenate([
        matched_cat['ra_1'],
        cat_a['ra'][ extra_cat_a ],
        cat_b['ra'][ extra_cat_b ]
    ])  
        
    final_dec = np.concatenate([
        matched_cat['dec_1'],
        cat_a['dec'][ extra_cat_a ],
        cat_b['dec'][ extra_cat_b ]
    ]) 
    
    final_mag = np.concatenate([
        matched_cat['MAG_AUTO_1'],
        cat_a['MAG_AUTO'][ extra_cat_a ],
        cat_b['MAG_AUTO'][ extra_cat_b ]
    ]) 
    final_size = np.concatenate([
        matched_cat['gal_size_1'],
        cat_a['gal_size'][ extra_cat_a ],
        cat_b['gal_size'][ extra_cat_b ]
    ]) 
            
    if 'z_1' in list(matched_cat.dtype.names):
        final_z = np.concatenate([
            z,
            cat_a['z'][ extra_cat_a ],
            cat_b['z'][ extra_cat_b ]
        ]) 
        



        obs_data = {'x':final_x, 
                'y':final_y, 
                'gamma1':final_g1, 
                'gamma2':final_g2,
                'e1':final_e1, 
                'e2':final_e2,
                'RA':final_ra,
                'DEC':final_dec,
                    'z':final_z,
                'NUMBER':np.arange(final_dec.shape[0])+1,
                'MAG':final_mag,
                'SIZE':final_size,
               }

    else:
        



        obs_data = {'x':final_x, 
                'y':final_y, 
                'gamma1':final_g1, 
                'gamma2':final_g2,
                'e1':final_e1, 
                'e2':final_e2,
                'RA':final_ra,
                'DEC':final_dec,
                'NUMBER':np.arange(final_dec.shape[0])+1,
                'MAG':final_mag,
                'SIZE':final_size,
               }   
    new_cols = []
    for ikey in obs_data.keys():
            new_cols.append(
                fits.Column(
                    name=ikey,
                    format=obs_data[ikey].dtype,
                    array=obs_data[ikey]
                ))
    hdu = fits.BinTableHDU.from_columns(new_cols)
        
        
    return hdu.data


def bin_obs_data( in_obs_data, image_size = 100, npix=100 ):
    
    obs_data = in_obs_data.copy()
    
    g = np.sqrt( obs_data['gamma1']**2 + obs_data['gamma2']**2)

    obs_data['gamma1'] = 2.*obs_data['gamma1'] / (1.+g**2)
    obs_data['gamma2'] = 2.*obs_data['gamma2'] / (1.+g**2)

    delta_ra, delta_dec = ra_dec_to_simulation_image_pos( obs_data )


    e1_radec, e2_radec = bin2d( 
                delta_ra, delta_dec, 
                v=(obs_data['gamma1'], 
                   obs_data['gamma2']),
                npix=npix,
                extent=[
                    -image_size//2,image_size//2,-image_size//2,image_size//2
                ]
            )

    ### TO SPEED UP THE AUGMENTATION ####
    edges = np.linspace(-image_size//2,image_size//2,npix+1)
    
    ix = np.digitize(delta_ra, edges) - 1
    iy = np.digitize(delta_dec, edges) - 1
    
    
    valid = (
        (ix >= 0) & (ix < image_size) &
        (iy >= 0) & (iy < image_size)
    )

    ix = ix[valid]
    iy = iy[valid]

    flat_bin = iy * image_size + ix
    
    
    ###################

    ngal = bin2d( 
                delta_ra, delta_dec, 
                v=None,
                npix=npix,
                extent=[
                    -image_size//2,image_size//2,
                    -image_size//2,image_size//2
                ]
            )

    max_val = np.max([
        e1_radec, e2_radec
    ])
    
    min_val = np.min([
        e1_radec, e2_radec
    ])
    
    e1_radec -= min_val 
    e1_radec /= max_val 

    e2_radec -= min_val 
    e2_radec /= max_val 

    return {'e1':e1_radec, 'e2':e2_radec, 'ngal':ngal,
            'delta_ra':delta_ra,'delta_dec':delta_dec,
            'valid':valid, 'flat_bin':flat_bin}


def ra_dec_to_simulation_image_pos( 
    obs_data, 
    fov_sim=2e3*units.kpc, 
    jwst_pixel_size = 0.02*units.arcsecond, 
    zl = 0.305,
    pixel_size_kpc = 20.*units.kpc,
    image_size = 100,
    ra_0=None, dec_0=None,
    ):
    


    conversion = (1.*units.radian.to(units.arcsecond)/Planck18.angular_diameter_distance(zl).to(units.kpc))
    
    pixel_size_arc = conversion*pixel_size_kpc

    fov_sim_arcsec = fov_sim * conversion / 2.
    
    if ra_0 is None:
        ra_0 = np.median(obs_data['x']) 
    if dec_0 is None:
        dec_0 = np.median(obs_data['y']) 
    delta_ra = (obs_data['x'] - ra_0)*jwst_pixel_size 
    delta_dec = (obs_data['y'] - dec_0)*jwst_pixel_size

    #rescale them to be between 0 and 2 Mpc
    delta_ra /= fov_sim_arcsec / (image_size//2)
    delta_dec /= fov_sim_arcsec / (image_size//2)
    
    return delta_ra.value, delta_dec.value


def rebin(arr, Q):
    """
    Rebin an array of shape (N, M, P, P) to (N, M, Q, Q)
    using block-wise averaging.

    Parameters
    ----------
    arr : ndarray
        Input array of shape (N, M, P, P).
    Q : int
        Desired output size of the last two dimensions.

    Returns
    -------
    ndarray
        Rebinned array of shape (N, M, Q, Q).
    """
    N, M, P1, P2 = arr.shape
    if P1 != P2:
        raise ValueError("Last two dimensions must be square.")
    if P1 % Q != 0:
        raise ValueError("P must be divisible by Q.")

    factor = P1 // Q

    coarse =  arr.reshape(
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
        if not 'convergence' in root:
            if len(files) == 0:
                continue
            for ifile in files:
                
                newroot = root.replace("code_nature/data",f"code/data/{int(new_resolution)}")
                
                if not os.path.isdir(f"{newroot}"):
                    os.system(f"mkdir -p {newroot}")    
                
                if 'pkl' in ifile:
                    old_path = f"{root}/{ifile}"
                    meta, data = pkl.load(open(old_path,"rb"))
                    new_data = rebin( data, new_resolution)

                    new_path = f"{newroot}/{ifile}"
                    
                    if new_path == old_path:
                        raise ValueError("Overwriting original data")
                        
                    pkl.dump([meta, new_data], 
                             open(new_path, "wb")
                            )
                    

def get_model_names(
        ifilter='concat', 
        model_dir="../models",
        model='final', 
        exclude_str=None,
        include_str=None,
        load_model_args=None,
        nmodels=None
    ):
    
    list_of_models = glob(
        f"{model_dir}/{ifilter}/*{model}.pth"
    )
    
    if exclude_str is not None:
        list_of_models  = [ i for i in list_of_models if not exclude_str in i ]
    
    if include_str is not None:
        list_of_models  = [ i for i in list_of_models if include_str in i ]
     
    if nmodels is not None:
        list_of_models = list_of_models[:nmodels]
        
    
    assert len(list_of_models) > 0,f"No models found check path ({model_dir})"
        
    #get the seed number index
    names_in_model = np.array(list_of_models[0].split('/')[-1].split('_'))
    seed_index  = np.where(names_in_model == 'seed')[0][0] + 1
    
    if load_model_args is not None:
        all_models = []
        for imodel in list_of_models:
            load_model_args.checkpoint = imodel
            this_model = create_model(load_model_args)

            model_name = f"seed_{this_model.args.seed}"
            #if target_test_accuracy[ifilter][model_name]['target_test_f1'] <0.48:
            #    continue

            all_models.append(this_model)
        
        return np.array(all_models), seed_index
    return np.array(list_of_models), seed_index

def sig_mean( redshift ):
    sig = sigma_critical( 0.305, redshift, Planck18).value
    return np.sum(redshift/sig)/np.sum(1/sig)
    
def get_source_redshift( ifilter, data_dir='data/100/a2744', cuts=None ):
    
    if cuts is None:
        cuts= {'f115w':{'signal_noise_cut':0, 'stat_type':'snr'}, 
       'f150w':{'signal_noise_cut':5, 'stat_type':'snr'}}
    data = get_obs_data( ifilter, photoz=True , data_dir=data_dir, cuts=cuts)

    #remove artifical redshifts 
    nz = data['redshift'][ np.abs( data['redshift'] - np.median(data['redshift'])) > 1e-3]
    
    source_redshift = sig_mean( nz )

    return source_redshift


def return_error_in_mean( all_thresholds, correction=0.18 ):
    
    nmodels = all_thresholds.shape[0]
    
    std = np.std(all_thresholds,axis=0)
    
    return std/all_thresholds.shape[0]**correction

if __name__ == "__main__":
    

    
    main( 
        search_path="data/100/convergence/*.pkl", 
        h=0.7, 
        sample_data=False, 
        add_ncomps=True 
    )
    #Final data, h=0.7 so that the data is correct for final outputs
    main( 
        search_path="data/100/convergence/*.pkl", 
        h=0.7, 
        sample_data=True, 
        data_dir='data/100/a2744' 
    )
