color_ds = \
{'-1.00':'grey',
 '0.00':'b',
 '0.01':'g',
 '0.05':'r',
 '0.07':'k',
 '0.10':'y',
 '0.20':'c',
 '0.30':'purple',
 '1.00':'grey'
}

from scipy.optimize import curve_fit
from matplotlib.path import Path
import matplotlib.patches as patches
from RRGtools import run_match
from scipy.stats import chi, norm, cauchy
from astropy.io import fits
from astropy import units
from astropy.cosmology import Planck18

from getColorFromRange import colourFromRange
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from torchvision import transforms
from dataset import prepare_dataloaders, get_cross_section_from_filename, sigma_critical, apply_intrinsic_ell, rescale_lens_source_configuration
from model import create_model
from train import evaluate, train_epoch
from utils import parse_args, set_seed, setup_wandb, calculate_class_weights
from add_shear_to_data import *


import numpy as np
from matplotlib import pyplot as plt
from netloader.network import Network
from torchvision.models import (
    ResNet18_Weights, resnet18, ResNet34_Weights, resnet34,
    MobileNet_V3_Small_Weights, mobilenet_v3_small,
    SqueezeNet1_1_Weights, squeezenet1_1,
)
from tqdm import tqdm
import glob
import pickle as pkl
from scipy.stats import ks_2samp, norm, lognorm
import wandb
from matplotlib.gridspec import GridSpec
import lenspack
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection

from scipy.special import lambertw

from refactor_parameter_space import *
from matplotlib.ticker import FuncFormatter

from pyRRG.calc_shear import calc_shear
from RRGtools import run_match

zs = {
    'f115w':1.36,
    'f150w':1.56,
    'concat':1.46
}

colour_scheme = {
    'darkskies':"#E69F00",  # orange
    'bahamas': "#56B4E9",  # sky blue
    'tng':"#009E73",  # bluish green
    'flamingo':"#D55E00",  # yellow
}

vec = np.arange(100)-49.5
xg, yg = np.meshgrid(vec*20.,vec*20.)
rgrid = np.sqrt(xg**2 + yg **2)


def remove_cluster_members( cat_raw, photoz):
    
    combined = run_match( cat_raw, photoz)

    all_cat = fits.open(cat_raw)[1].data
    

    cluster_numbers = combined[1].data['NUMBER_1'][ combined[1].data['z'] < 0.4]
    
    all_numbers = all_cat['NUMBER']
    
    keep_numbers = np.array([ np.where(i==all_numbers)[0][0] for i in all_numbers if i not in cluster_numbers ])
    
    nremove = cluster_numbers.shape[0]
    print(f"REMOVING {nremove} GALAXIES")
    

    


    return all_cat[ keep_numbers ]
    
    
    
def ta( ifilter, cuts=None, data_dir='data/' ):
    
    # Cuts already taken during the WL process - verified.
    cuts = {
        'size_cut':[2,200],
        'signal_noise_cut':0,
        'stat_type':'median',
        'mag_cut':[0,30],
        'verbose':False
    }     
    
    if ifilter == 'concat':

        cat_a_name =     f"{data_dir}/a2744_f115w_filtered.fits"
        cat_b_name =     f"{data_dir}/a2744_f150w_filtered.fits"
        obs_data = combine_catalogues( cat_a_name, cat_b_name, identifier='NUMBER' )
        fits.writeto(  f"{data_dir}/a2744_concat_filtered.fits", obs_data, overwrite=True) 
        
    else:
        shear_cat =  f"{data_dir}/abell2744clu-grizli-v5.4-{ifilter}-clear_drc_sci_clean.shears"
        obs_data = fits.open(shear_cat)[1].data

        calc_shear(
            obs_data, 
            f"{data_dir}/a2744_{ifilter}_filtered.fits", 
            **cuts
        )

        obs_data = fits.open(
            f"{data_dir}/a2744_{ifilter}_filtered.fits"
        )[1].data
        
    return obs_data
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
        concat_e_2,
        cat_a['ra'][ extra_cat_a ],
        cat_b['ra'][ extra_cat_b ]
    ])  
        
    final_dec = np.concatenate([
        concat_e_2,
        cat_a['dec'][ extra_cat_a ],
        cat_b['dec'][ extra_cat_b ]
    ]) 
    
    if 'z_1' in list(matched_cat.dtype.names):
        final_z = np.concatenate([
            z,
            cat_a['z'][ extra_cat_a ],
            cat_b['z'][ extra_cat_b ]
        ]) 
        
        print(f"{matched_cat.shape[0]} galaxies in common")
        print(f"{extra_cat_a.shape[0]} extra galaxies in cat_a")
        print(f"{extra_cat_b.shape[0]} extra galaxies in cat_b")
        print(f"{final_g2.shape[0]} total galaxies")




        obs_data = {'x':final_x, 
                'y':final_y, 
                'gamma1':final_g1, 
                'gamma2':final_g2,
                'e1':final_e1, 
                'e2':final_e2,
                'RA':final_ra,
                'DEC':final_dec,
                    'z':final_z,
                'NUMBER':np.arange(final_dec.shape[0])+1

               }

    else:
        
        print(f"{matched_cat.shape[0]} galaxies in common")
        print(f"{extra_cat_a.shape[0]} extra galaxies in cat_a")
        print(f"{extra_cat_b.shape[0]} extra galaxies in cat_b")
        print(f"{final_g2.shape[0]} total galaxies")




        obs_data = {'x':final_x, 
                'y':final_y, 
                'gamma1':final_g1, 
                'gamma2':final_g2,
                'e1':final_e1, 
                'e2':final_e2,
                'RA':final_ra,
                'DEC':final_dec,
                'NUMBER':np.arange(final_dec.shape[0])+1
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
    
    
    
    
def get_probabilities( 
        target_domain,
        list_of_models,
        args=None,
        data_loaders=None,
        test_set='target_test',
        quiet=False
    ):
    if args is None:
        args = get_temp_args()
        
    args.target_domain=target_domain
 
    all_models = []
    for imodel in list_of_models:
        args.checkpoint = imodel
        all_models.append(create_model(args))
        
    if not quiet:
        print(f"Found {len(all_models)} models")

    device='mps'
    all_cross_sections=[]
    all_binary_labels=[]
    probabilities=[]
    very_good = []
    very_good_idx = []
    very_bad = []
    very_bad_idx = []
    limit = 0.2
    indexes = []



    ###

    with torch.no_grad():
        for imodel in tqdm(all_models,disable=quiet):
            this_probabilities = []
            this_cross = []
            args.seed = imodel.args.seed
            
            if data_loaders is None:
                dataloaders = prepare_dataloaders(args)
            else:
                dataloaders = data_loaders
            
            all_data = [ [ j[0] for j in i ] for i in dataloaders[test_set]  ]

            for idx, batch_data in enumerate(dataloaders[test_set][0]):

                data, cross_sections, binary_labels, file_idx, image_idx = batch_data


                data = [ i[idx] for i in all_data ]

            


                outputs_dict = imodel(data)

                
                binary_labels = binary_labels.to(device)
                targets = binary_labels
                this_cross.append(cross_sections.cpu())
                all_binary_labels.append(binary_labels.cpu())
                prob = torch.softmax( outputs_dict['classification'], dim=1 )

                this_probabilities.append(prob)
                
                indexes.append( image_idx )
                very_good.append(data[0][np.where( (prob[:,0] < limit) & (cross_sections==0.05) )[0]])
                very_bad.append(data[0][np.where( (prob[:,0] > 1-limit) & (cross_sections==0.05) )[0]])

                very_good_idx.append( image_idx[np.where( (prob[:,0] < limit) & (cross_sections==0.05) )[0]])
                very_bad_idx.append( image_idx[np.where( (prob[:,0] > 1-limit) & (cross_sections==0.05) )[0]])

            probabilities.append(torch.cat(this_probabilities))
            all_cross_sections.append(torch.cat(this_cross))
        
    return {
        "all_cross_sections" : torch.stack( all_cross_sections ),
        "all_binary_labels" : torch.cat( all_binary_labels ),
        "probabilities" : torch.stack(probabilities),
        "indexes" :np.concatenate(indexes),
        "very_good" : np.concatenate(very_good),
        "very_bad" : np.concatenate(very_bad),
        "very_good_idx" : np.concatenate(very_good_idx),
        "very_bad_idx" : np.concatenate( very_bad_idx ),
        "data_loaders":dataloaders
    }



def plot_predictions( 
        results, 
        calibrate_mean=None, 
        samplesizes =  [1, 10, 50,100, 200], 
        return_calibrate_mean=False, 
        function=np.mean, 
        gs=None, return_gs=False,
        hist_kwargs=None, line_kwargs=None):
    
    line_kwargs = line_kwargs or {}
    hist_kwargs = hist_kwargs or {}  

    unique_cross =  torch.unique(results['all_cross_sections'])

    if not return_calibrate_mean:
        if gs is None:
            fig = plt.figure(figsize=(len(unique_cross)*4.,3.))
            gs = GridSpec(1, len(unique_cross)+12)
        final_ax = plt.subplot(gs[-5:])
        single_ax = plt.subplot(gs[-11:-6])

        
    with torch.no_grad():
        for iax, icross in enumerate(unique_cross):
            
            nsigma_not_cdm = []
            var_nsigma_not_cdm = []

            prob =  results['probabilities'][ icross == results['all_cross_sections'], : ]

            if icross == 0:
                if calibrate_mean is None:
                    calibrate_mean = function(prob[:,0].detach().numpy())
                    print(f"Calibrated mean is {calibrate_mean}")
                if return_calibrate_mean:
                    return calibrate_mean
            ax = plt.subplot( gs[iax] )         

            hist_kwargs['color'] = color_ds[ "%0.2f" % icross ]
            
            ax.hist(prob[:,0] , bins=np.linspace(0,1,20), **hist_kwargs)
   
            for isamplesize in samplesizes:
                isamplesize = np.min([isamplesize, prob.shape[0]])
                nsigma = []
                if isamplesize == 1:
                    
                    nsigma = (calibrate_mean - prob[:,0].detach().numpy())/np.std(prob[:,0].detach().numpy())
                    
                else:
                    for imonte in range(100):
                        select_these =  np.random.choice( prob[:,0], isamplesize, replace=False )

                        mean = function(select_these)


                        error = np.std(select_these)/np.sqrt(select_these.shape[0])

                        nsigma.append(
                            (calibrate_mean -  mean) / error
                        )
                    
                nsigma_not_cdm.append(np.mean(nsigma))
                var_nsigma_not_cdm.append(np.std(nsigma))

            ylim = ax.get_ylim()[1]
            ax.plot([
                torch.mean(prob[:,0]),
                torch.mean(prob[:,0])
            ],[0,2*ax.get_ylim()[1]],'k--')
            print(f"{icross}: {torch.mean(prob[:,0])}")
            ax.plot([
                    0.5, 0.5
            ],[0,2*ax.get_ylim()[1]],'k:')
            
            ax.set_ylim((0,ylim))
            ax.set_yticklabels([])
            
            if not 'fmt' in line_kwargs.keys():
                line_kwargs['fmt'] = 'o-'
            line_kwargs['color'] = color_ds[ "%0.2f" % icross ]
                
            final_ax.errorbar( 
                samplesizes,
                nsigma_not_cdm, 
                yerr=var_nsigma_not_cdm,
                capsize=2,
                **line_kwargs)
            single_ax.errorbar( 
                np.log10(1+icross), nsigma_not_cdm[0], 
                var_nsigma_not_cdm[0],
                fmt='o-', capsize=2
            )
    final_ax.plot([0,210],[0,0],'--', color='grey')
    final_ax.plot([0,210],[1,1],'--', color='grey')
    final_ax.plot([0,210],[3,3],'--', color='grey')
    final_ax.plot([0,210],[5,5],'--', color='grey')
    final_ax.set_xlim(-10.,210)
    
    if return_gs:
        return gs

    
    
def get_temp_args(noisey=True):
    if noisey:
        return args
    else:
        temp_args = args
        args.intrinsic_ell = 0.
        return args
    
class args:
    pretrained=True
    in_channels=2
    adaptation='cdan'
    use_mixup=False
    mixup_strategy=False
    weighting_scheme='inverse_frequency'
    aug_h_flip_prob=0.5
    aug_v_flip_prob=0.5
    aug_rotation_degrees=360
    aug_rotation_prob=1.
    image_size=100
    aug_crop_scale_min=0.9
    aug_crop_scale_max=1.1
    data_dir='../data/'
    aug_crop_prob=0.5
    use_log_transform=False
    use_normalization=False
    train_split=0.8
    batch_size=32
    num_workers=0
    cnn_base_channels=32
    mass_index=0
    dtypes=['image']
    meta_names=[]
    device="MPS"
    num_avgpool_head=1
    domain_discriminator=None
    num_avgpool_head=1
    source_domain='a2744'
    target_domain='darkskies_obs'
    model="squeezenet1_1"
    verbose=False
    shape_measurement_bias = {
        'e1':{'c':0, 'm':0},
        'e2':{'c':0, 'm':0}
    }
    seed=10
    zl=0.305
    zs=1.36
    apply_intrinsic_ell=1
    jwst_filter='concat'
    med_norm = -1
    ignore_dataset=['']
    unbalance=False
    log_mass_cut=0
    
    print(f"Source redshift:{zs}")

          
    
def get_threshold_for_cross( results_list, dataset=None, function=np.mean,  
                            mass_cut=None, quiet=True, integrated_mass=False, 
                            mass_weights=None, h=0.7):
    '''
    For the output of 
    get_probabilities
    get a dict of the mean values as returned by the ML
    
    '''
    if (mass_cut is not None) & (dataset is None):
        raise ValueError("I need a dataset if you want a masscut")
  
    if not isinstance( results_list, list):
        results_list = [results_list]
     

    all_thresholds = []
    all_threshold_err = []
    all_probs = []
    all_indexes = []
    
    if mass_cut is not None:
        mass_data = {}
        
        if integrated_mass:
            zl = 0.305
            zs = 1.65
            
            critical_kappa = lenspack.utils.sigma_critical(zl, zs, Planck18).to(units.Msun/units.kpc/units.kpc)
            
            
    for ir, results in enumerate(results_list):
        unique_cross =  torch.unique(results['all_cross_sections'])
        thresholds = []
        threshold_err = []
        these_probs = []
        these_indexes = []
        

        with torch.no_grad():
            for iax, icross in enumerate(unique_cross):

                nsigma_not_cdm = []
                var_nsigma_not_cdm = []
                if mass_cut is not None:
                    icross_lab = str(icross)
                    if icross_lab not in list(mass_data.keys()):
                        if icross != 0:
                            pklfile = glob(f"../data/shear/{dataset}_{icross:0.1g}*.pkl")[0]

                        elif dataset in ['flamingo','tng']:
                            pklfile = f"../data/shear/{dataset}.pkl"
                        else:
                            pklfile = f"../data/shear/{dataset}_cdm.pkl"

                        meta, data = pkl.load(open(pklfile,'rb'))

                        if integrated_mass:
                            pixelsize=20e-2
                            kappa  = data[:,2]
                            mass_data[icross_lab] = np.log10((np.sum(kappa[:,rgrid<750.],axis=-1)*critical_kappa*(20*units.kpc)**2).value)

                        else:
                            mass_data[icross_lab] = meta['mass']
                        
                    
                    mass = mass_data[icross_lab][ results['indexes'][(icross == results['all_cross_sections'][0])]]
                    
                    mass_indexes = (mass > mass_cut[0]) & (mass < mass_cut[1])

                else:
                    mass_indexes = np.ones(results['indexes'][(icross == results['all_cross_sections'][0])].shape[0])==1
                
                prob =  results['probabilities'][ icross == results['all_cross_sections'], : ][mass_indexes,:]
                
        
                probs  = prob[:,0].detach().numpy()

                
                if mass_weights is None:
                    this_threshold = function(probs)
                else:
                    
                    weights = mass_weights['y'][ np.digitize(mass[mass_indexes], mass_weights['x'])]
                    this_threshold = np.nansum(weights*probs)/np.sum(weights)
                  
                    
                
                    
                these_probs.append(prob)
                thresholds.append(this_threshold)
                threshold_err.append(np.nanquantile(prob[:,0].detach().numpy(), [0.16,0.84])/np.sqrt(prob[:,0].shape[0]))
                these_indexes.append(   results['indexes'][ icross == results['all_cross_sections'][0] ][mass_indexes])
                
        all_thresholds.append(thresholds)
        all_threshold_err.append(threshold_err)
        all_probs.append(these_probs)
        all_indexes.append(these_indexes)
        
    all_thresholds = np.mean( np.array(all_thresholds), axis=0)
    
    all_threshold_err = np.mean(all_threshold_err, axis=0)/np.sqrt(len(results_list))

    
    all_probs = [ np.concatenate([ all_probs[j][i] for j in range(len(results_list))]) for i in range(len(unique_cross))]
                   
    all_indexes = [ np.concatenate([ all_indexes[j][i] for j in range(len(results_list))]) for i in range(len(unique_cross))]

    return {'thresholds':all_thresholds, 'threshold_err':all_threshold_err, 
            'cross_sections':unique_cross.detach().numpy(),
            'probabilities':all_probs,
            'indexes':all_indexes
            }


def ra_dec_to_simulation_image_pos( 
    obs_data, 
    fov_sim=2e3*units.kpc, 
    jwst_pixel_size = 0.02*units.arcsecond, 
    zl = 0.305,
    pixel_size_kpc = 20.*units.kpc,
    image_size = 100
    ):
    


    conversion = (1.*units.radian.to(units.arcsecond)/Planck18.angular_diameter_distance(zl).to(units.kpc))
    
    pixel_size_arc = conversion*pixel_size_kpc

    fov_sim_arcsec = fov_sim * conversion / 2.
    

    ra_0 = np.median(obs_data['x']) 
    dec_0 = np.median(obs_data['y']) 
    delta_ra = (obs_data['x'] - ra_0)*jwst_pixel_size 
    delta_dec = (obs_data['y'] - dec_0)*jwst_pixel_size

    #rescale them to be between 0 and 2 Mpc
    delta_ra /= fov_sim_arcsec / (image_size//2)
    delta_dec /= fov_sim_arcsec / (image_size//2)
    
    return delta_ra.value, delta_dec.value

def get_kappa( all_cats, smooth=1, extent=None, correct_for_ngal=False):
#set the resolution of the map 

    npix = 100


    e1_radec, e2_radec = lenspack.utils.bin2d( 
        all_cats['x'], all_cats['y'], 
        v=(all_cats['gamma1'], all_cats['gamma2']),
        npix=npix, extent=extent
    )
    
    ngal = lenspack.utils.bin2d( 
        all_cats['x'], all_cats['y'], 
        v=None,
        npix=npix, extent=extent
    )
    if correct_for_ngal:
        e1_radec *= ngal/np.median(ngal[ngal!=0])
        e2_radec *= ngal/np.median(ngal[ngal!=0])
   
    
    ke_radec, kb_radec = lenspack.image.inversion.ks93( e1_radec, e2_radec)

    kappa_e_map = gaussian_filter(ke_radec,smooth)
    kappa_b_map = gaussian_filter(kb_radec,smooth)
    return kappa_e_map, kappa_b_map


def get_src_target_thresholds( jwst_filter, list_of_models=None ):
    
    if list_of_models is None:
        list_of_models = np.sort(glob(
            f"../models/{jwst_filter}/cdan_adapt_pre_squeezenet1_aw_4.0_pad_rot_1_shear_avgpool_gauss_seed_*_final_ft_70_best_finetuned.pth"
        ))

    if len(list_of_models) == 0:
        raise ValueError("No models found for this jwst_filter")
        
    filter_stats = {
        'f150w':{'zl':1.56, 'intrinsic_ell':0.25},
        'f115w':{'zl':1.36, 'intrinsic_ell':0.25},
        'concat':{'zl':1.46, 'intrinsic_ell':0.25}
    }
    
    if jwst_filter not in list(filter_stats.keys()):
        raise ValueError("No stats for this jwst_filter")
        
    args.intrinsic_ell = 0.25
    args.jwst_filter = jwst_filter
    
    test_domain = "darkskies_obs"
    tgt_results = get_probabilities( 
                test_domain,
                list_of_models,
                args
    )
    tgt = get_threshold_for_cross(tgt_results, function=np.mean)

    test_domain = "bahamas_obs"
    src_results = get_probabilities( 
                test_domain,
                list_of_models,
                args
    )

    
    src = get_threshold_for_cross(src_results, function=np.mean)

    return {'src_results':src_results, 'src_thresholds':src,
            'tgt_results':tgt_results, 'tgt_thresholds':tgt
           }

def get_nsigma( value, dist, statistic='median'):
    xcum = np.sort( dist)
    this_cumsum = np.cumsum(xcum)/np.sum(xcum)

    if statistic == 'maxlike':
        y, x = np.histogram(dist,50)
        xc = (x[:-1]+x[1:])/2.
        median_val = xc[np.argmax(y)]
    elif statistic == 'median':
        median_val = np.median(dist)
    elif statistic == 'mean':
        median_val = np.mean(dist)
    else:
        raise ValueError("Stat not recofnised")
        
    #how far from the median is the value in units of width in that direction    
    one_sigma_low = median_val - xcum[ np.argmin(np.abs( 0.16 - this_cumsum )) ]
    one_sigma_high = xcum[ np.argmin(np.abs( 0.84 - this_cumsum )) ] - median_val
    nsigma = []
    for ivalue in np.atleast_1d(value):
        if ivalue >  median_val:
            nsigma.append((ivalue -  median_val)/one_sigma_high)
        else:
            nsigma.append((median_val - ivalue )/one_sigma_low)
    return np.stack(nsigma)
            
def get_direct_prob( value, dist):
    xcum = np.sort( dist)
    this_cumsum = np.cumsum(xcum)/np.sum(xcum)

    median_val = np.median(dist)
    #how far from the median is the value in units of width in that direction    
    one_sigma_low = median_val - xcum[ np.argmin(np.abs( 0.16 - this_cumsum )) ]
    one_sigma_high = xcum[ np.argmin(np.abs( 0.84 - this_cumsum )) ] - median_val
    nsigma = []
    for ivalue in np.atleast_1d(value):
        if ivalue < median_val:
            nsigma.append(this_cumsum[ np.argmin( np.abs(xcum - ivalue)) ] / 0.5)
        else:
            nsigma.append((1-this_cumsum[ np.argmin( np.abs(xcum - ivalue)) ] )/0.5)
 
        
    return np.stack(nsigma)
            
def plot_observations( filename, ifilter, 
                      ax=None, correction=2.1, 
                      error_index=0.38,
                      noise=False, 
                      uncertainty=[68],
                      legend=False,
                      plot_args = {}, fill_args={},
                     plotpdf=False):
    if 'color' not in plot_args.keys():
        plot_args['color'] = 'black'    
    if isinstance( filename, list):
  
        if noise:
            data = np.concat([ pkl.load(open(i,"rb"))[2][ifilter] for i in filename ]).flatten()
        else:
            data = np.concat([ pkl.load(open(i,"rb"))[1][ifilter] for i in filename ])
            
            
        nmodels = data.shape[0]/len(filename)
    else:
        models, probabilities, probabilities_noise = pkl.load(open(filename,'rb'))
        
        if noise:
            data = probabilities_noise[ifilter].flatten()
        else:
            data = probabilities[ifilter]
            
        nmodels = data.shape[0]
    
        
    if ax is None:
        ax = plt.gca()
    
    means = 1-np.mean(data)
    error = np.std(data)/nmodels**(error_index)*correction
    
    if plotpdf:
        
        xpdf = np.linspace(0.45,0.7, 1000)

        ypdf = norm.pdf( xpdf, *(means, error))

        ax.plot(  xpdf, ypdf,
                             color=plot_args['color'] )

        ax.fill_between( xpdf,
                            ypdf,
                            np.zeros(1000),
                            **fill_args)  

        return
    
    ax.plot( np.logspace(-3,1,100), np.zeros(100)+means, **plot_args )
    
    for iunc in uncertainty:
        
        if iunc == 68:
            nsigma=1
        if iunc == 95:
            nsigma=2.
        if iunc == 99:
            nsigma=3.
        
        err = [error*nsigma + means, means - error*nsigma ]
        print(err)
        
        if ifilter == 'concat':
            if not noise:
                ax.text( 0.002,means,"A2744 UNCOVERS DATA", ha='left',va='bottom', fontsize=12)
                ax.text( 0.002,err[1],f"{iunc}\% Uncertainty", ha='left',va='bottom', fontsize=12)

        if not noise:
            ax.fill_between(  np.logspace(-3,1,100), np.zeros(100)+err[0],  np.zeros(100)+err[1], **fill_args)
            ax.plot( np.logspace(-3,1,100), np.zeros(100)+err[0], '--', color=plot_args['color'])
            ax.plot( np.logspace(-3,1,100), np.zeros(100)+err[1], '--', color=plot_args['color'])
   
    
    if legend:
        ax.legend()
        
def get_latent_space( model_list,
                    quiet=False):
    
    if not isinstance(model_list, list):
        model_list = [model_list]
        
    all_cross =[]
    latent_spaces = []
    
    all_models = []
    for imodel in model_list:
        args.checkpoint = imodel
        all_models.append(create_model(args))
        
    args.source_domain = 'bahamas_obs'
    args.target_domain = 'darkskies_obs'
    
    obs_meta, obs_data = pkl.load(open(f"../data/a2744/obs_data_concat.pkl","rb"))

    this_latent = []
    this_cross = []
            
    for imx, imodel in tqdm(enumerate(all_models), disable=quiet):

        args.seed = imodel.args.seed

        this_latent.append(imodel.backbone(torch.tensor(obs_data[0][None,:,:,:],dtype=torch.float32))[:,:,0,0])
        this_cross.append(torch.tensor([-1]))  
        
    
              
    latent_spaces.append(torch.cat(this_latent))
    all_cross.append(torch.cat(this_cross))      
    
    with torch.no_grad():
        for imx, imodel in tqdm(enumerate(all_models), disable=quiet):
            this_latent = []
            this_cross = []
            args.seed = imodel.args.seed
        
            dataloaders = prepare_dataloaders(args)

            for test_set in ['source_val','target_test']:

                all_data = [ [ j[0] for j in i ] for i in dataloaders[test_set]  ]

                for idx, batch_data in enumerate(dataloaders[test_set][0]):

                    data, cross_sections, binary_labels, file_idx, image_idx = batch_data


                    this_latent.append(imodel.backbone(data)[:,:,0,0])
                    this_cross.append(cross_sections.cpu())




            latent_spaces.append(torch.cat(this_latent))

            all_cross.append(torch.cat(this_cross))

    return {
        "all_cross_sections" : all_cross ,
        "latent_space" : latent_spaces 
    }


def get_mass_cut( ifilter, zl=0.305, zs=1.6, thresh = 0., nsigma=2, study='harvey' ):
    
    obs = {'harvey': {'core':16, 'nw':10.8, 'n':6.5},
           'jauzac':{'core':27.7, 'nw':18., 'n':8.6}
          }
    err = {'harvey': {'core':[0.9,0.6],'nw':[1.0,0.3],'n':[0.9,0.7]},
           'jauzac': {'core':[0.1,0.1],'nw':[1.0,1.0],'n':[2.2,2.2]}}
  
    

    
    choice = obs[study]
    choice_err = err[study]
    
    est = np.sum( [choice[i] for i in choice.keys() ])
    err_lo = np.sqrt( np.sum( [ (choice_err[i][0]/choice[i])**2 for i in choice.keys() ]))*est
    err_hi = np.sqrt( np.sum( [ (choice_err[i][1]/choice[i])**2 for i in choice.keys() ]))*est

    
    return np.log10(est*1e13),[np.log10((est-nsigma*err_lo)*1e13),np.log10((est+nsigma*err_hi)*1e13)]
        
    critical_density = sigma_critical(zl, zs, Planck18).to(units.Msun/units.kpc/units.kpc)

    obs_meta, obs_data = pkl.load(open(f"../data/a2744/obs_data_{ifilter}.pkl","rb"))
    ke, kb = lenspack.image.inversion.ks93(obs_data[0][0], obs_data[0][1])

    
    positive_mass = np.sum(gaussian_filter(ke[ke/np.std(kb)>thresh],2) * critical_density * (20*units.kpc)**2 )
    err_mass_per_pixel = np.std(gaussian_filter(kb[ke/np.std(kb)>thresh],2) * critical_density * (20*units.kpc)**2 ).value




    err_mass = len(kb[ke/np.std(kb)>thresh])*err_mass_per_pixel
    mass_cut = [ 
        np.log10(positive_mass.value - err_mass/2.),
        np.log10(positive_mass.value + err_mass/2.)
    ]
    
    return np.log10(positive_mass.value), mass_cut


def curly_brace(ax, x1, x2, y, height, upward=True):
    """
    Draw a curly brace between x1 and x2 at height y.
    """
    mid = (x1 + x2) / 2
    sign = 1 if upward else -1

    verts = [
        (x1, y),
        (x1, y + sign*height/2),
        (mid, y + sign*height/2),
        (mid, y + sign*height),
        (mid, y + sign*height/2),
        (x2, y + sign*height/2),
        (x2, y)
    ]

    codes = [
        Path.MOVETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE3,
    ]

    path = Path(verts, codes)
    patch = patches.PathPatch(path, fill=False, lw=2)
    ax.add_patch(patch)
