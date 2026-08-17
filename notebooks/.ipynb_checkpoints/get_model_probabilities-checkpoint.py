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

import matplotlib.colors as mcolors
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from matplotlib import gridspec
from scipy.optimize import curve_fit
from matplotlib.path import Path
import matplotlib.patches as patches
from RRGtools import run_match
from scipy.stats import chi, norm, cauchy
from astropy.io import fits
from astropy import units
from astropy.cosmology import Planck18
from scipy.stats import gaussian_kde
from matplotlib.ticker import LinearLocator

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
from add_shear_to_data import get_obs_data, bin_obs_data, ra_dec_to_simulation_image_pos, get_source_redshift
from scipy.stats import binned_statistic_2d,binned_statistic
from scipy.special import logsumexp
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import zoom

colour_scheme = {
    'darkskies':"#E69F00",  # orange
    'bahamas': "#56B4E9",  # sky blue
    'tng':"#009E73",  # bluish green
    'flamingo':"#D55E00",  # yellow
}

vec = np.arange(100)-49.5
xg, yg = np.meshgrid(vec*20.,vec*20.)
rgrid = np.sqrt(xg**2 + yg **2)

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
    apply_intrinsic_ell=1
    jwst_filter='concat'
    med_norm = -1
    ignore_dataset=['']
    unbalance=False
    log_mass_cut=0
    downsample=1
    cluster_member_contamination=0.
    zs=get_source_redshift(jwst_filter)
    default_zl = 0.305
    default_zs = 1.77
    print(f"Source redshift:{zs}")



    
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

                if data[0].shape[0] == 1:
                    if args.verbose:
                        print("Only 1 sample skipping batch")
                    continue
                else:
                    if args.verbose:
                        print(f"Testting {data[0].shape[0]} in batch ")

                try:
                    outputs_dict = imodel(data)
                except:
                    raise ValueError(f"Data issue - what is the shape? {data[0].shape}")
                
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
    

          
    
def get_threshold_for_cross( results_list, dataset=None, function=np.mean,  
                            mass_cut=None, quiet=True, integrated_mass=False, 
                            mass_weights=None, h=0.7, ncomponents=None):
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
            zs = get_source_redshift('concat')
            
            critical_kappa = lenspack.utils.sigma_critical(zl, zs, Planck18).to(units.Msun/units.kpc/units.kpc)
            
    components = {}
        
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
                              
                        meta, data = get_dataset_meta(dataset, icross)

                        if integrated_mass:
                            pixelsize=20e-2
                            kappa  = data[:,2]
                            mass_data[icross_lab] = np.log10((np.sum(kappa[:,rgrid<750.],axis=-1)*critical_kappa*(20*units.kpc)**2).value)

                        else:
                            mass_data[icross_lab] = meta['mass']
                        
                        components[ icross_lab ] = meta['ncomponents']
                        
                    mass = mass_data[icross_lab][ results['indexes'][(icross == results['all_cross_sections'][0])]]
    
                    this_components = components[ icross_lab ][ results['indexes'][(icross == results['all_cross_sections'][0])]]
  
                    if ncomponents is not None:
                        mass_indexes = (mass > mass_cut[0]) & (mass < mass_cut[1]) & (this_components > ncomponents)
                    else:
                        mass_indexes = (mass > mass_cut[0]) & (mass < mass_cut[1])
                elif ncomponents is not None:
                    icross_lab = str(icross)

                    if icross_lab not in list(components.keys()):
                        meta, data = get_dataset_meta( dataset, icross)
                        
                        components[ icross_lab ] = meta['ncomponents']
                               
                    this_components = components[ icross_lab ][ results['indexes'][(icross == results['all_cross_sections'][0])]]
                    
                    mass_indexes =  this_components > ncomponents
  
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

def get_dataset_meta( dataset, icross ):
    if dataset.endswith('vd'):
        pklfile = f"../data/100/shear/vd_{dataset[:-2]}.pkl"
    elif icross != 0:
        pklfile = f"../data/100/shear/{dataset}_{icross:0.1g}.pkl"
    elif dataset in ['flamingo','tng']:
        pklfile = f"../data/100/shear/{dataset}.pkl"
    else:
        pklfile = f"../data/100/shear/{dataset}_cdm.pkl"
    
    return pkl.load(open(pklfile,'rb'))

   
def get_kappa( all_cats, smooth=1, extent=None, correct_for_ngal=False):
#set the resolution of the map 

    npix = 100


    e1_radec, e2_radec = lenspack.utils.bin2d( 
        all_cats['x'], all_cats['y'], 
        v=(2.*all_cats['gamma1'], 2.*all_cats['gamma2']),
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
    if smooth > 0:
        kappa_e_map = gaussian_filter(ke_radec,smooth)
        kappa_b_map = gaussian_filter(kb_radec,smooth)
        return kappa_e_map, kappa_b_map
    else:
        return ke_radec, kb_radec


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
                ax.text( 0.002,means,"A2744 UNCOVER DATA", ha='left',va='bottom', fontsize=12)
                ax.text( 0.002,err[1],f"{iunc}\% Uncertainty", ha='left',va='bottom', fontsize=12)

        if not noise:
            ax.fill_between(  np.logspace(-3,1,100), np.zeros(100)+err[0],  np.zeros(100)+err[1], **fill_args)
            ax.plot( np.logspace(-3,1,100), np.zeros(100)+err[0], '--', color=plot_args['color'])
            ax.plot( np.logspace(-3,1,100), np.zeros(100)+err[1], '--', color=plot_args['color'])
   
    
    if legend:
        ax.legend()
        
def get_latent_space( model_list,
                     bias=[0,0],
                    quiet=False,
                    args=None,
                    targets=['source_val','target_test'],
                    ):
    
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
    args.apply_intrinsic_ell = 1.0
    obs_meta, obs_data = pkl.load(open(f"../data/100/a2744/obs_data_concat.pkl","rb"))
        
    obs_data[0,0,:,:] += bias[0]
    obs_data[0,1,:,:] += bias[1]
  
    
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

            for test_set in targets:
                target_latent = []
                target_cross = []

                for idx, batch_data in enumerate(dataloaders[test_set][0]):

                    data, cross_sections, binary_labels, file_idx, image_idx = batch_data

                    
                    target_latent.append(imodel.backbone(data)[:,:,0,0])
                    target_cross.append(cross_sections.cpu())
                this_latent.append(torch.cat(target_latent))
                this_cross.append(torch.cat(target_cross))



            latent_spaces.append(this_latent)

            all_cross.append(this_cross)

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
        np.log10(positive_mass.value - err_mass),
        np.log10(positive_mass.value + err_mass)
    ]
    
    return np.log10(positive_mass.value), mass_cut

def gp_invert( gp, threshold):
    sigma_grid = np.logspace(-6, 2, 10000)

    X_grid = np.log10(sigma_grid).reshape(-1,1)

    y_pred, y_std = gp.predict(
        X_grid,
    return_std=True
    )

    if isinstance(threshold, float):
        loglike = norm.logpdf(
                threshold,
                loc=y_pred,
                scale=y_std*2
            )
        
    else:
        loglike= []
        for i in threshold.flatten():
            loglike.append(norm.logpdf(
                i,
                loc=y_pred,
                scale=y_std*2
            ))
        loglike = np.array(loglike)
            

    return sigma_grid, loglike

from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    WhiteKernel
)
from sklearn.gaussian_process import GaussianProcessRegressor

def get_gaussian_process():
    
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        *
        Matern(
            length_scale=0.3,
            length_scale_bounds=(1e-2, 10),
            nu=1.5
        )
        +
        WhiteKernel(
            noise_level=1e-3,
            noise_level_bounds=(1e-6, 1e-1)
        )
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-4
    )

    return gp

def thresh_to_cross( thresh, ifilter='concat' ):
    
    models, probs = pkl.load(open("pickles/probs_for_cross_concat_nob1.pkl","rb"))
    models, probabilities, probabilities_noise =pkl.load(open("pickles/model_on_data.pkl","rb"))
    nmodels = probabilities[ifilter].shape[0]

    gp = get_gaussian_process()
    
    all_thresholds = []
    for imodel in range(nmodels):
        #Get the X and y values for a given model
        all_thresholds.append(1.-np.array([ np.mean(probs[i][imodel]) for i in probs.keys()]))

    all_thresholds = np.array(all_thresholds)
    thresholds = np.mean(all_thresholds,axis=0)
    err = np.std(all_thresholds,axis=0)

    cross = np.array([float(i) for i in probs.keys() ])

    cdm_thresh = thresholds[ cross == 0]

    err = err[ cross != 0 ]
    thresholds = thresholds[ cross != 0 ]
    cross = cross[ cross != 0]

    #cross[cross==0] = 1e-3
    cross = np.log10(cross)


    gp.fit(cross.reshape(-1, 1), thresholds)
    
    
    sigma_grid, loglike = gp_invert( gp, thresh )
    
    if isinstance(thresh, float):
        
        return sigma_grid[ np.argmax( loglike ) ]
    else:
        dim = thresh.shape
        return np.array([ sigma_grid[ np.argmax( i ) ] for i in loglike ]).reshape(dim)

    
import umap
import matplotlib.pyplot as plt


def get_density( x,y):

    kde = gaussian_kde(np.vstack([x, y]))

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    xx, yy = np.mgrid[
        xmin:xmax:200j,
        ymin:ymax:200j
    ]

    positions = np.vstack([xx.ravel(), yy.ravel()])
    density = kde(positions).reshape(xx.shape)
    return xx, yy, density


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
    