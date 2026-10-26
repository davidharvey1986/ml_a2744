# Copied from Dark_Matter_DA/code/notebooks/component_extractor.py; the SExtractor
# configuration (wavdetectpy.sex, py.param, gauss_5.0_9x9.conv) is in sex_files/.
from RRGtools import pysex as sex
from scipy.ndimage import gaussian_filter as gauss_filter
from astropy.io import fits 
from matplotlib import pyplot as plt
import numpy as np
from numpy.lib.recfunctions import append_fields as append_rec
import os as os
from lenspack.image.inversion import ks93inv
from scipy.interpolate import RectBivariateSpline
from astropy.cosmology import Planck18
from lenspack.utils import sigma_critical

def component_extractor( image, smoothing=None, \
                        filtername='gauss_5.0_9x9.conv', 
                        conf_file='wavdetectpy.sex',
                        conf_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sex_files'),
                        use_sex=True, 
                        **conf_args):
    '''
    This is a general script that should extract a given
    mass component from the simulations.   

    
    '''

    if smoothing is not None:
        smoothed_image = gauss_filter( image, smoothing )
    else:
        smoothed_image = image
        
 #   pixel_size_degrees = pixel_size_kpc/(lensing.ang_distance( redshift )*1e3)*180./np.pi
    
    #Create a random string to component find
    if use_sex:
        file_name = 'component_finding_'+str(np.random.randint(0,1000001))+'.fits'
        fits.writeto( file_name, smoothed_image, overwrite=True )

        hot_conf = "%s/%s" % (conf_path, conf_file)
        param_file="%s/py.param" % conf_path

        filtername = conf_path+'/'+filtername

        default_conf_args = {'WEIGHT_TYPE':'NONE',
                     'FILTER_NAME':filtername,
                        'FILTER':'YES',
                    'VERBOSE_TYPE':'QUIET'}

        for iarg in default_conf_args.keys():
            if iarg not in conf_args.keys():
                conf_args[iarg] = default_conf_args[iarg]

        sources = sex.run(  
            file_name,   
            conf_file=hot_conf, 
            conf_args=conf_args, 
            param_file=param_file
        )
        os.system('rm -fr '+file_name)
 
    else:
        if not 'xgrid' in conf_args.keys():
            conf_args['xgrid'], conf_args['ygrid'] = np.meshgrid( 
                np.arange(image.shape[0])-image.shape[0]/2.,
                np.arange(image.shape[1])-image.shape[0]/2.
            )

        x_center = np.sum( image*conf_args['xgrid'] ) / np.sum(image)
        y_center = np.sum( image*conf_args['ygrid'] ) / np.sum(image)
        
        sources = np.array( (x_center, y_center), 
                           dtype=[('X_image',float), ('Y_image', float)]
                          )
        

#    ra = sources['X_IMAGE']*pixel_size_degrees
#    dec = sources['Y_IMAGE']*pixel_size_degrees
#    sources = append_rec( sources, 'RA', ra, usemask=False, asrecarray=True)
#    sources = append_rec( sources, 'DEC', dec)

    
    
    return sources

    
    
    
def model_free( 
    data, 
    bcg_pos, 
    random_sample=False,
    rcut=[100],
    ngal=10000,
    sigma_e=0.26,
    zl=0.23,
    zs=0.8
    ):
    
    '''
    
    Use the tangential shear around the BCG to find the center of the halo
    '''
    critical_mass = sigma_critical( zl, zs, Planck18).to("Msun/Mpc**2")

    kappa = data / critical_mass
    if kappa.unit != "":
        raise ValueError("Kappa should be unitless buit is %s" % kappa.unit)
        
    xvec = np.arange(data.shape[0]) - bcg_pos[0]
    yvec =  np.arange(data.shape[0]) - bcg_pos[1]

    xg, yg = np.meshgrid( xvec, yvec) 
    
    g1_grid, g2_grid = ks93inv( kappa, kappa*0.)
    
    g1_grid = g1_grid / (1.-kappa)
    g2_grid = g2_grid / (1.-kappa)
  
    
    random_g1 = np.random.normal(0., sigma_e/np.sqrt(2.))
    random_g2 = np.random.normal(0., sigma_e/np.sqrt(2.))
    
    random_x = np.random.uniform( 0, data.shape[0], ngal) - bcg_pos[0]
    random_y = np.random.uniform( 0, data.shape[0], ngal) - bcg_pos[1]
    
    random_r = np.sqrt( 
        random_x**2 +
        random_y**2
    )

    g1_fct = RectBivariateSpline( 
        xvec, 
        yvec,
        g1_grid, kx=3, ky=3
    )
    
    g2_fct = RectBivariateSpline( 
        xvec, 
        yvec,
        g2_grid, kx=3, ky=3
    )
    
    g1 = g1_fct( random_x, random_y, grid=False) + random_g1
    g2 = g2_fct( random_x, random_y, grid=False) + random_g2

    phi = np.arctan2( 
            random_y, 
            random_x
        )+np.pi/2.
        
    gamma_t = -g1*np.cos(2.*phi) \
               + g2*np.sin(2.*phi)
    

    offsets = []
    all_gamma_t = []
    for ircut in rcut:
    
        x = np.sum( 
            gamma_t[ random_r < ircut ]
            *random_x[ random_r < ircut ] 
        ) / np.sum(gamma_t[ random_r < ircut ])
        y = np.sum( 
            gamma_t[ random_r < ircut ]*
            random_y[ random_r < ircut ] 
        ) / np.sum(gamma_t[ random_r < ircut ])
        offsets.append([x,y])
        all_gamma_t.append(gamma_t[ random_r < ircut ])
        
    return {'offsets': np.array(offsets), 
            'gamma_t': all_gamma_t}