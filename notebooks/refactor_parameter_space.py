from astropy.modeling.physical_models import NFW
from astropy import units
from astropy.cosmology import Planck18
import numpy as np
from copy import deepcopy as cp
from astropy.constants import G
from matplotlib.lines import Line2D
from glob import glob as glob
from matplotlib import pyplot as plt
from getColorFromRange import colourFromRange

def get_w( velocity, sigma, sigma0=100):
    return velocity * (sigma0 / sigma - 1.)**(-0.5)
    
    
def get_sigma0( velocity, sigma, w=100.):
    return sigma*(1.+(velocity/w)**2)
    
def sigma_vd( velocity, sigma0=3.04, w=100.):
    return sigma0 / (1+velocity**2/w**2)

def convert_param_space( mass, cross_section, w=100., cdm=0.01):
    """
    Convert a cluster simulated in a velocity indpendent 
    space to a velocity dependnet space.
    
    assuming that
    
    sigma = sigma_0 ( 1 + v**2 / w**2 )**-1
    w = 1./sqrt( sigma_0 / sigma - 1 )
    
    and that we are working with clusters so we can assume
    that the velocity > w such that we can fix w since we
    are not sensitive to this any way.
    
    We fix w=560km/s
    
    We use the virial theorem to convert between mass 
    and velocity
    
    inputs :
        mass : either a single float or numpy array of floats 
               of the absolute viiral mass
        cross_section : float or numpy array (matching mass)
               of the cross-sectio jof the cluster
    keywords:
        w : the assumed turnover velocity (assumed to be that of Robertson model)
        cdm : the value for an input cross-section of 0 which is unphysical
    returns
        velocity : float of dimensions matching mass
        sigma_0 : the new parameter, the normalisaiton of the 
                velocity dependnent cross-section
                
    """
    use_crosssection = cp(cross_section) # because if i change it, 
    # it changes the pointed to array
    
    if type(use_crosssection) == np.ndarray:
        use_crosssection[ use_crosssection == 0] = cdm
        
    elif type(use_crosssection) == float:
        if use_crosssection == 0:
            use_crosssection = cdm
    else:
        raise ValueError("Do not recognise the input type")
    
    virial_velocities = velocity_from_mass( mass )
    
    #return the function converting
    return get_sigma0( virial_velocities, use_crosssection )
    

def convert_sidm_label( mass ):
    """
    convert the sidm cluster to a velocity
    """
    
    virial_velocities = velocity_from_mass( mass )
    
    return sigma_vd( virial_velocities )


def velocity_from_mass( mass ):
    
    #if( np.any(mass < 1e6)):
    #    raise ValueError("Some (or all) masses are too low"\
     #                    "make sure you are giving absolute mass")
        
    #Create a bunch of NFW halos since this contains nice 
    #functions that we can use
    if type(mass) == np.ndarray:
        list_of_nfw_objects = [ NFW( mass=i*units.Msun, 
                                 massfactor='virial') 
                           for i in mass ]
    elif type(mass) == float:
         list_of_nfw_objects = [ NFW( mass=mass*units.Msun, 
                                 massfactor='virial') ] 
    #Use its inbult function to find velocities
    virial_velocities = \
        np.array([ i.circular_velocity(
            i.r_virial.to(units.kpc)).to_value() 
                   for i in list_of_nfw_objects ])
    
    return virial_velocities


def mass_from_velocity( velocity, redshift=0.3, overdensity=200.):
    
    '''
    
    
    '''
    rho = Planck18.critical_density(redshift).to(units.Msun/units.km**3)*overdensity
    G_astro = G.to(units.km**3/(units.Msun*units.s**2))
    M = velocity**3 / G_astro**(3./2.)*(1./rho*3./(4.*np.pi))**(1/2.)*(3./5.)**(3./2.)
    
    return M.value


def v_from_m( mass, redshift=0.3, overdensity=200.):
    
    '''
    
    
    '''
    rho = Planck18.critical_density(redshift).to(units.Msun/units.km**3)*overdensity
    G_astro = G.to(units.km**3/(units.Msun*units.s**2))
    
    #velocity_cubed = mass*G_astro**(3./2.)*(1./rho*3./(4.*np.pi))**(1/2.)*(3./5.)**(3./2.)
    velocity_squared = 5/3.*G_astro*mass**(2./3.)*(4./3.*np.pi*rho)**(1./3)
    
    return velocity_squared.value**(1./2.)


marker_names = list(Line2D.markers.keys())

obs_results = {}
for iresults in glob("constraints/*"):
    if 'err' in iresults:
        continue
    
    iname = iresults.split('/')[-1].split('.')[0]
    obs_results[iname] = np.loadtxt(iresults).T


def plot_constraints( ax=None, select_these=None, labels=None ):
    if ax is None:
        ax = plt.gca()
     
    points = np.loadtxt("constraints/err_points.csv")
    lower = np.loadtxt("constraints/err_lower.csv")
    upper = np.loadtxt("constraints/err_upper.csv")
  
    upper_y = upper[ np.argsort(upper[:,0]), 1]
    lower_y = lower[ np.argsort(lower[:,0]), 1]
    points_y = points[ np.argsort(points[:,0]), 1]
    color = colourFromRange([0,len(list(obs_results.keys()))], cmap='jet')

    if select_these is None:
        select_these = list(obs_results.keys())
        

    
    for i, ikey in enumerate(obs_results.keys()):
        
        if ikey not in select_these:
            continue
        if labels == None:
            label = None
        else:
            label=labels[ikey]

        idx_low = [
            np.argmin(np.abs(lower[:,0] - i)) for i in np.atleast_1d(obs_results[ikey][0])
        ]
        idx_upper = [
            np.argmin(np.abs(upper[:,0] - i)) for i in np.atleast_1d(obs_results[ikey][0])
        ]


        upper_err = np.abs(upper[ idx_upper, 1 ] - obs_results[ikey][1] )

        lower_err = np.abs(obs_results[ikey][1] -  lower[ idx_low, 1 ] )
        
        
        ax.errorbar( obs_results[ikey][0], 
                     obs_results[ikey][1], 
                 yerr = [lower_err,
                         upper_err], fmt='o',
                    markersize=8, capsize=4, lw=2,
                    alpha=0.5, color=color[i]
               )
        ax.plot( obs_results[ikey][0], 
                     obs_results[ikey][1], 'o',
                    markersize=8, lw=2,
                    alpha=0.5, color=color[i], label=label
               )
            
        lims = upper_err/obs_results[ikey][1] < 0.2
        ax.errorbar( np.atleast_1d(obs_results[ikey][0])[lims], 
                     np.atleast_1d(obs_results[ikey][1])[lims], 
                 yerr = [lower_err[lims],
                         upper_err[lims]], fmt='o',
                    markersize=8, capsize=4, lw=2,
                    alpha=0.5, color=color[i], 
                    uplims=True,
               )