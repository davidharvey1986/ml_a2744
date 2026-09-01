

from get_model_probabilities import *
from add_shear_to_data import return_error_in_mean
from scipy.stats import binned_statistic, gaussian_kde
from scipy.interpolate import interp1d
from sklearn.ensemble import ExtraTreesRegressor

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



def g1_func( snr, a, b, c, d ):
    return a + b*np.exp( snr /c) + d*snr
    #return a + b*snr + c*snr**2
    #return a+b*snr

def g1_inv( snr, a, b, c, d):
    
    #return np.log((snr - a)/b)*c
    
    z =  b/(c*d)*np.exp((snr-a)/(c*d))
    
    return ( snr - a)/d - c*lambertw(z)
    #return (snr -a)/b
    
def jacob( snr, a, b, c, d):

    z = b/(c*d)*np.exp((snr-a)/(c*d))
    
    deriv = 1. / (d*(1+lambertw(z) ))
    #deriv = c*lambertw(z) / (z*(1+lambertw(z) ))
    return deriv

    #return c/(snr-a)
    #return 1./b
    
    
def interp_invert(
        interp, threshold
    ):
    sigma_grid = np.linspace(-10, 10, 100000)
    
    y_pred = interp(
        sigma_grid
    )

    
    return sigma_grid[ np.argmin( np.abs(threshold - y_pred)) ]
                      

def gp_invert( gp, threshold, smoothing=1):
    sigma_grid = np.logspace(-10,10, 100000)

    X_grid = np.log10(sigma_grid).reshape(-1,1)

    y_pred, y_std = gp.predict(
        X_grid,
    return_std=True
    )

    
    loglike = norm.logpdf(
            threshold,
            loc=y_pred,
            scale=y_std*smoothing
        )
    

    return sigma_grid, loglike

def get_probs_for_cross(
        results_file,
        filter_name = 'concat',
        get_mass_weights=False,
    ):



    if isinstance(results_file, str):
        if not os.path.isfile( results_file ):
            raise ValueError("Cant find results file")
        all_results = pkl.load(open(results_file,"rb"))
    else:
        all_results = results_file



    domain = {
        'tgt':'darkskies_obs',
        'src':'bahamas_obs'
    }
    model_index = {}

    probs_for_cross = {}
    for imx, imodel in tqdm(enumerate(all_results.keys())):

        imodel_probs = {}
        imodel_index = {}
        if isinstance(all_results[imodel], dict):
            domain = {
                'tgt':'darkskies_obs',
                'src':'bahamas_obs'
            }
            for target in all_results[imodel].keys():
                tgt = get_threshold_for_cross( 
                        all_results[imodel][target], 
                        quiet=False)

                for idx, icross in enumerate(tgt['cross_sections']):
                    keyname = f"{icross:0.2f}"
                    if keyname not in imodel_probs.keys():
                        imodel_probs[ keyname ] = []
                        imodel_index[ keyname ] = []

                    imodel_probs[keyname].append(
                        tgt['probabilities'][idx][:,0]
                    )
                    imodel_index[keyname].append(tgt['indexes'][idx])
        else:

            tgt = get_threshold_for_cross( 
                    all_results[imodel], 
                    quiet=False)
            
            for idx, icross in enumerate(tgt['cross_sections']):
                keyname = f"{icross:0.2f}"
                if keyname not in imodel_probs.keys():
                    imodel_probs[ keyname ] = []
                    imodel_index[ keyname ] = []

                imodel_probs[keyname].append(
                    tgt['probabilities'][idx][:,0]
                )
                imodel_index[keyname].append(tgt['indexes'][idx])
                

        for ikey in imodel_probs.keys():
            if ikey not in probs_for_cross:
                probs_for_cross[ikey] = []
                model_index[ikey] = []


            model_index[ikey].append(np.hstack(imodel_index[ikey]))
            probs_for_cross[ikey].append(np.hstack(imodel_probs[ikey]))

    return probs_for_cross


    
def infer_sidm_fct(
        probabilities,
        probs_for_cross,
    ):
       

    thresholds = np.array([ probs_for_cross[i] for i in probs_for_cross.keys()])

    cross = np.array([float(i) for i in probs_for_cross.keys() ])


    cross[cross==0] = 1e-3
    cross = np.log10(cross)


    ## CURVE FITTING
    popt, pcov = curve_fit(
                g1_func,
                cross,
                thresholds,
                bounds=(np.array([-np.inf,-np.inf,-np.inf,0.01]),
                        np.array([np.inf, np.inf, np.inf, np.inf])
                        ))

    estimates  =  np.array([ g1_inv(  1-iest, *popt ) for iest in probabilities]).real
    estimates[ estimates < -5] = -5 # Im not more sensitive than this

    return estimates

def infer_sidm(
        probabilities,
        probs_for_cross,
        jwst_filter='concat',
        return_all=False,
        smoothing=2.,
        statistic='mean'
    ):
 
    thresholds = np.array([ probs_for_cross[i] for i in probs_for_cross.keys()])

    cross = np.array([float(i) for i in probs_for_cross.keys() ])

    cross[cross==0] = 1e-3
    cross = np.log10(cross)

    gp.fit(cross.reshape(-1, 1), thresholds)

    estimates = []
    pdfs = []
    for iest in probabilities:
        obs_thresh =  1-iest
        xpdf, obs_loglike  =  gp_invert( gp, obs_thresh, smoothing=smoothing )

        likelihood = np.exp(obs_loglike)

        ypdf  = likelihood / np.trapz(
            likelihood,
            x=np.log10(xpdf)
        )

        ### Final constraints
        max_like = xpdf[np.argmax(ypdf)]

        estimates.append(np.log10(max_like))
        
        pdfs.append({'x':xpdf,'y':ypdf})

    obsstd = return_error_in_mean(np.array(estimates))
    
    if return_all:
        return {'gp':gp, 'estimates':np.array(estimates),
               'pdfs':pdfs}
    else:
        kde = gaussian_kde( estimates, obsstd )
        xpdf = np.linspace(-10,10,1000)
        ypdf = norm.pdf(xpdf, np.mean(estimates),
                    return_error_in_mean(estimates))

        #max_like = np.sum(ypdf*xpdf)/np.sum(ypdf)
        max_like = xpdf[np.argmax(ypdf)]
        
        cdf = cumulative_trapezoid(
            ypdf,
            xpdf,
            initial=0
        )

        cdf /= cdf[-1]

        low_value = 0.16 #cdf[ np.argmax(ypdf)] - 0.34
        hig_value = 0.84 #cdf[ np.argmax(ypdf)] + 0.34


        low = np.interp(
            low_value,
            cdf,
            xpdf
        )
        high = np.interp(
            hig_value,
            cdf,
            xpdf
        )

        error = np.array([max_like - low, high - max_like])
    
        if statistic == 'mean':
            return_est = np.mean(estimates)
        elif statistic == 'max_like':
            return_est = max_like
        elif statistic == 'median':
            return_est = np.median(estimates)
        else:
            raise ValueError("Unknown statistic")
            
        return return_est, error
    


def infer_sidm_interp(
        probabilities,
        probs_for_cross,
        jwst_filter='concat'
    ):

    
    thresholds = np.array([ probs_for_cross[i] for i in probs_for_cross.keys()])

    cross = np.array([float(i) for i in probs_for_cross.keys() ])


    cross[cross==0] = 1e-3
    cross = np.log10(cross)

    interp = interp1d(
        cross,
        thresholds,
        fill_value="extrapolate"
    )
        
    estimates  =  np.array([ interp_invert( interp, 1-iest ) for iest in probabilities])
    
    estimates[ estimates < -5] = -5 # Im not more sensitive than this
    return estimates



def infer_sidm_latent(
    latent_space_file='pickles/latent.pkl',
    regressor_pkl='pickles/regressors.pkl',
    regressor=None
    ):
    
    if os.path.isfile(regressor_pkl):
        return np.log10(pkl.load(open(regressor_pkl,'rb'))).flatten()
    
    latent = pkl.load(open("pickles/latent.pkl","rb"))
    
    if regressor is None:
        regressor = ExtraTreesRegressor
        
    estimates = []
    
    for imodel in tqdm(range(30)):
        X =  latent['latent_space'][1+imodel][0].detach().numpy()
        y =  latent['all_cross_sections'][ 1+imodel ][0] 
        
        svr = regressor()

        svr.fit(X, y)
        
        estimates.append(svr.predict( latent['latent_space'][0][imodel].detach().numpy().reshape(1, -1) ))
        
    pkl.dump(np.array(estimates),open(regressor_pkl,'wb'))
    
    return np.log10(np.array(estimates).flatten())


    
    